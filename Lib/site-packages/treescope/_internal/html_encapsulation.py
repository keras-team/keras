# Copyright 2024 The Treescope Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Outer runner to manage interacting HTML outputs in IPython notebooks.

When rendering HTML outputs in IPython notebooks, we want to support:

- Compression, so that large repetitive HTML outputs do not cause excessively
  large notebook sizes,
- Streaming output, where the structure of an output can be sent to the
  browser while NDArrays and other contents are still being loaded and
  rendered in the IPython kernel,
- Final output stealing (for streaming output mode), where the temporary
  rendering can be moved into the final "result" display cell in IPython
  systems that show Out[...] markers,
- Duplication-safety, so that if the same output is repeated multiple times in
  a notebook display environment (e.g. in a separate view-only cell), each copy
  is rendered independently,
- Shadow-root encapsulation, where the rendering is contained in the "shadow
  DOM". This keeps its JS/CSS separate from the main notebook, prevents
  interference between different renderings, and may allow rendering
  optimizations in some browsers.

This module defines the necessary logic to enable these features. It takes as
input a sequence of HTML blobs that should be rendered into a single container
in a streaming fashion, and transforms them into a new sequence of compressed,
duplication-safe HTML blobs that first create the container and then render each
of the contents into it.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import io
from typing import Iterable, Iterator
import uuid
import zlib


class SegmentType(enum.Enum):
  """Type of HTML segment."""

  # The blob that creates and owns the container.
  CONTAINER = enum.auto()
  # A blob that inserts new data into the initial container, but doesn't display
  # any content itself.
  CONTAINER_UPDATE = enum.auto()
  # A final blob that will take ownership of the container, moving the container
  # into itself. Only used when stealing is enabled.
  FINAL_OUTPUT_STEALER = enum.auto()


@dataclasses.dataclass(frozen=True)
class HTMLOutputSegment:
  """A piece of partial HTML output."""

  html_src: str
  segment_type: SegmentType


CONTAINER_TEMPLATE = """
<script>
(()=>{
  if (customElements.get('treescope-container') === undefined) {
    // Custom container element that contains the rendering and also owns
    // JavaScript definitions for the rendering. Intentionally does not contain
    // fixed implementations of those definitions, because in some notebook
    // environments (e.g. JupyterLab) every notebook output exists in the global
    // browser window namespace, and we can't have multiple versions of the
    // same custom element.
    class TreescopeContainer extends HTMLElement {
      constructor() {
        super();
        this.attachShadow({mode: "open"});
        this.defns = {};
        this.state = {};
      }
    }
    customElements.define("treescope-container", TreescopeContainer);
  }
  if (customElements.get('treescope-run-here') === undefined) {
    // Custom "run-here" element that runs a script in the context of a DOM
    // node. This element should contain a single script tag with the attribute
    // type="application/octet-stream" that contains the script to run. The
    // script will be executed inside an anonymous function, with `this` bound
    // to the DOM node. Afterward, the element will be removed.
    class RunHere extends HTMLElement {
      constructor() { super() }
      connectedCallback() {
        const run = child => {
          const fn = new Function(child.textContent);
          child.textContent = "";
          fn.call(this);
          this.remove();
        };
        const child = this.querySelector("script");
        if (child) {
          run(child);
        } else {
          // In some cases, the connected callback may fire before the script
          // tag has actually been added; this can occur when loading from an
          // iframe srcdoc attribute. In this case, we need to wait for the
          // script tag to appear. We assume the only mutation that will occur
          // is adding the script tag, since the element is only supposed to
          // contain a single script tag.
          new MutationObserver(()=>{
            run(this.querySelector("script"));
          }).observe(this, {childList: true});
        }
      }
    }
    customElements.define("treescope-run-here", RunHere);
  }
})();
</script>
<treescope-container class="{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}"
style="display:block"></treescope-container>
<treescope-run-here><script type="application/octet-stream">
// Find the target container based on its ID class. We use a class instead of
// an HTML id because it's possible that the notebook display system will
// display this multiple times. Using the suffix, we can make sure to only
// return each container once for each suffix, so that if we duplicate the
// container and also a script that modifies it, each script will modify a
// different container.
const root = (
  Array.from(document.getElementsByClassName(
      "{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}"))
  .filter((elt) => !elt.dataset.setup)
)[0];
root.dataset.setup = 1;
const msg = document.createElement("span");
msg.style = "color: #cccccc; font-family: monospace;";
msg.textContent = "(Loading...)";
root.state.loadingMsg = msg;
root.shadowRoot.appendChild(msg);
root.state.chain = new Promise((resolve, reject) => {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        resolve();
        observer.disconnect();
        return;
      }
    }
  }, {rootMargin: "1000px"});
  window.setTimeout(() => {
    observer.observe(root);
  }, 0);
});
root.state.deferring = false;
const _insertNode = (node) => {
  // Script trick: Script tags that were inserted using innerHTML don't
  // execute when they get added to the document. The scripts in the
  // templates may have been inserted this way, which would prevent them
  // from executing in the root. To get them to execute, we need to
  // rebuild the scripts before inserting them.
  for (let oldScript of node.querySelectorAll("script")) {
      let newScript = document.createElement("script");
      newScript.type = oldScript.type;
      newScript.textContent = oldScript.textContent;
      oldScript.parentNode.replaceChild(newScript, oldScript);
  }
  if (root.state.loadingMsg) {
    root.state.loadingMsg.remove();
    root.state.loadingMsg = null;
  }
  root.shadowRoot.appendChild(node);
};
root.defns.insertContent = ((contentNode, compressed) => {
  if (compressed) {
    root.state.deferring = true;
  }
  if (root.state.deferring) {
    root.state.chain = (async () => {
      await root.state.chain;
      if (compressed) {
        // contentNode is a script.
        const encoded = contentNode.textContent;
        const blob = new Blob([
            Uint8Array.from(atob(encoded), (m) => m.codePointAt(0))
        ]);
        const reader = blob.stream().pipeThrough(
          new DecompressionStream("deflate")
        ).pipeThrough(
          new TextDecoderStream("utf-8")
        ).getReader();
        const parts = [];
        while (true) {
          const step = await reader.read();
          if (step.done) { break; }
          parts.push(step.value);
        }
        const tpl = document.createElement('template');
        tpl.innerHTML = parts.join("");
        _insertNode(tpl.content);
      } else {
        // contentNode is a template.
        _insertNode(contentNode.content);
      }
    })();
  } else {
    // contentNode is a template.
    _insertNode(contentNode.content);
  }
});
</script></treescope-run-here>
"""

STEP_TEMPLATE = """
<div style="display:none">
<template>{__REPLACE_ME_WITH_CONTENT_HTML__}</template>
<treescope-run-here><script type="application/octet-stream">
const root = (
  Array.from(document.getElementsByClassName(
      "{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}"))
  .filter((elt) => !elt.dataset['step{__REPLACE_ME_WITH_STEP__}'])
)[0];
root.dataset['step{__REPLACE_ME_WITH_STEP__}'] = 1;
root.defns.insertContent(this.parentNode.querySelector('template'), false);
this.parentNode.remove();
</script></treescope-run-here>
</div>
"""

COMPRESSED_STEP_TEMPLATE = """
<div style="display:none">
<script type="application/octet-stream"
>{__REPLACE_ME_WITH_COMPRESSED_CONTENT_HTML__}</script>
<treescope-run-here><script type="application/octet-stream">
const root = (
  Array.from(document.getElementsByClassName(
      "{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}"))
  .filter((elt) => !elt.dataset['step{__REPLACE_ME_WITH_STEP__}'])
)[0];
root.dataset['step{__REPLACE_ME_WITH_STEP__}'] = 1;
root.defns.insertContent(
    this.parentNode.querySelector('script[type="application/octet-stream"]'),
    true
);
this.parentNode.remove();
</script></treescope-run-here>
</div>
"""

STEALER_TEMPLATE = """
<treescope-run-here><script type="application/octet-stream">
const root = (
  Array.from(document.getElementsByClassName(
      "{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}"))
  .filter((elt) => !elt.dataset.stolen)
)[0];
root.dataset.stolen = 1;
// Some notebook environments (in particular, VSCode's embedded notebook)
// will hide outputs that are empty, but moving elements between different
// positions can confuse it. To avoid this, we add and remove small amounts of
// padding to trigger resize detection.
// 1. Insert temporary element before the stolen element in the old cell.
const temp = document.createElement("div");
temp.style = "height: 1px; width: 1px;";
root.parentNode.insertBefore(temp, root);
// 2. Move the stolen element into this cell.
this.parentNode.replaceChild(root, this);
// 3. Remove the temporary element.
temp.remove();
// 4. Add padding to the stolen element, now that it is in this cell.
root.style.paddingTop = "1px";
</script></treescope-run-here>
"""


def _prep_html_js_and_strip_comments(src):
  stream = io.StringIO()
  for line in src.splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("//"):
      stream.write(stripped)
      stream.write(" ")
  return stream.getvalue()[:-1]


def encapsulate_streaming_html(
    inner_iterator: Iterable[str],
    *,
    compress: bool = True,
    stealable: bool = False,
) -> Iterator[HTMLOutputSegment]:
  """Encapsulates a sequence of inner HTML blobs into robust iframe updates.

  This function accepts an iterator of HTML blobs that should each run in the
  same iframe, and transforms them into another iterator of HTML blobs that
  can be inserted directly into a notebook environment. The first output will
  set up the iframe, and all later updates will insert content into that
  original iframe. Optionally, the final output will be a "stealer" that will
  move the iframe into itself, ensuring that the iframe is associated with the
  correct "result" cell in IPython notebook systems that show Out[...] markers.
  Updates will be duplication-safe, in the sense that repeating the same
  sequence of outputs in a single HTML page will produce multiple copies of the
  iframe, each with the same contents, and the code in the inner iterator will
  only be executed once in each iframe.

  Args:
    inner_iterator: A nonempty iterator of HTML blobs to encapsulate.
    compress: Whether to compress the HTML blobs.
    stealable: Whether to include a final "stealer" blob that will move the
      iframe into itself.

  Yields:
    HTML output segments that can be displayed in a notebook environment or
    saved.
  """
  inner_iterator = iter(inner_iterator)

  stream = io.StringIO()
  # Build the initial iframe, and assign it a unique ID.
  # Note: This is unique in the Python program, but if the output is repeated
  # multiple times in the notebook output, we may have multiple iframes with
  # the same ID.
  unique_id_class = f"treescope_out_{uuid.uuid4().hex}"

  outer_content = _prep_html_js_and_strip_comments(CONTAINER_TEMPLATE).replace(
      "{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}", unique_id_class
  )
  stream.write(outer_content)

  for i, step_content in enumerate(inner_iterator):
    if compress:
      # Compress the input string. We use ZLIB, which is natively supported by
      # modern browsers.
      compressed = zlib.compress(
          step_content.encode("utf-8"), zlib.Z_BEST_COMPRESSION
      )
      # Serialize it as base64.
      serialized = base64.b64encode(compressed).decode("ascii")
      # Embed it.
      step_content = (
          _prep_html_js_and_strip_comments(COMPRESSED_STEP_TEMPLATE)
          .replace("{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}", unique_id_class)
          .replace("{__REPLACE_ME_WITH_STEP__}", str(i))
          .replace("{__REPLACE_ME_WITH_COMPRESSED_CONTENT_HTML__}", serialized)
      )
    else:
      step_content = (
          _prep_html_js_and_strip_comments(STEP_TEMPLATE)
          .replace("{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}", unique_id_class)
          .replace("{__REPLACE_ME_WITH_STEP__}", str(i))
          .replace("{__REPLACE_ME_WITH_CONTENT_HTML__}", step_content)
      )
    stream.write(step_content)
    if i == 0:
      segment_type = SegmentType.CONTAINER
    else:
      segment_type = SegmentType.CONTAINER_UPDATE
    yield HTMLOutputSegment(
        html_src=stream.getvalue(), segment_type=segment_type
    )
    stream = io.StringIO()

  if stealable:
    stealer_content = _prep_html_js_and_strip_comments(
        STEALER_TEMPLATE
    ).replace("{__REPLACE_ME_WITH_CONTAINER_ID_CLASS__}", unique_id_class)
    yield HTMLOutputSegment(
        html_src=stealer_content,
        segment_type=SegmentType.FINAL_OUTPUT_STEALER,
    )


def encapsulate_html(html_src: str, compress: bool = True) -> str:
  """Encapsulates HTML source code into a duplication-safe container.

  Args:
    html_src: The HTML source code to encapsulate.
    compress: Whether to compress the HTML source code.

  Returns:
    An HTML output segment that can be displayed in a notebook environment or
    saved.
  """
  [converted] = encapsulate_streaming_html(
      [html_src], compress=compress, stealable=False
  )
  return converted.html_src
