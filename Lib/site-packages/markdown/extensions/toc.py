# Table of Contents Extension for Python-Markdown
# ===============================================

# See https://Python-Markdown.github.io/extensions/toc
# for documentation.

# Original code Copyright 2008 [Jack Miller](https://codezen.org/)

# All changes Copyright 2008-2024 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Add table of contents support to Python-Markdown.

See the [documentation](https://Python-Markdown.github.io/extensions/toc)
for details.
"""

from __future__ import annotations

from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """ Slugify a string, to make it URL friendly. """
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[{}\s]+'.format(separator), separator, value)


def slugify_unicode(value: str, separator: str) -> str:
    """ Slugify a string, to make it URL friendly while preserving Unicode characters. """
    return slugify(value, separator, unicode=True)


IDCOUNT_RE = re.compile(r'^(.*)_([0-9]+)$')


def unique(id: str, ids: MutableSet[str]) -> str:
    """ Ensure id is unique in set of ids. Append '_1', '_2'... if not """
    while id in ids or not id:
        m = IDCOUNT_RE.match(id)
        if m:
            id = '%s_%d' % (m.group(1), int(m.group(2))+1)
        else:
            id = '%s_%d' % (id, 1)
    ids.add(id)
    return id


@deprecated('Use `render_inner_html` and `striptags` instead.')
def get_name(el: etree.Element) -> str:
    """Get title name."""

    text = []
    for c in el.itertext():
        if isinstance(c, AtomicString):
            text.append(html.unescape(c))
        else:
            text.append(c)
    return ''.join(text).strip()


@deprecated('Use `run_postprocessors`, `render_inner_html` and/or `striptags` instead.')
def stashedHTML2text(text: str, md: Markdown, strip_entities: bool = True) -> str:
    """ Extract raw HTML from stash, reduce to plain text and swap with placeholder. """
    def _html_sub(m: re.Match[str]) -> str:
        """ Substitute raw html with plain text. """
        try:
            raw = md.htmlStash.rawHtmlBlocks[int(m.group(1))]
        except (IndexError, TypeError):  # pragma: no cover
            return m.group(0)
        # Strip out tags and/or entities - leaving text
        res = re.sub(r'(<[^>]+>)', '', raw)
        if strip_entities:
            res = re.sub(r'(&[\#a-zA-Z0-9]+;)', '', res)
        return res

    return HTML_PLACEHOLDER_RE.sub(_html_sub, text)


def unescape(text: str) -> str:
    """ Unescape Markdown backslash escaped text. """
    c = UnescapeTreeprocessor()
    return c.unescape(text)


def strip_tags(text: str) -> str:
    """ Strip HTML tags and return plain text. Note: HTML entities are unaffected. """
    # A comment could contain a tag, so strip comments first
    while (start := text.find('<!--')) != -1 and (end := text.find('-->', start)) != -1:
        text = f'{text[:start]}{text[end + 3:]}'

    while (start := text.find('<')) != -1 and (end := text.find('>', start)) != -1:
        text = f'{text[:start]}{text[end + 1:]}'

    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def escape_cdata(text: str) -> str:
    """ Escape character data. """
    if "&" in text:
        # Only replace & when not part of an entity
        text = RE_AMP.sub('&amp;', text)
    if "<" in text:
        text = text.replace("<", "&lt;")
    if ">" in text:
        text = text.replace(">", "&gt;")
    return text


def run_postprocessors(text: str, md: Markdown) -> str:
    """ Run postprocessors from Markdown instance on text. """
    for pp in md.postprocessors:
        text = pp.run(text)
    return text.strip()


def render_inner_html(el: etree.Element, md: Markdown) -> str:
    """ Fully render inner html of an `etree` element as a string. """
    # The `UnescapeTreeprocessor` runs after `toc` extension so run here.
    text = unescape(md.serializer(el))

    # strip parent tag
    start = text.index('>') + 1
    end = text.rindex('<')
    text = text[start:end].strip()

    return run_postprocessors(text, md)


def remove_fnrefs(root: etree.Element) -> etree.Element:
    """ Remove footnote references from a copy of the element, if any are present. """
    # Remove footnote references, which look like this: `<sup id="fnref:1">...</sup>`.
    # If there are no `sup` elements, then nothing to do.
    if next(root.iter('sup'), None) is None:
        return root
    root = deepcopy(root)
    # Find parent elements that contain `sup` elements.
    for parent in root.findall('.//sup/..'):
        carry_text = ""
        for child in reversed(parent):  # Reversed for the ability to mutate during iteration.
            # Remove matching footnote references but carry any `tail` text to preceding elements.
            if child.tag == 'sup' and child.get('id', '').startswith('fnref'):
                carry_text = f'{child.tail or ""}{carry_text}'
                parent.remove(child)
            elif carry_text:
                child.tail = f'{child.tail or ""}{carry_text}'
                carry_text = ""
        if carry_text:
            parent.text = f'{parent.text or ""}{carry_text}'
    return root


def nest_toc_tokens(toc_list):
    """Given an unsorted list with errors and skips, return a nested one.

        [{'level': 1}, {'level': 2}]
        =>
        [{'level': 1, 'children': [{'level': 2, 'children': []}]}]

    A wrong list is also converted:

        [{'level': 2}, {'level': 1}]
        =>
        [{'level': 2, 'children': []}, {'level': 1, 'children': []}]
    """

    ordered_list = []
    if len(toc_list):
        # Initialize everything by processing the first entry
        last = toc_list.pop(0)
        last['children'] = []
        levels = [last['level']]
        ordered_list.append(last)
        parents = []

        # Walk the rest nesting the entries properly
        while toc_list:
            t = toc_list.pop(0)
            current_level = t['level']
            t['children'] = []

            # Reduce depth if current level < last item's level
            if current_level < levels[-1]:
                # Pop last level since we know we are less than it
                levels.pop()

                # Pop parents and levels we are less than or equal to
                to_pop = 0
                for p in reversed(parents):
                    if current_level <= p['level']:
                        to_pop += 1
                    else:  # pragma: no cover
                        break
                if to_pop:
                    levels = levels[:-to_pop]
                    parents = parents[:-to_pop]

                # Note current level as last
                levels.append(current_level)

            # Level is the same, so append to
            # the current parent (if available)
            if current_level == levels[-1]:
                (parents[-1]['children'] if parents
                 else ordered_list).append(t)

            # Current level is > last item's level,
            # So make last item a parent and append current as child
            else:
                last['children'].append(t)
                parents.append(last)
                levels.append(current_level)
            last = t

    return ordered_list


class TocTreeprocessor(Treeprocessor):
    """ Step through document and build TOC. """

    def __init__(self, md: Markdown, config: dict[str, Any]):
        super().__init__(md)

        self.marker: str = config["marker"]
        self.title: str = config["title"]
        self.base_level = int(config["baselevel"]) - 1
        self.slugify = config["slugify"]
        self.sep = config["separator"]
        self.toc_class = config["toc_class"]
        self.title_class: str = config["title_class"]
        self.use_anchors: bool = parseBoolValue(config["anchorlink"])
        self.anchorlink_class: str = config["anchorlink_class"]
        self.use_permalinks = parseBoolValue(config["permalink"], False)
        if self.use_permalinks is None:
            self.use_permalinks = config["permalink"]
        self.permalink_class: str = config["permalink_class"]
        self.permalink_title: str = config["permalink_title"]
        self.permalink_leading: bool | None = parseBoolValue(config["permalink_leading"], False)
        self.header_rgx = re.compile("[Hh][123456]")
        if isinstance(config["toc_depth"], str) and '-' in config["toc_depth"]:
            self.toc_top, self.toc_bottom = [int(x) for x in config["toc_depth"].split('-')]
        else:
            self.toc_top = 1
            self.toc_bottom = int(config["toc_depth"])

    def iterparent(self, node: etree.Element) -> Iterator[tuple[etree.Element, etree.Element]]:
        """ Iterator wrapper to get allowed parent and child all at once. """

        # We do not allow the marker inside a header as that
        # would causes an endless loop of placing a new TOC
        # inside previously generated TOC.
        for child in node:
            if not self.header_rgx.match(child.tag) and child.tag not in ['pre', 'code']:
                yield node, child
                yield from self.iterparent(child)

    def replace_marker(self, root: etree.Element, elem: etree.Element) -> None:
        """ Replace marker with elem. """
        for (p, c) in self.iterparent(root):
            text = ''.join(c.itertext()).strip()
            if not text:
                continue

            # To keep the output from screwing up the
            # validation by putting a `<div>` inside of a `<p>`
            # we actually replace the `<p>` in its entirety.

            # The `<p>` element may contain more than a single text content
            # (`nl2br` can introduce a `<br>`). In this situation, `c.text` returns
            # the very first content, ignore children contents or tail content.
            # `len(c) == 0` is here to ensure there is only text in the `<p>`.
            if c.text and c.text.strip() == self.marker and len(c) == 0:
                for i in range(len(p)):
                    if p[i] == c:
                        p[i] = elem
                        break

    def set_level(self, elem: etree.Element) -> None:
        """ Adjust header level according to base level. """
        level = int(elem.tag[-1]) + self.base_level
        if level > 6:
            level = 6
        elem.tag = 'h%d' % level

    def add_anchor(self, c: etree.Element, elem_id: str) -> None:
        anchor = etree.Element("a")
        anchor.text = c.text
        anchor.attrib["href"] = "#" + elem_id
        anchor.attrib["class"] = self.anchorlink_class
        c.text = ""
        for elem in c:
            anchor.append(elem)
        while len(c):
            c.remove(c[0])
        c.append(anchor)

    def add_permalink(self, c: etree.Element, elem_id: str) -> None:
        permalink = etree.Element("a")
        permalink.text = ("%spara;" % AMP_SUBSTITUTE
                          if self.use_permalinks is True
                          else self.use_permalinks)
        permalink.attrib["href"] = "#" + elem_id
        permalink.attrib["class"] = self.permalink_class
        if self.permalink_title:
            permalink.attrib["title"] = self.permalink_title
        if self.permalink_leading:
            permalink.tail = c.text
            c.text = ""
            c.insert(0, permalink)
        else:
            c.append(permalink)

    def build_toc_div(self, toc_list: list) -> etree.Element:
        """ Return a string div given a toc list. """
        div = etree.Element("div")
        div.attrib["class"] = self.toc_class

        # Add title to the div
        if self.title:
            header = etree.SubElement(div, "span")
            if self.title_class:
                header.attrib["class"] = self.title_class
            header.text = self.title

        def build_etree_ul(toc_list: list, parent: etree.Element) -> etree.Element:
            ul = etree.SubElement(parent, "ul")
            for item in toc_list:
                # List item link, to be inserted into the toc div
                li = etree.SubElement(ul, "li")
                link = etree.SubElement(li, "a")
                link.text = item.get('name', '')
                link.attrib["href"] = '#' + item.get('id', '')
                if item['children']:
                    build_etree_ul(item['children'], li)
            return ul

        build_etree_ul(toc_list, div)

        if 'prettify' in self.md.treeprocessors:
            self.md.treeprocessors['prettify'].run(div)

        return div

    def run(self, doc: etree.Element) -> None:
        # Get a list of id attributes
        used_ids = set()
        for el in doc.iter():
            if "id" in el.attrib:
                used_ids.add(el.attrib["id"])

        toc_tokens = []
        for el in doc.iter():
            if isinstance(el.tag, str) and self.header_rgx.match(el.tag):
                self.set_level(el)
                innerhtml = render_inner_html(remove_fnrefs(el), self.md)
                name = strip_tags(innerhtml)

                # Do not override pre-existing ids
                if "id" not in el.attrib:
                    el.attrib["id"] = unique(self.slugify(html.unescape(name), self.sep), used_ids)

                data_toc_label = ''
                if 'data-toc-label' in el.attrib:
                    data_toc_label = run_postprocessors(unescape(el.attrib['data-toc-label']), self.md)
                    # Overwrite name with sanitized value of `data-toc-label`.
                    name = escape_cdata(strip_tags(data_toc_label))
                    # Remove the data-toc-label attribute as it is no longer needed
                    del el.attrib['data-toc-label']

                if int(el.tag[-1]) >= self.toc_top and int(el.tag[-1]) <= self.toc_bottom:
                    toc_tokens.append({
                        'level': int(el.tag[-1]),
                        'id': el.attrib["id"],
                        'name': name,
                        'html': innerhtml,
                        'data-toc-label': data_toc_label
                    })

                if self.use_anchors:
                    self.add_anchor(el, el.attrib["id"])
                if self.use_permalinks not in [False, None]:
                    self.add_permalink(el, el.attrib["id"])

        toc_tokens = nest_toc_tokens(toc_tokens)
        div = self.build_toc_div(toc_tokens)
        if self.marker:
            self.replace_marker(doc, div)

        # serialize and attach to markdown instance.
        toc = self.md.serializer(div)
        for pp in self.md.postprocessors:
            toc = pp.run(toc)
        self.md.toc_tokens = toc_tokens
        self.md.toc = toc


class TocExtension(Extension):

    TreeProcessorClass = TocTreeprocessor

    def __init__(self, **kwargs):
        self.config = {
            'marker': [
                '[TOC]',
                'Text to find and replace with Table of Contents. Set to an empty string to disable. '
                'Default: `[TOC]`.'
            ],
            'title': [
                '', 'Title to insert into TOC `<div>`. Default: an empty string.'
            ],
            'title_class': [
                'toctitle', 'CSS class used for the title. Default: `toctitle`.'
            ],
            'toc_class': [
                'toc', 'CSS class(es) used for the link. Default: `toclink`.'
            ],
            'anchorlink': [
                False, 'True if header should be a self link. Default: `False`.'
            ],
            'anchorlink_class': [
                'toclink', 'CSS class(es) used for the link. Defaults: `toclink`.'
            ],
            'permalink': [
                0, 'True or link text if a Sphinx-style permalink should be added. Default: `False`.'
            ],
            'permalink_class': [
                'headerlink', 'CSS class(es) used for the link. Default: `headerlink`.'
            ],
            'permalink_title': [
                'Permanent link', 'Title attribute of the permalink. Default: `Permanent link`.'
            ],
            'permalink_leading': [
                False,
                'True if permalinks should be placed at start of the header, rather than end. Default: False.'
            ],
            'baselevel': ['1', 'Base level for headers. Default: `1`.'],
            'slugify': [
                slugify, 'Function to generate anchors based on header text. Default: `slugify`.'
            ],
            'separator': ['-', 'Word separator. Default: `-`.'],
            'toc_depth': [
                6,
                'Define the range of section levels to include in the Table of Contents. A single integer '
                '(b) defines the bottom section level (<h1>..<hb>) only. A string consisting of two digits '
                'separated by a hyphen in between (`2-5`) defines the top (t) and the bottom (b) (<ht>..<hb>). '
                'Default: `6` (bottom).'
            ],
        }
        """ Default configuration options. """

        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add TOC tree processor to Markdown. """
        md.registerExtension(self)
        self.md = md
        self.reset()
        tocext = self.TreeProcessorClass(md, self.getConfigs())
        md.treeprocessors.register(tocext, 'toc', 5)

    def reset(self) -> None:
        self.md.toc = ''
        self.md.toc_tokens = []


def makeExtension(**kwargs):  # pragma: no cover
    return TocExtension(**kwargs)
