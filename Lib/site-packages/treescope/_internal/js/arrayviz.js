/**
 * Copyright 2024 The Treescope Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview JavaScript implementation of arrayviz.
 * This file is inserted into the generated Treescope HTML when arrayviz is
 * used from Python.
 */

/** @type {*} */
const arrayviz = (() => {
  /**
   * Converts a base64 string to a Uint8 array.
   * @param {string} s
   * @return {!Uint8Array}
   */
  function deserializeUint8(s) {
    return Uint8Array.from(atob(s), (m) => m.codePointAt(0));
  }

  /**
   * Converts a base64 string to a Float32 array.
   * @param {string} s
   * @return {!Float32Array}
   */
  function deserializeFloat32(s) {
    const bytes = Uint8Array.from(atob(s), (m) => m.codePointAt(0));
    const floats = new Float32Array(bytes.buffer);
    return floats;
  }

  /**
   * Converts a base64 string to a Int32 array.
   * @param {string} s
   * @return {!Int32Array}
   */
  function deserializeInt32(s) {
    const bytes = Uint8Array.from(atob(s), (m) => m.codePointAt(0));
    const floats = new Int32Array(bytes.buffer);
    return floats;
  }


  /**
   * Data defining a colormap as a list of RGB color stops. Each inner array
   * should be length 3, representing [r, g, b] with each channel between 0 and
   * 255.
   * @typedef {!Array<!Array<number>>}
   */
  let ColormapRGBData;

  /**
   * Computes a color for a value.
   * @param {number} value The value between 0 and 1 to compute a color for.
   * @param {!ColormapRGBData} colormapData Color stops for the colormap.
   * @return {!Array<number>} A color string.
   */
  function getContinuousColor(value, colormapData) {
    let base_index, offset;
    if (value <= 0) {
      base_index = 0;
      offset = 0;
    } else if (value >= 1) {
      base_index = colormapData.length - 2;
      offset = 1.0;
    } else {
      const cts_index = value * (colormapData.length - 1);
      base_index = Math.floor(cts_index);
      offset = cts_index % 1;
    }
    const r = colormapData[base_index][0] * (1 - offset) +
        colormapData[base_index + 1][0] * offset;
    const g = colormapData[base_index][1] * (1 - offset) +
        colormapData[base_index + 1][1] * offset;
    const b = colormapData[base_index][2] * (1 - offset) +
        colormapData[base_index + 1][2] * offset;
    return [r, g, b];
  }

  /**
   * Constructs a symmetric log transformation (via arcsinh) that is dynamically
   * adjusted to make keypointValue visible.
   * Assumes that an input of zero should map to 0.
   * @param {number} keypointValue A keypoint value which should have high
   *   contrast, e.g. the value of the cell the user is hovering over.
   * @param {number} clipMax The value that would map to 1 without dynamic
   *   adjustment, e.g. the max after removing outliers.
   * @param {number} rawMin The smallest absolute value in the array.
   * @param {number} rawMax The largest absolute value in the array, including
   *   outliers.
   * @param {number} dynamicGap Target magnitude gap for the keypoint value.
   *   The remapping will try to ensure that the keypoint is at least this far
   *   from zero and from one, to dynamically improve visual contrast.
   * @return {function(number):number} A callable that remaps input values to
   *   new values between -1 and 1.
   */
  function dynamicArcsinhRemap(
      keypointValue, clipMax, rawMin, rawMax, dynamicGap) {
    function stableScaledArcsinh(x, tau) {
      if (!isFinite(tau) || Math.abs(x) / tau < 1e-4) {
        return x;
      } else {
        return Math.sign(x) * tau *
            Math.log(Math.abs(x) / tau + Math.sqrt(1 + Math.pow(x / tau, 2)));
      }
    }
    // When hovering over an exactly-zero value, raise contrast of all nonzero.
    if (keypointValue == 0) {
      if (rawMin == 0) {
        keypointValue = clipMax;
      } else {
        keypointValue = rawMin;
      }
    }
    keypointValue = Math.abs(keypointValue);
    rawMax = Math.max(rawMax, clipMax);
    if (keypointValue < clipMax) {
      // Use arcsinh remapping to ensure keypointValue maps to at least
      // dynamicGap, while ensuring clipMax maps to 1.
      // If keypointValue is close to clipMax already, approach a linear
      // scale; if it is close to zero, approach a (sym)log scale.
      //
      // Goal: arcsinh(keypoint/tau) = (1 - dynamicGap) * arcsinh(clipMax/tau)
      // If keypoint is much smaller than clipMax, arcsinh will behave like
      // log, so we can instead aim for
      //    log(keypoint/tau) = (1 - dynamicGap) * log(clipMax/tau)
      //    => log(k/t) = (1 - d) * log(c/t)
      //    => log(k) - log(t) = (1 - d) * (log(c) - log(t))
      //    => d log(t) = log(k) - d log(c)
      // which means
      //    tau = exp((log(keypoint) - dynamicGap * log(clipMax)))
      // On the other hand, if keypoint is similar to clipMax, we want tau to
      // to go infinity so that arcsinh behaves like a linear scale, so we
      // additionally divide by (1 - keypointValue/clipMax).
      const taubase =
          Math.exp(Math.log(keypointValue) - dynamicGap * Math.log(clipMax));
      const tau = taubase / (1 - keypointValue / clipMax);
      let maxval = stableScaledArcsinh(clipMax, tau);
      if (maxval == 0) {
        maxval = 1e-10;
      }
      const remapper = (x) => stableScaledArcsinh(x, tau) / maxval;
      return remapper;
    } else {
      // Use arcsinh remapping to extend the colormap upward past clipMax so
      // that keypointValue maps to at most 1 - dynamicGap. One exception: if
      // rawMax maps to 1, it's OK to let keypointValue be larger than
      // 1 - dynamicGap. Again, if keypointValue is close to clipMax, approach
      // a linear scale; as it grows much larger, approach a (sym)log scale.
      //
      // In this case, we set tau to match clipMax when keypointValue is large,
      // and to go to infinity if keypointValue is close to clipMax.
      const tau = clipMax / (1 - clipMax / keypointValue);
      // We then choose the upper end of the colormap to ensure the value is
      // chosen appropriately.
      let denominator = Math.min(
          stableScaledArcsinh(keypointValue, tau) / (1 - dynamicGap),
          stableScaledArcsinh(rawMax, tau));
      if (denominator == 0) {
        denominator = 1e-10;
      }
      return (x) => stableScaledArcsinh(x, tau) / denominator;
    }
  }

  /**
   * Converts a color tuple to a CSS string.
   * @param {!Array<number>} color An array [r, g, b] of color channels.
   * @return {string} A CSS color string for this color.
   */
  function toStyleString(color) {
    const [r, g, b] = color;
    return `rgb(${r} ${g} ${b})`;
  }

  /**
   * Constructs a CSS string for a color that contrasts with the given color.
   * @param {!Array<number>} color An array [r, g, b] of color channels.
   * @return {string} A CSS color string that contrasts with this color. This
   * will be white if the color is dark, or black if the color is light.
   */
  function contrastingStyleString(color) {
    const [r, g, b] = color;
    // https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
    const intensity = .2126 * r + .7152 * g + .0722 * b;
    if (intensity > 128) {
      return 'black';
    } else {
      return 'white';
    }
  }


  /**
   * Utility to fill in pixels of a context with a particular pattern.
   * @param {!CanvasRenderingContext2D} ctx Context to render to.
   * @param {number} x X coordinate of top-left pixel.
   * @param {number} y Y coordinate of top-left pixel.
   * @param {!Array<!Array<number>>} pattern Array that is nonzero at locations
   * we should fill in.
   */
  function fillPattern(ctx, x, y, pattern) {
    for (let i = 0; i < pattern.length; i++) {
      const row = pattern[i];
      for (let j = 0; j < row.length; j++) {
        if (row[j]) {
          ctx.fillRect(x + j, y + i, 1, 1);
        }
      }
    }
  }

  // Modified version of D3's category10 / category20, with white as 0
  const digitColorPrimary = [
    '#ffffff', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#bcbd22', '#17becf'
  ];
  const digitColorSecondary = [
    '#7f7f7f', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#dbdb8d', '#9edae5'
  ];
  // unused color from category10: c7c7c7

  const cellSize = 7;  // in pixels

  /**
   * Configuration for the rendering of individual cells.
   * - type should be either "continuous", "palette_index", or "digitbox".
   * - min, max are only used for "continuous" mode, and give bounds for the
   *   colormap.
   * - dynamic is only used for "continuous" mode, and indicates whether the
   *   colormap should be adjusted dynamically to make the value under the
   *   cursor more visible.
   * - rawMinAbs, rawMaxAbs are only used for "continuous" mode when dynamic is
   *   true, and give the minimum and maximum absolute values in the array.
   * - cmapData is used for either "continuous" or "palette_index" modes.
   * @typedef {{
   *    type: string,
   *    min: (number | undefined),
   *    max: (number | undefined),
   *    dynamic: (boolean | undefined),
   *    rawMinAbs: (number | undefined),
   *    rawMaxAbs: (number | undefined),
   *     cmapData: (!ColormapRGBData | undefined),
   * }}
   */
  let ColormapConfig;

  /**
   * Fills in a single cell of an arrayviz rendering.
   * @param {!CanvasRenderingContext2D} ctx Context to render to.
   * @param {number} row Row to render.
   * @param {number} col Column to render.
   * @param {number} value Value for the array at this location.
   * @param {boolean} isValid Whether this value is "valid" (e.g. was not masked
   * out)
   * @param {!ColormapConfig} colormapConfig Configuration for the colormap.
   * @param {?function(number):number} remapper Function mapping input values to
   * colormap positions between 0 and 1. Only required for continuous colormaps.
   */
  function drawOneCell(
      ctx, row, col, value, isValid, colormapConfig, remapper) {
    if (!isValid) {
      ctx.fillStyle = 'oklch(55% 0 0)';
      ctx.fillRect(col * 7, row * 7, 7, 7);
      ctx.fillStyle = 'oklch(70% 0 0)';
      fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
      ]);
    } else if (colormapConfig.type === 'continuous') {
      const cmapData =
          /** @type {!ColormapRGBData} */ (colormapConfig.cmapData);
      if (isNaN(value)) {
        ctx.fillStyle = 'black';
        ctx.fillRect(col * 7, row * 7, 7, 7);
        ctx.fillStyle = 'rgb(255, 0, 255)';
        fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
          [1, 0, 0, 0, 1],
          [0, 1, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [1, 0, 0, 0, 1],
        ]);
      } else if (value == Infinity) {
        const color = getContinuousColor(1.0, cmapData);
        ctx.fillStyle = toStyleString(color);
        ctx.fillRect(col * 7, row * 7, 7, 7);
        ctx.fillStyle = contrastingStyleString(color);
        fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
        ]);
      } else if (value == -Infinity) {
        const color = getContinuousColor(0.0, cmapData);
        ctx.fillStyle = toStyleString(color);
        ctx.fillRect(col * 7, row * 7, 7, 7);
        ctx.fillStyle = contrastingStyleString(color);
        fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
          [0, 0, 1, 1, 1],
          [0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 1, 1],
        ]);
      } else {
        const remapped = remapper(value);
        if (remapped > 1.0) {
          const color = getContinuousColor(1.0, cmapData);
          ctx.fillStyle = toStyleString(color);
          ctx.fillRect(col * 7, row * 7, 7, 7);
          ctx.fillStyle = contrastingStyleString(color);
          fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
          ]);
        } else if (remapped < 0.0) {
          const color = getContinuousColor(0.0, cmapData);
          ctx.fillStyle = toStyleString(color);
          ctx.fillRect(col * 7, row * 7, 7, 7);
          ctx.fillStyle = contrastingStyleString(color);
          fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
          ]);
        } else {
          if (isNaN(remapped)) {
            // Shouldn't happen, but do something noticeable if it does.
            ctx.fillStyle = 'rgb(255, 0, 255)';
          } else {
            ctx.fillStyle =
                toStyleString(getContinuousColor(remapped, cmapData));
          }
          ctx.fillRect(col * 7, row * 7, 7, 7);
        }
      }
    } else if (colormapConfig.type == 'palette_index') {
      // Discrete palette colormap.
      const cmapData =
          /** @type {!ColormapRGBData} */ (colormapConfig.cmapData);
      const color = cmapData[value];
      if (color === undefined) {
        ctx.fillStyle = 'black';
        ctx.fillRect(col * 7, row * 7, 7, 7);
        ctx.fillStyle = 'rgb(255, 0, 255)';
        fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
          [1, 0, 0, 0, 1],
          [0, 1, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [1, 0, 0, 0, 1],
        ]);
      } else {
        const [r, g, b] = color;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(col * 7, row * 7, 7, 7);
      }
    } else if (colormapConfig.type == 'digitbox') {
      // Rendering arbitrary integers.
      const drawPartsByDigit = [
        [],
        [[0, 7]],
        [[0, 7], [2, 3]],
        [[0, 7], [1, 5], [2, 3]],
        [[0, 7], [1, 6], [2, 4], [3, 2]],
        [[0, 7], [1, 6], [2, 5], [3, 4], [4, 2]],
        [[0, 7], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2]],
        [[0, 7], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]],
      ];
      let invalid = false;
      if (value % 1 != 0) {
        // Not an integer
        invalid = true;
      } else {
        const negative = value < 0;
        const magnitude = Math.abs(value);
        const digits = magnitude.toString().split('');
        if (digits.length < drawPartsByDigit.length) {
          const draw_parts = drawPartsByDigit[digits.length];
          let prevent_repeats_of = null;
          for (let i = 0; i < digits.length; ++i) {
            const [start, sidelength] = draw_parts[i];
            const digit = Number(digits[i]);
            if (digit === prevent_repeats_of) {
              ctx.fillStyle = digitColorSecondary[digit];
              prevent_repeats_of = null;
            } else {
              ctx.fillStyle = digitColorPrimary[digit];
              prevent_repeats_of = digit;
            }
            ctx.fillRect(
                col * 7 + start, row * 7 + start, sidelength, sidelength);
          }
          // Render sign as a black rect.
          if (negative) {
            ctx.fillStyle = 'black';
            // ctx.fillRect(col * 7, row * 7, 2, 2);
            fillPattern(ctx, col * 7, row * 7, [
              [1, 1, 1],
              [1, 1, 0],
              [1, 0, 0],
            ]);
          }
        } else {
          // Too long.
          invalid = true;
        }
      }

      if (invalid) {
        ctx.fillStyle = 'black';
        ctx.fillRect(col * 7, row * 7, 7, 7);
        ctx.fillStyle = 'rgb(255, 0, 255)';
        fillPattern(ctx, col * 7 + 1, row * 7 + 1, [
          [1, 0, 0, 0, 1],
          [0, 1, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [1, 0, 0, 0, 1],
        ]);
      }
    } else {
      ctx.fillStyle = 'rgb(255, 0, 255)';
      ctx.fillRect(col * 7, row * 7, 7, 7);
    }
  }


  /**
   * Formats hover tooltips for a cell value based on instructions.
   * @param {number} value Value at this cell.
   * @param {boolean} isValid Whether this value is valid.
   * @param {!Object<string, number>} indices Map from axis names to their
   *    indices.
   * @param {!Array<?>} instructions Sequence of rendering instructions
   *    (see implementation).
   * @return {string} A rendering of the value and its indices.
   */
  function formatValueAndIndices(value, isValid, indices, instructions) {
    const parts = [];
    for (const instr of instructions) {
      if (instr['type'] == 'literal') {
        parts.push(instr['value']);
      } else if (instr['type'] == 'index') {
        const offset = indices[instr['axis']];
        if (offset == instr['skip_start']) {
          parts.push(`${instr['skip_start']}:${instr['skip_end']}`);
        } else if (offset < instr['skip_start']) {
          parts.push(`${offset}`);
        } else {
          parts.push(`${offset - instr['skip_start'] - 1 + instr['skip_end']}`);
        }
      } else if (instr['type'] == 'value') {
        if (isValid) {
          parts.push(`${value}`);
        } else {
          parts.push('(masked out)');
        }
      } else if (instr['type'] == 'value_lookup') {
        if (!isValid && !instr['ignore_invalid']) {
          parts.push('(masked out)');
        } else {
          const substitution = instr['lookup_table'][value];
          if (substitution === undefined) {
            parts.push('(unknown)');
          } else {
            parts.push(substitution);
          }
        }
      } else if (instr['type'] == 'axis_lookup') {
        let offset = indices[instr.axis];
        if (offset > instr['skip_start']) {
          offset = offset - instr['skip_start'] - 1 + instr['skip_end'];
        }
        const substitution = instr['lookup_table'][offset];
        if (substitution === undefined) {
          parts.push('(unknown)');
        } else {
          parts.push(substitution);
        }
      } else {
        parts.push('<?>');
        console.error('Unknown instruction', instr);
      }
    }
    return parts.join('');
  }


  /**
   * Configuration for a named axis. Start is inclusive, end exclusive.
   * Name is the label for each axis internally; label is the label we should
   * use as the annotation of the axis in the figure.
   * @typedef {{name: string, label: string, start: number, end: number}}
   */
  let AxisSpec;

  /* Delays an action until a destination element becomes visible. */
  function _delayUntilVisible(destination, action) {
    // Trigger rendering as soon as the destination becomes visible.
    const visibilityObserver = new IntersectionObserver((entries) => {
      if (entries[0].intersectionRatio > 0) {
        const loadingMarkers = destination.querySelectorAll('.loading_message');
        action();
        for (let elt of loadingMarkers) {
          elt.remove();
        }
        visibilityObserver.disconnect();
      }
    }, {});
    requestAnimationFrame(() => {
      visibilityObserver.observe(destination);
    });
  }

  /**
   * Renders an array to a destination object.
   *  config.info: Info for the figure; drawn on the bottom.
   *  config.arrayBase64: Base64-encoded array of either float32 or int32 data.
   *  config.arrayDtype: Either "float32" or "int32".
   *  config.validMaskBase64: Base64-encoded array of uint8-encoded boolean
   *    data.
   *  config.dataStrides: Strides for the data in `arrayBase64` and
   *    `validMaskBase64`. Expressed as a map from axis ID to the stride (i.e.
   *    the number of elements to skip when iterating over that axis).
   *  config.xAxisSpecs: Specs for each named axis to arrange as a group on the
   *    figure's X axis, from innermost to outermost.
   *  config.yAxisSpecs: Specs for each named axis to arrange as a group on the
   *    figure's Y axis, from innermost to outermost.
   *  config.slicedAxisSpecs: Specs for each named axis to show a slice for,
   *    and allow manipulation of the slice with a slider. Ordered from top to
   *    bottom.
   *  config.colormapConfig: Configuration for the rendering of individual
   *    array elements. Determines which colormap to use, and whether to render
   *    as continuous or discrete.
   *  config.pixelsPerCell: The initial number of pixels per cell to render (and
   *    therefore the initial value for the zoom slider).
   *  config.valueFormattingInstructions: Instructions for rendering array
   *    indices and values on mouse hover/click.
   * @param {!HTMLElement} destination The element to render into.
   * @param {{
   *    info: string,
   *    arrayBase64: string,
   *    arrayDtype: string,
   *    validMaskBase64: string,
   *    dataStrides: !Object<string, number>,
   *    xAxisSpecs: !Array<!AxisSpec>,
   *    yAxisSpecs: !Array<!AxisSpec>,
   *    slicedAxisSpecs: !Array<!AxisSpec>,
   *    colormapConfig: !ColormapConfig,
   *    pixelsPerCell: number,
   *    valueFormattingInstructions: !Array<?>,
   * }} config Configuration for the setup.
   */
  function buildArrayvizFigure(destination, config) {
    _delayUntilVisible(destination, () => {
      _buildArrayvizFigure(destination, config);
    });
  }
  function _buildArrayvizFigure(destination, config) {
    const info = config.info;
    const arrayBase64 = config.arrayBase64;
    const arrayDtype = config.arrayDtype;
    const validMaskBase64 = config.validMaskBase64;
    const dataStrides = config.dataStrides;
    const xAxisSpecs = config.xAxisSpecs;
    const yAxisSpecs = config.yAxisSpecs;
    const slicedAxisSpecs = config.slicedAxisSpecs;
    const colormapConfig = config.colormapConfig;
    const pixelsPerCell = config.pixelsPerCell;
    const valueFormattingInstructions = config.valueFormattingInstructions;

    // Decode the array.
    let array;
    if (arrayDtype == 'float32') {
      array = deserializeFloat32(arrayBase64);
    } else if (arrayDtype == 'int32') {
      array = deserializeInt32(arrayBase64);
    } else {
      throw new Error('unsupported dtype');
    }
    const valid_mask = deserializeUint8(validMaskBase64);

    // Compute separations between axis names that are mapped to the same
    // physical axis (X or Y). Spaces follow the Fibonacci sequence.
    let old = 1;
    let current = 1;
    let seps = [0];
    for (let i = 0; i < Math.max(xAxisSpecs.length, yAxisSpecs.length); i++) {
      seps.push(current);
      current = old + current;
      old = current;
    }

    // Compute display strides and widths/heights.
    // Axis widths and heights measure how many cells are contained inside each
    // block for slices along that axis, including any spaces inserted between
    // inner blocks.
    let totalCols = 1;
    let colWidths = [1];
    let xAxisSizes = [];
    for (let i = 0; i < xAxisSpecs.length; i++) {
      const spec = xAxisSpecs[i];
      const n = spec.end - spec.start;
      xAxisSizes.push(n);
      totalCols = totalCols * n + seps[i] * (n - 1);
      colWidths.push(totalCols);
    }
    let totalRows = 1;
    let rowHeights = [1];
    let yAxisSizes = [];
    for (let i = 0; i < yAxisSpecs.length; i++) {
      const spec = yAxisSpecs[i];
      const n = spec.end - spec.start;
      yAxisSizes.push(n);
      totalRows = totalRows * n + seps[i] * (n - 1);
      rowHeights.push(totalRows);
    }

    // Variable tracking the current slices. This can be mutated to redraw
    // different slices.
    const activeSliceOffsets = {};
    for (let i = 0; i < slicedAxisSpecs.length; i++) {
      const spec = slicedAxisSpecs[i];
      activeSliceOffsets[spec.name] = 0;
    }

    // Canvas element that we will draw into.
    const canvas =
        /** @type {!HTMLCanvasElement} */ (document.createElement('canvas'));
    canvas.width = (totalCols + 2) * cellSize;
    canvas.height = (totalRows + 2) * cellSize;
    const activeCanvasContext =
        /** @type {!CanvasRenderingContext2D} */ (canvas.getContext('2d'));

    // Rendering function, which will render the array into a particular
    // context.
    let dynamicRemapKeypoint = null;
    const drawAllCells = () => {
      // Set up the colormap.
      let colormapRemapper = null;
      if (colormapConfig.type == 'continuous') {
        if (colormapConfig.dynamic) {
          let max = /** @type {number} */ (colormapConfig.max);
          if (max == 0) {
            max = 1e-10;
          }
          const rawMinAbs = /** @type {number} */ (colormapConfig.rawMinAbs);
          const rawMaxAbs = /** @type {number} */ (colormapConfig.rawMaxAbs);
          if (dynamicRemapKeypoint === null) {
            colormapRemapper = (x) => (x + max) / (2 * max);
          } else {
            const arcsinhMapper = dynamicArcsinhRemap(
                dynamicRemapKeypoint, max, rawMinAbs, rawMaxAbs, 0.15);
            colormapRemapper = (x) => (arcsinhMapper(x) + 1) / 2;
          }
        } else {
          const min = /** @type {number} */ (colormapConfig.min);
          const max = /** @type {number} */ (colormapConfig.max);
          if (min == max) {
            colormapRemapper = (x) => 0.5;
          } else {
            colormapRemapper = (x) => (x - min) / (max - min);
          }
        }
      }

      // Precompute strides for the slices, which don't change during rendering.
      let dataIndex = 0;
      for (const spec of slicedAxisSpecs) {
        dataIndex += dataStrides[spec.name] * activeSliceOffsets[spec.name];
      }

      const xOffsets = xAxisSpecs.map(() => 0);
      const yOffsets = yAxisSpecs.map(() => 0);
      let col = 1;
      while (true) {  // Loop over X
        let row = 1;

        while (true) {  // Loop over Y
          // Draw the current cell.
          const value = array[dataIndex];
          const isValid = valid_mask === null ? true : !!valid_mask[dataIndex];
          drawOneCell(
              activeCanvasContext, row, col, value, isValid, colormapConfig,
              colormapRemapper);
          // Increment displayed Y cell, skipping over spaces whenever we hit
          // the end of an axis group.
          let yAdvanceCount = 0;
          for (let i = 0; i < yAxisSpecs.length; i++) {
            yOffsets[i] += 1;
            row += rowHeights[i] + seps[i];
            dataIndex += dataStrides[yAxisSpecs[i].name];
            if (yOffsets[i] >= yAxisSizes[i]) {
              // Reached the end of this named axis group, advance past it.
              yAdvanceCount += 1;
              yOffsets[i] = 0;
              dataIndex -= dataStrides[yAxisSpecs[i].name] * yAxisSizes[i];
              row -= (rowHeights[i] + seps[i]) * yAxisSizes[i];
            } else {
              // In the middle of this named axis group. Continue rendering.
              break;
            }
          }
          if (yAdvanceCount == yOffsets.length) {
            // We advanced past every axis! Done with this column.
            break;
          }
        }  // End loop over Y

        // Increment displayed X cell, skipping over spaces whenever we hit
        // the end of an axis group.
        let xAdvanceCount = 0;
        for (let i = 0; i < xAxisSpecs.length; i++) {
          xOffsets[i] += 1;
          col += colWidths[i] + seps[i];
          dataIndex += dataStrides[xAxisSpecs[i].name];
          if (xOffsets[i] >= xAxisSizes[i]) {
            // Reached the end of this named axis group, advance past it.
            xAdvanceCount += 1;
            xOffsets[i] = 0;
            dataIndex -= dataStrides[xAxisSpecs[i].name] * xAxisSizes[i];
            col -= (colWidths[i] + seps[i]) * xAxisSizes[i];
          } else {
            // In the middle of this named axis group. Continue rendering.
            break;
          }
        }
        if (xAdvanceCount == xOffsets.length) {
          // We advanced past every axis! Done with this row.
          break;
        }
      }

      // Draw outlines. We use a similar loop as before, but we start with the
      // SECOND array axis, if it exists, because we don't want to draw an
      // outline for the innermost axis.
      activeCanvasContext.lineWidth = 1;
      activeCanvasContext.strokeStyle = 'black';
      activeCanvasContext.lineCap = 'square';
      activeCanvasContext.lineJoin = 'miter';
      while (true) {  // Loop over X
        let row = 1;

        while (true) {  // Loop over Y
          // Stroke the border of this group (the second-innermost, with a width
          // usually greater than one cell).
          activeCanvasContext.strokeRect(
              col * cellSize - 0.5,
              row * cellSize - 0.5,
              cellSize * (colWidths[1] ?? 1) + 1,
              cellSize * (rowHeights[1] ?? 1) + 1,
          );
          // Increment Y with overflow, but start at 1.
          let yAdvanceCount = 1;
          for (let i = 1; i < yAxisSpecs.length; i++) {
            yOffsets[i] += 1;
            row += rowHeights[i] + seps[i];
            dataIndex += dataStrides[yAxisSpecs[i].name];
            if (yOffsets[i] >= yAxisSizes[i]) {
              // Reached the end of this named axis group, advance past it.
              yAdvanceCount += 1;
              yOffsets[i] = 0;
              dataIndex -= dataStrides[yAxisSpecs[i].name] * yAxisSizes[i];
              row -= (rowHeights[i] + seps[i]) * yAxisSizes[i];
            } else {
              // In the middle of this named axis group. Continue rendering.
              break;
            }
          }
          if (yAdvanceCount >= yOffsets.length) {
            // We advanced past every axis! Done with this column.
            break;
          }
        }

        // Increment X with overflow, but start at 1
        let xAdvanceCount = 1;
        for (let i = 1; i < xAxisSpecs.length; i++) {
          xOffsets[i] += 1;
          col += colWidths[i] + seps[i];
          dataIndex += dataStrides[xAxisSpecs[i].name];
          if (xOffsets[i] >= xAxisSizes[i]) {
            // Reached the end of this named axis group, advance past it.
            xAdvanceCount += 1;
            xOffsets[i] = 0;
            dataIndex -= dataStrides[xAxisSpecs[i].name] * xAxisSizes[i];
            col -= (colWidths[i] + seps[i]) * xAxisSizes[i];
          } else {
            // In the middle of this named axis group. Continue rendering.
            break;
          }
        }
        if (xAdvanceCount >= xOffsets.length) {
          // We advanced past every axis! Done with this row.
          break;
        }
      }
    };

    // Dynamic remapping helper.
    function setRemapKeypoint(newKeypoint) {
      if (newKeypoint == dynamicRemapKeypoint) return;
      dynamicRemapKeypoint = newKeypoint;
      if (colormapConfig.type == 'continuous' && colormapConfig.dynamic) {
        drawAllCells();
      }
    }

    // Function to look up the cell index for a given pixel location, which is
    // used for hover tooltips and clicking.
    function lookupPixel(x, y) {
      // Look up pixel coords
      const col = Math.floor((x - 1) / cellSize) - 1;
      const row = Math.floor((y - 1) / cellSize) - 1;

      if (col < 0 || row < 0 || col >= colWidths[colWidths.length - 1] ||
          row >= rowHeights[rowHeights.length - 1]) {
        return null;
      }

      // Compute positions along each named axis and the overall offset into
      // the data array. We handle slider offsets first, then column and row
      // offsets starting from the outermost in the rendering.
      const result = {};
      let dataIndex = 0;
      for (const spec of slicedAxisSpecs) {
        const index = spec.start + activeSliceOffsets[spec.name];
        result[spec.name] = index;
        dataIndex += dataStrides[spec.name] * index;
      }
      let colRemainder = col;
      for (let i = xAxisSpecs.length - 1; i >= 0; i--) {
        const spec = xAxisSpecs[i];
        const stride = colWidths[i] + seps[i];
        const modulus = colWidths[i + 1] + seps[i + 1];
        colRemainder = colRemainder % modulus;
        const coord = Math.floor(colRemainder / stride);
        const index = coord + spec.start;
        if (coord < 0 || index >= spec.end) {
          return null;
        }
        result[spec.name] = index;
        dataIndex += dataStrides[spec.name] * index;
      }
      let rowRemainder = row;
      for (let i = yAxisSpecs.length - 1; i >= 0; i--) {
        const spec = yAxisSpecs[i];
        const stride = rowHeights[i] + seps[i];
        const modulus = rowHeights[i + 1] + seps[i + 1];
        rowRemainder = rowRemainder % modulus;
        const coord = Math.floor(rowRemainder / stride);
        const index = coord + spec.start;
        if (coord < 0 || index >= spec.end) {
          return null;
        }
        result[spec.name] = index;
        dataIndex += dataStrides[spec.name] * index;
      }
      return [array[dataIndex], valid_mask[dataIndex], result, row, col];
    }

    // HTML structure: The main figure, wrapped in a series of containers to
    // help with axis label layout.
    const container = /** @type {!HTMLDivElement} */ (
        destination.appendChild(document.createElement('div')));
    const inner =
        /** @type {!HTMLDivElement} */ (
            container.appendChild(document.createElement('div')));
    inner.appendChild(canvas);

    container.style.width = 'fit-content';
    container.style.height = 'fit-content';
    container.style.fontFamily = 'monospace';
    container.style.transformOrigin = 'top left';
    container.style.setProperty('image-rendering', 'pixelated');
    container.style.setProperty('--arrayviz-zoom', '1');
    container.style.setProperty('--base-font-size', '14px');

    inner.style.width = `calc(var(--arrayviz-zoom) * ${canvas.width}px)`;
    inner.style.height = `calc(var(--arrayviz-zoom) * ${canvas.height}px)`;
    inner.style.position = 'relative';

    canvas.style.transformOrigin = 'top left';
    canvas.style.setProperty('scale', 'var(--arrayviz-zoom)');

    // Temporary padding; will be overridden once we can inspect the actual
    // layout.
    container.style.paddingLeft = `calc(${yAxisSpecs.length} * (1.5em + 1px))`;
    container.style.paddingTop = `calc(${xAxisSpecs.length} * (1.5em + 1px))`;

    // HTML structure: Axis labels.
    const allLabels = [];
    const yLabelParts = [];
    const setCommonLabelStyles = (label) => {
      label.style.whiteSpace = 'pre';
      label.style.borderBottom = '0.1em solid black';
      label.style.fontSize =
          'calc(min(1, max(0.6, var(--arrayviz-zoom))) * var(--base-font-size))';
    };
    for (let i = 0; i < xAxisSpecs.length; i++) {
      const spec = xAxisSpecs[i];
      const label =
          /** @type {!HTMLDivElement} */ (document.createElement('div'));
      label.style.position = 'absolute';
      label.style.left = `calc(var(--arrayviz-zoom) * ${cellSize}px)`;
      label.style.bottom = `calc(100% + ${i} * 1.5em)`;
      label.style.width =
          `calc(var(--arrayviz-zoom) * ${colWidths[i + 1] * cellSize}px)`;
      label.textContent = spec.label;
      label.style.textAlign = 'left';
      setCommonLabelStyles(label);
      inner.appendChild(label);
      allLabels.push(label);
    }
    for (let i = 0; i < yAxisSpecs.length; i++) {
      const spec = yAxisSpecs[i];
      const label =
          /** @type {!HTMLDivElement} */ (document.createElement('div'));
      label.style.position = 'absolute';
      label.style.setProperty('rotate', '-90deg');
      label.style.transformOrigin = 'bottom right';
      label.style.right = `calc(100% + ${i} * 1.5em)`;
      label.style.bottom = `calc(100% - var(--arrayviz-zoom) * ${cellSize}px)`;
      label.style.width =
          `calc(var(--arrayviz-zoom) * ${rowHeights[i + 1] * cellSize}px)`;
      label.style.textAlign = 'right';
      setCommonLabelStyles(label);
      // Fix things up so that content overflows to the left if necessary
      const innerLabel =
          /** @type {!HTMLDivElement} */ (document.createElement('div'));
      innerLabel.textContent = spec.label;
      innerLabel.style.position = 'absolute';
      innerLabel.style.bottom = '0';
      innerLabel.style.right = '0';
      innerLabel.style.minWidth = '100%';
      innerLabel.style.setProperty('transform-origin', 'bottom right');
      label.appendChild(innerLabel);
      inner.appendChild(label);
      allLabels.push(innerLabel);
      yLabelParts.push({inner: innerLabel, outer: label});
    }

    // HTML structure: Sliders for sliced axes.
    const sliceBoxes =
        /** @type {!HTMLDivElement} */ (
            destination.appendChild(document.createElement('div')));
    sliceBoxes.classList.add('info', 'sliders');

    for (const spec of slicedAxisSpecs) {
      const size = spec.end - spec.start;
      const currentSliceBox =
          /** @type {!HTMLSpanElement} */ (
              sliceBoxes.appendChild(document.createElement('span')));
      currentSliceBox.append(`\n${spec.label}[`);
      const numbox =
          /** @type {!HTMLInputElement} */ (document.createElement('input'));
      numbox.type = 'number';
      numbox.value = spec.start;
      numbox.min = spec.start;
      numbox.max = spec.end - 1;
      numbox.step = 1;
      currentSliceBox.appendChild(numbox);
      currentSliceBox.append('] ');
      const slider =
          /** @type {!HTMLInputElement} */ (document.createElement('input'));
      slider.type = 'range';
      slider.value = spec.start;
      slider.min = spec.start;
      slider.max = spec.end - 1;
      slider.step = 1;
      slider.style.width = `calc(max(40ch, ${size}px))`;
      currentSliceBox.appendChild(slider);
      const sliceSliderCallback = ((evt) => {
        slider.value = evt.target.value;
        numbox.value = evt.target.value;
        // Update the active slice, then redraw.
        activeSliceOffsets[spec.name] = Number(evt.target.value - spec.start);
        drawAllCells();
      });
      slider.addEventListener('input', sliceSliderCallback);
      numbox.addEventListener('input', sliceSliderCallback);
    }

    // HTML structure: Build zoom slider and info annotations.
    const datalist =
        /** @type {!HTMLDataListElement} */ (
            destination.appendChild(document.createElement('datalist')));
    datalist.id = 'arrayviz-markers';
    for (const val of [1, 2, 3.5, 7, 14, 21]) {
      const datalistOption = /** @type {!HTMLOptionElement} */ (
          datalist.appendChild(document.createElement('option')));
      datalistOption.value = val;
    }
    const zoomslider =
        /** @type {!HTMLInputElement} */ (document.createElement('input'));
    zoomslider.type = 'range';
    zoomslider.min = '1';
    zoomslider.max = '21';
    zoomslider.step = 'any';
    zoomslider.value = pixelsPerCell;
    zoomslider.setAttribute('list', datalist.id);
    const infodiv = /** @type {!HTMLDivElement} */ (
        destination.appendChild(document.createElement('div')));
    infodiv.classList.add('info');
    infodiv.style.whiteSpace = 'pre';
    infodiv.append('Zoom: -');
    infodiv.append(zoomslider);
    infodiv.append('+    ' + info);

    // HTML structure: Hover tooltip and click info elements.
    const hovertip =
        /** @type {!HTMLDivElement} */ (
            inner.appendChild(document.createElement('div')));
    hovertip.className = 'hovertip';
    const hoverbox =
        /** @type {!HTMLDivElement} */ (
            inner.appendChild(document.createElement('div')));
    hoverbox.className = 'hoverbox';
    hoverbox.style.width = `calc(var(--arrayviz-zoom) * ${cellSize}px)`;
    hoverbox.style.height = `calc(var(--arrayviz-zoom) * ${cellSize}px)`;
    const clickdata = /** @type {!HTMLDivElement} */ (
        destination.appendChild(document.createElement('div')));
    clickdata.classList.add('info', 'clickdata');

    // Size adjustment to ensure labels do not overflow.
    function fixMargins() {
      const baseRect = inner.getBoundingClientRect();
      // Possibly rotate inner Y labels.
      let nextRight = `${baseRect.right - baseRect.left}px`;
      for (let i = 0; i < yLabelParts.length; i++) {
        const curParts = yLabelParts[i];
        curParts.outer.style.right = nextRight;
        curParts.inner.style.setProperty('transform', '');
        let labelRect = curParts.inner.getBoundingClientRect();
        if (labelRect.height > baseRect.height) {
          curParts.inner.style.setProperty(
              'transform', 'rotate(90deg) translate(-0.5ch, 1.2em)');
          labelRect = curParts.inner.getBoundingClientRect();
          nextRight = `calc(${baseRect.right - labelRect.left}px + 2ch)`;
        } else {
          nextRight = `calc(${baseRect.right - labelRect.left}px + 0.5em)`;
        }
      }
      // Fix outer margins.
      let fullRect = baseRect;
      for (let i = 0; i < allLabels.length; i++) {
        const curRect = allLabels[i].getBoundingClientRect();
        fullRect = {
          left: Math.min(fullRect.left, curRect.left),
          top: Math.min(fullRect.top, curRect.top),
          right: Math.max(fullRect.right, curRect.right),
          bottom: Math.max(fullRect.bottom, curRect.bottom),
        };
      }
      container.style.paddingLeft = `${baseRect.left - fullRect.left}px`;
      container.style.paddingTop =
          `calc(0.25em + ${baseRect.top - fullRect.top}px)`;
      container.style.paddingRight = `${fullRect.right - baseRect.right}px`;
      container.style.paddingBottom = `${fullRect.bottom - baseRect.bottom}px`;
    }

    // Event handlers.
    let pendingZoomPixels = 0;
    const updateZoomFromSlider = () => {
      // For accurate pixel alignment: Round device pixel ratio to an integer
      // so that zoom level 1 is an integer number of pixels.
      let roundedPxRatio = Math.round(window.devicePixelRatio);
      if (window.devicePixelRatio < 1) {
        roundedPxRatio = 1 / Math.round(1 / window.devicePixelRatio);
      }
      const pixelRatioAdjustment = roundedPxRatio / window.devicePixelRatio;
      // For accurate pixel alignment: Round to the closest number of physical
      // pixels, then map back to logical scale.
      let cssPxTarget = parseFloat(zoomslider.value);
      if (cssPxTarget < 1) {
        cssPxTarget = 1;
      }
      if ((cssPxTarget * roundedPxRatio) % cellSize == 0) {
        container.style.setProperty('image-rendering', 'pixelated');
      } else {
        container.style.setProperty('image-rendering', 'auto');
      }
      const scale = (cssPxTarget / cellSize) * pixelRatioAdjustment;
      container.style.setProperty('--arrayviz-zoom', `${scale}`);
      fixMargins();
    };
    const watchPixelRatio = () => {
      const media =
          window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      const listener = () => {
        media.removeEventListener('change', listener);
        updateZoomFromSlider();
        watchPixelRatio();
      };
      media.addEventListener('change', listener);
    };
    zoomslider.addEventListener('input', () => {
      pendingZoomPixels = 0;
      const oldTop = zoomslider.offsetTop;
      updateZoomFromSlider();
      // Try to keep it in a consistent place:
      window.scrollBy(
          {top: zoomslider.offsetTop - oldTop, behavior: 'instant'});
      // In case that didn't work, make sure it's at least visible:
      zoomslider.scrollIntoView({behavior: 'instant', block: 'nearest'});
    });
    canvas.addEventListener('mousemove', (evt) => {
      const lookup = lookupPixel(evt.offsetX, evt.offsetY);
      if (lookup === null) {
        hovertip.style.display = 'none';
        hoverbox.style.display = 'none';
      } else {
        const [value, is_valid, indices, row, col] = lookup;
        hoverbox.style.display = 'block';
        hoverbox.style.left =
            `calc(var(--arrayviz-zoom) * ${(col + 1) * cellSize}px)`;
        hoverbox.style.top =
            `calc(var(--arrayviz-zoom) * ${(row + 1) * cellSize}px)`;
        hovertip.textContent = formatValueAndIndices(
            value, is_valid, indices, valueFormattingInstructions);
        hovertip.style.display = 'block';
        const hoverpad = 6;
        const paddedWidth = hovertip.offsetWidth + hoverpad;
        const paddedHeight = hovertip.offsetHeight + hoverpad;
        if (evt.offsetX - paddedWidth > 0 &&
            evt.offsetX + paddedWidth > inner.offsetWidth) {
          hovertip.style.left = '';
          hovertip.style.right =
              `calc(100% - var(--arrayviz-zoom) * ${evt.offsetX - hoverpad}px)`;
        } else {
          hovertip.style.left =
              `calc(var(--arrayviz-zoom) * ${evt.offsetX + hoverpad}px)`;
          hovertip.style.right = '';
        }
        if (evt.offsetY - paddedHeight > 0 &&
            evt.offsetY + paddedHeight > inner.offsetHeight) {
          hovertip.style.top = '';
          hovertip.style.bottom =
              `calc(100% - var(--arrayviz-zoom) * ${evt.offsetY - hoverpad}px)`;
        } else {
          hovertip.style.top =
              `calc(var(--arrayviz-zoom) * ${evt.offsetY + hoverpad}px)`;
          hovertip.style.bottom = '';
        }
      }
    });
    canvas.addEventListener('mouseout', (evt) => {
      hovertip.style.display = 'none';
      hoverbox.style.display = 'none';
    });
    canvas.addEventListener('click', (evt) => {
      const lookupResult = lookupPixel(evt.offsetX, evt.offsetY);
      if (lookupResult == null) {
        setRemapKeypoint(null);
      } else {
        const [value, is_valid, indices, _] = lookupResult;
        if (!is_valid) {
          setRemapKeypoint(null);
        } else {
          clickdata.textContent = 'Clicked: ' +
              formatValueAndIndices(value, is_valid, indices,
                                    valueFormattingInstructions);
          if (value == dynamicRemapKeypoint) {
            setRemapKeypoint(null);
          } else {
            setRemapKeypoint(value);
          }
        }
      }
    });
    canvas.addEventListener('wheel', (evt) => {
      if (evt.altKey) {
        const deltaY = (/** @type {!WheelEvent} */ (evt)).deltaY;
        evt.preventDefault();
        if (Math.sign(pendingZoomPixels) != Math.sign(deltaY)) {
          pendingZoomPixels = 0;
        }
        pendingZoomPixels += deltaY;
        const origZoomLevel = parseFloat(zoomslider.value);
        let curZoomLevel = origZoomLevel;
        while (pendingZoomPixels > 4) {
          pendingZoomPixels -= 4;
          if (curZoomLevel < parseFloat(zoomslider.max)) {
            curZoomLevel = curZoomLevel + 1;
          }
        }
        while (pendingZoomPixels < -4) {
          pendingZoomPixels += 4;
          if (curZoomLevel > parseFloat(zoomslider.min)) {
            curZoomLevel = curZoomLevel - 1;
          }
        }
        zoomslider.value = curZoomLevel;
        updateZoomFromSlider();
        // Try to keep it in a consistent place. offsetX/offsetY are relative to
        // the zoom level, so we have to scale them.
        const baseRect = inner.getBoundingClientRect();
        window.scrollBy({
          top: (evt.clientY - baseRect.top) / origZoomLevel *
              (curZoomLevel - origZoomLevel),
          left: (evt.clientX - baseRect.left) / origZoomLevel *
              (curZoomLevel - origZoomLevel),
          behavior: 'instant',
        });
      }
    }, {passive: false});

    // Draw the initial output.
    drawAllCells();
    updateZoomFromSlider();
    watchPixelRatio();
  }

  /**
   * Renders a single cell representing an integer digit.
   *  config.value: Integer value to render.
   *  config.labelTop: Label to draw above the box.
   *  config.labelBottom: Label to draw below the box.
   * @param {!HTMLElement} destination The element to render into.
   * @param {{
   *    value: number,
   *    labelTop: string,
   *    labelBottom: string,
   * }} config Configuration for the digitbox.
   */
  function renderOneDigitbox(destination, config) {
    _delayUntilVisible(destination, () => {
      const value = config.value;
      const labelTop = config.labelTop;
      const labelBottom = config.labelBottom;

      const canvas =
          /** @type {!HTMLCanvasElement} */ (document.createElement('canvas'));
      canvas.width = cellSize;
      canvas.height = cellSize;
      const ctx =
          /** @type {!CanvasRenderingContext2D} */ (canvas.getContext('2d'));
      drawOneCell(ctx, 0, 0, value, true, {type: 'digitbox'}, null);

      const container =
          /** @type {!HTMLDivElement} */ (document.createElement('div'));
      container.style.width = '1cap';
      container.style.height = '1cap';
      container.style.marginTop = '0.75em';
      container.style.display = 'inline-block';
      container.style.position = 'relative';
      container.style.overflow = 'visible';
      container.style.fontFamily = 'monospace';
      container.style.whiteSpace = 'pre';
      container.style.lineHeight = '2.5';
      container.style.setProperty('image-rendering', 'pixelated');

      container.style.outline = '1px solid black';

      container.appendChild(canvas);
      canvas.style.width = '1cap';
      canvas.style.height = '1cap';
      canvas.style.position = 'absolute';

      if (labelTop) {
        const labelElt =
            /** @type {!HTMLDivElement} */ (document.createElement('div'));
        container.appendChild(labelElt);
        labelElt.style.width = '200%';
        labelElt.style.fontSize =
            `${0.75 * Math.min(1, 3.5 / labelTop.length)}em`;
        labelElt.style.position = 'absolute';
        labelElt.style.left = '-50%';
        labelElt.style.bottom = '110%';
        labelElt.style.textAlign = 'center';
        labelElt.style.lineHeight = '1';
        labelElt.textContent = labelTop;
      }
      if (labelBottom) {
        const labelElt =
            /** @type {!HTMLDivElement} */ (document.createElement('div'));
        container.appendChild(labelElt);
        labelElt.style.width = '200%';
        labelElt.style.fontSize =
            `${0.75 * Math.min(1, 3.5 / labelBottom.length)}em`;
        labelElt.style.position = 'absolute';
        labelElt.style.left = '-50%';
        labelElt.style.top = '110%';
        labelElt.style.textAlign = 'center';
        labelElt.style.lineHeight = '1';
        labelElt.textContent = labelBottom;
      }
      destination.appendChild(container);
    });
  }

  return {
    renderOneDigitbox: renderOneDigitbox,
    buildArrayvizFigure: buildArrayvizFigure,
  };
})();
