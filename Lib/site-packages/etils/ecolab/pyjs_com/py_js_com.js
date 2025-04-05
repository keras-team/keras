/**
 * @fileoverview Bi-directional Python communications.
 */

/**
 * Call a Python function from Javascript.
 *
 * @param {string} fn_name Function name. Should be registered with
 * `register_js_fn`
 * @param {!Array=} args Args passed to `fn_name(*args)`
 * @param {!Object=} kwargs Kwargs passed to `fn_name(**kwargs)`
 * @returns {!Object} Output of the function `out = fn_name()` (json-like
 * structure)
 */
async function call_python(fn_name, args = [], kwargs = {}) {
  // TODO(epot): Also support VSCode notebooks
  // TODO(epot): Is there a cleaner way ? Currently this has to be defined
  // inside because `const` in JS do not work well with Colab reload.
  /**
   * Enum of notebook types
   */
  const NbType = {
    COLAB: Symbol('colab'),
    JUPYTER: Symbol('jupyter'),

    /**
     * Returns the current notebook type.
     *
     * @returns {!Symbol} Current notebook type
     */
    get_current() {
      try {
        /** @suppress {suspiciousCode} */
        google.colab.kernel;
      } catch (error) {
        return this.JUPYTER;
      }
      return this.COLAB;
    },
  };

  const data = await (async function() {
    switch (NbType.get_current()) {
      case NbType.COLAB:
        return await _call_python_colab(fn_name, args, kwargs);
      case NbType.JUPYTER:
        return await _call_python_jupyter(fn_name, args, kwargs);
      default:
        throw new Error('Unknown notebook type');
    }
  })();
  if ('__etils_pyjs__' in data) {
    return data['__etils_pyjs__'];  // Unwrap inner value
  } else {
    return data;
  }
}

/**
 * Call a Python function from Javascript.
 *
 * @param {string} fn_name Function name. Should be registered with
 * `register_js_fn`
 * @param {!Array=} args Args passed to `fn_name(*args)`
 * @param {!Object=} kwargs Kwargs passed to `fn_name(**kwargs)`
 * @returns {!Object} Output of the function `out = fn_name()` (json-like
 * structure)
 */
async function _call_python_colab(fn_name, args = [], kwargs = {}) {
  const out = await google.colab.kernel.invokeFunction(fn_name, args, kwargs);
  return out.data['application/json'];
}

/**
 * Call a Python function from Javascript.
 *
 * @param {string} fn_name Function name. Should be registered with
 * `register_js_fn`
 * @param {!Array=} args Args passed to `fn_name(*args)`
 * @param {!Object=} kwargs Kwargs passed to `fn_name(**kwargs)`
 * @returns {!Object} Output of the function `out = fn_name()` (json-like
 * structure)
 */
async function _call_python_jupyter(fn_name, args = [], kwargs = {}) {
  // TODO(epot): Is there a better way to not re-create a new com at each
  // call ?

  const comm = Jupyter.notebook.kernel.comm_manager.new_comm(fn_name, {});
  // Send data to the function
  comm.send({args: args, kwargs: kwargs});

  // Wait for the answer
  const callOutputPromise = new Promise((resolve, reject) => {
    // Register a handler and capture the output of the
    comm.on_msg(function(msg) {
      resolve(msg.content.data);
    });
  });
  return await callOutputPromise;
}
