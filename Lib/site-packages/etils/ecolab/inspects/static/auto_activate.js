/**
 * @fileoverview Auto-activate inspect on-demand.
 */


/**
 * Add the `button` to activate auto-inspect.
 * @param {string} id_ HTML id of the node
 */
function add_auto_activate(id_) {
  // TODO(epot): Make the current `output-body` invisible (if no outputs).
  // TODO(epot): In Jupyter, this is class='output_subarea'
  const output_classname = 'output-body';

  const root = document.getElementById(output_classname);
  root.classList.add('inspect_activable');

  // Fetch all the cell output
  const children = root.querySelectorAll(`.execute_result`);

  if (children.length != 1) {  // No outputs
    return;  // No output, shouldn't be added in the first place
  }

  const last_output = children[0];

  // Create partial
  function onclick_called() {
    activate_inspect(last_output, id_);
  }

  const button = document.createElement('button');
  // TODO(epot): Replace text by icon.
  button.innerHTML = 'Inspect';
  button.onclick = onclick_called;
  button.classList.add('colab', 'activate_inspect');
  last_output.appendChild(button);
}

/**
 * Activate inspect (after the click).
 * @param {!HTMLElement} elem Display element to replace by the inspect
 * @param {string} id_ HTML id of the node
 */
async function activate_inspect(elem, id_) {
  // Compute the HTML content in Python
  const html_content = await call_python('get_inspect_html', [id_]);

  // Replace the output by its equivalent
  elem.innerHTML = html_content;

  // Activate the element
  load_content(id_);
}
