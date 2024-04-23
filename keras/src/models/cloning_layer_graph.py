import keras
from keras.src.api_export import keras_export
from keras import tree
from keras.src.ops.symbolic_arguments import SymbolicArguments
from keras.src.models import Functional
from keras import Sequential
from keras.src.models.functional import is_input_keras_tensor


# Implementation note: Why not modify the existing functional.clone_graph_nodes ?
# Context: The purpose of the existing function is to create a new
#          functional model from any intermediate Keras tensors.
#
#       1) The existing function does not re-execute the operations (layers)
#          in the graph. This new function does, as it is necessary to run clone_fn.
#          => best to keep them distinct functions
#       2) The existing function copies input tensors to replace them
#          with proper InputLayers. This is essential when creating a subgraph
#          starting at intermediate tensors. This new function always assumes
#          all inputs are InputLayers.
#       3) The existing function does not have a clone_fn. Adding one would
#          complexify the code.

@keras_export("keras.models.clone_layer_graph")
def clone_layer_graph(input, output, clone_fn, enter_nested=True):
    """Clone the layer graph between input and output. Actual layers are NOT cloned,
    but shared. Only the graph of layers is re-created. The clone_fn function
    is called when cloning each node (i.e. layer invocation in the graph) which
    allows you to modify the topology of the graph.

    Recommended usage:
    ```python
    def clone_fn(layer, *args, **kwargs)
        # here, you can insert layers, wrap layers, extract activations, etc.
        return layer(*args, **kwargs) # default to identity
    model = ... # a keras model
    output = clone_layer_graph(model.input, model.output, clone_fn)
    new_model = keras.Model(model.input, output)
    ```

    - When cloning a layer graph, shared layers remain shared.
    - Since no actual cloning of layers occurs, layers do not need to have
      serialization implemented (i.e. implement `get_config()`).
    - Cloning a layer graph with nested subgraphs (i.e. layers that are themselves
      Functional or Sequential models) is possible. If a clone_fn is provided,
      the nested subgraph will be cloned as a new Functional (Note: this
      will change Sequential subgraphs to Functional)

    Args:
        input: Instance of `KerasTensor` or pytree of `KerasTensor`s.
               All inputs must be of type `keras.Input`. If you wish to
               clone a layer graph that starts with intermediate KerasTensors,
               you have to create a new Functional model first by calling
               `model = keras.Model(intermediate_tensors, output)` which will
               create proper `Input` tensors instead of the intermediate ones.
        output: Instance of `KerasTensor` or pytree of `KerasTensor`s.
        clone_fn: Callable that will be called when each layer in the layer graph is
               invoked. The expected signature is `clone_fn(layer, *args, **kwargs)`.
               To leave a layer unchanged, `return layer(*args, **kwargs)`.

    Examples:

    ```python
    # clone the layer graph identically (actual layers will be shared, not cloned)
    def clone_fn(layer, *args, **kwargs):
        output = layer(*args, **kwargs)  # identity call
        return output
    model = ... # a keras model
    output = clone_layer_graph(model.input, model.output, clone_fn)
    new_model = keras.Model(model.input, output)
    ```

    ```python
    # wrap every Dense layer in custom layer WrapDense
    def clone_fn(layer, *args, **kwargs):
        if isinstance(layer, layers.Dense):
            wrapper = WrapDense(layer)
            return wrapper(*args, **kwargs)
        else:
            return layer(*args, **kwargs)  # default to identity
    model = ... # a keras model
    output = clone_layer_graph(model.input, model.output, clone_fn)
    new_model = keras.Model(model.input, output)
    ```

    ```python
    # Insert an extra Dense(128) layer after every Dense layer
    def clone_fn(layer, *args, **kwargs):
        if isinstance(layer, layers.Dense):
            output = layer(*args, **kwargs)
            output = layers.Dense(128)(output)
            return output
        else:
            return layer(*args, **kwargs)  # default to identity
    model = ... # a keras model
    output = clone_layer_graph(model.input, model.output, clone_fn)
    new_model = keras.Model(model.input, output)
    ```

    ```python
    # Collect inner activations from the model and create a new model that returns them
    activations = []
    def clone_fn(layer, *args, **kwargs):
        output = layer(*args, **kwargs)  # identity call
        activations.append(output)
        return output
    model = ... # a keras model
    output = clone_layer_graph(model.input, model.output, clone_fn)
    new_output = [output] + activations
    new_model = keras.Model(model.input, new_output)
    ```
    """

    # input is only used for checking that all inputs are real keras.Input layers.
    for t in tree.flatten(input):
        if not (isinstance(t, keras.KerasTensor) and is_input_keras_tensor(t)):
            raise ValueError(
                f"All input values must be KerasTensors. If you want to call "
                f"clone_layer_graph with intermediate tensors as inputs, call "
                f"model = keras.Model(intermediate_tensors, output) first, "
                f"which will create input tensors for the subgraph, then call "
                f"clone_layer_graph(model.input, model.output)\n"
                f"Received: input={input} "
                f"including invalid value {t} of type {type(t)}."
            )

    # TODO: check that the graph from input to output is connected. This is
    #       not strictly necessary because 'input' is used for typechecking
    #       only (that all inputs are keras.Input). The function will work
    #       as long as any Inputs are reached by walking back from "output".
    #       However, it would be better not allow a mismatched input/output
    #       so that this pathological use case does not become part of the API.

    # The "visited" dictionary is used to store three types of visited objects.
    # The key used is id(object). The value is the object depends on the type:
    # keras.ops.node.Node: visited nodes are stored as {id(node):True} to make
    #                      sure they are visited once and once only during graph
    #                      traversal.
    # keras.KerasTensor:   computed output tensors are stored in "visited", keyed
    #                      by the id of the original symbolic tensor in the graph.
    #                      Gathering input tensors for a node operation is simply:
    # args, kwargs = SymbolicArguments(*node.arguments.args, **node.arguments.kwargs).fill_in(visited)
    # keras.Functional / keras.Sequential:
    #                      Nested subgraphs, once cloned, are also stored in "visited"
    #                      keyed by the id of the original nested subgraph. When visited
    #                      a second time, the new, cloned, subgraph is retrieved from
    #                      "visited" and used.
    visited = {}
    # This implements depth-first graph traversal with node execution
    # triggered as soon as all inputs to a node have been computed.
    # Node execution triggers the clone_fn.
    output = _walk_back_tensors(output, visited, clone_fn, enter_nested)
    return output


def _walkback_one_tensor(tensor):
    (operation,
     node_index,
     tensor_index
     ) = tensor._keras_history
    node = operation._inbound_nodes[node_index]
    return node


def _handle_nested_node(node, visited, clone_fn, enter_nested):
    nested = isinstance(node.operation, Functional) or isinstance(node.operation, Sequential)

    # if this nested model was already visited, return the new op that was created
    if nested and enter_nested and id(node.operation) in visited:
        return visited[id(node.operation)]

    # enter nested layers only once, after that, just run the layer (i.e. do nothing here)
    if nested and enter_nested and id(node.operation) not in visited:
        visited.update({id(node.operation): node.operation})  # enter nested layers only once

        # Note: the subgraph of a nested layer is unique, even if the nested layer
        # is used more than once in the model graph: it's the subgraph between
        # node.operation.input and node.operation.output.

        output = _walk_back_tensors(node.operation.output, visited, clone_fn,
                                    enter_nested)  # jump into the nested layer

        # This is a very approximate test to know if the composite layer
        # was changed by applying clone_fn. In the future, it might be
        # interesting to detect clone_fn changes anything to the subgraph
        # and not re-create it in that case.
        has_changed = clone_fn is not None

        # recreate cloned nested node by calling Model(input, output)
        if has_changed:
            input = node.operation.input
            assert all([is_input_keras_tensor(t) for t in tree.flatten(input)]), \
                "Cannot clone a Functional graph that does not start with Inputs"

            new_composite_layer = keras.Model(input, output, name=node.operation.name + "_clone")
            # redirect the old operation to this new layer
            visited.update({id(node.operation): new_composite_layer})
            return new_composite_layer
    return


def _handle_input_node(node, visited):
    if node.is_input:
        input_tensor = node.output_tensors[0]  # An input layer is always a single tensor
        visited.update({id(input_tensor): input_tensor})  # keep the same value for inputs
        return True


def _clone_node(node, operation, visited, clone_fn):
    # now that all inputs are computed, compute output value
    arguments = SymbolicArguments(*node.arguments.args, **node.arguments.kwargs)
    # if some values are missing, i.e. have not been computed yet,  this call will fail
    args, kwargs = arguments.fill_in(visited)

    # when there is no clone_fn, clone by running the original operation
    # otherwise, clone by running clone_fn
    if clone_fn:
        output = clone_fn(operation, *args, **kwargs)
    else:
        output = operation(*args, **kwargs)

    outputs = tree.flatten(output)

    # TODO: error out of the output is not exactly the same structure as the original output
    # TODO: this does not work even if compute_output_spec is implemented on a custom layer
    #       node.outputs is always the flattened list
    #       node.operation.output works but also contains the flattened list most of the time
    # tree.assert_same_structure(node.operation.output, output, check_types=False)

    # At least, error out when the number of returned items is different
    if len(node.output_tensors) != len(outputs):
        raise TypeError(f"Error in clone_layer_graph: the output returned from clone_fn "
                        f"must match the input of the following node. \n"
                        f"clone_fn returned {output} \n"
                        f"while the expected number of output tensors was: {len(node.output_tensors)}")

    # write the new outputs to "visited"
    for old_tensor, new_tensor in zip(node.output_tensors, outputs):
        visited.update({id(old_tensor): new_tensor})


def _walk_back_tensors(tensor_struct, visited, clone_fn, enter_nested):
    # tensor_struct will be an actual pytree on the first call:
    # functree_walk_back_tensors(model.output)
    # On subsequent calls, it will be a flat list because the function will be called as:
    # functree_walk_back_tensors(node.input_tensors) and node.input_tensors is a flat list.
    tensors = tree.flatten(tensor_struct)

    for tensor in tensors:  # tensors: list of flattened outputs

        # retrieve node that produced tensor
        node = _walkback_one_tensor(tensor)

        # if this node was visited before, its outputs are already in "visisted"
        if id(node) in visited:
            continue

        # mark the node as visited
        visited.update({id(node): True})

        # handle input node
        if _handle_input_node(node, visited):
            continue

        # handle the nested node case, potentially cloning the nest as new nested layer new_op
        new_op = _handle_nested_node(node, visited, clone_fn, enter_nested)

        # recursively continue iterating to gather all inputs
        _walk_back_tensors(node.input_tensors, visited, clone_fn, enter_nested)  # flattened

        # run the node again, which will clone it
        operation = new_op if new_op is not None else node.operation
        _clone_node(node, operation, visited, clone_fn)

    # collect output from the (updated) list of visited tensors
    outputs = [visited[id(tensor)] for tensor in tensors]
    output = tree.pack_sequence_as(tensor_struct, outputs)
    return output
