import os

try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
if not pydot.find_graphviz():
    raise RuntimeError('Failed to import pydot. You must install pydot'
                       ' and graphviz for `pydotprint` to work.')


def model_to_dot(model, show_shapes=False, show_layer_names=True):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if model.__class__.__name__ == 'Sequential':
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # first, populate the nodes of the graph
    for layer in layers:
        layer_id = str(id(layer))
        if show_layer_names:
            label = str(layer.name) + ' (' + layer.__class__.__name__ + ')'
        else:
            label = layer.__class__.__name__

        if show_shapes:
            # Build the label that will actually contain a table with the
            # input/output
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # second, add the edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                # add edges
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot(model, to_file='model.png', show_shapes=False, show_layer_names=True):
    dot = model_to_dot(model, show_shapes, show_layer_names)
    _, format = os.path.splitext(to_file)
    if not format:
        format = 'png'
    else:
        format = format[1:]
    dot.write(to_file, format=format)


def figures(history,figure_name="plots"):
    from keras.callbacks import History  
    import matplotlib.pyplot as plt
  
    hist = history.history   
    epoch = history.epoch  
    acc = hist['acc']  
    loss = hist['loss']  
    val_loss = hist['val_loss']  
    val_acc  = hist['val_acc']
  
    plt.figure(1)  
  
    plt.subplot(221)  
    plt.plot(epoch,acc)  
    plt.title("Training accuracy vs Epoch")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")       
  
    plt.subplot(222)  
    plt.plot(epoch,loss)  
    plt.title("Training loss vs Epoch")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")    
  
    plt.subplot(223)  
    plt.plot(epoch,val_acc)  
    plt.title("Validation Acc vs Epoch")  
    plt.xlabel("Epoch")  
    plt.ylabel("Validation Accuracy")    
  
    plt.subplot(224)  
    plt.plot(epoch,val_loss)  
    plt.title("Validation loss vs Epoch")  
    plt.xlabel("Epoch")  
    plt.ylabel("Validation Loss")
    
    plt.tight_layout()  
    plt.savefig(figure_name)
