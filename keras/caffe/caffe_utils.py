def make_network(layers, phase):
    '''
        Construct a network from the layers by making all blobs and layers(operations) as nodes.
    '''
    nb_layers = len(layers)
    network = {}

    for l in range(nb_layers):
        included = False
        try:
            # try to see if the layer is phase specific
            if layers[l].include[0].phase == phase:
                included = True
        except IndexError:
            included = True

        if included:
            layer_key = 'caffe_layer_' + str(l)  # actual layers, special annotation to mark them
            if layer_key not in network:
                network[layer_key] = []
            top_blobs = map(str, layers[l].top)
            bottom_blobs = map(str, layers[l].bottom)
            blobs = top_blobs + bottom_blobs
            for blob in blobs:
                if blob not in network:
                    network[blob] = []
            for blob in bottom_blobs:
                network[blob].append(layer_key)
            for blob in top_blobs:
                    network[layer_key].append(blob)
    return network


def acyclic(network):
    '''
        Make the network truly acyclic by removing in-place operations.
        Takes in a normal graph and returns a DAG.
        If an edge is a cycle of the form:
            node -> layer -> node -> (futher_layers)
        replace it by:
            node -> layer -> new_node -> (further_layers)
        where 'new_node' is same as 'node' without the edge to 'opertion'.
        This is applied recursively to eliminate all cycles.
        NOTE: Here, the 'layer' is a node. 'layer' -> 'blob' pairs are hence formed
    '''
    for node in network.keys():
        if is_caffe_layer(node):
            continue  # actual layer - they cannot initiate cycles
        i = 0
        while i < len(network[node]):
            next_node = network[node][i]
            if node in network[next_node]:
                # loop detected: -> node -> next_node -> node ->
                # change it to: -> node -> next_node -> new_node ->
                new_node = node + '_' + str(i)  # create a additional node - 'new_node'
                network[node].remove(next_node)  # 'new_node' has all other edges but the current loop
                network[new_node] = network[node]
                network[node] = [next_node]  # point 'node' to 'next_node' only
                network[next_node] = [new_node]  # 'next_node' points to 'new_node'
                # update loops in 'new_node' to point at new_node and not at 'node'
                for n in network[new_node]:
                    if network[n] == [node]:
                        network[n] = [new_node]
                node = new_node
                i = 0
            else:
                i += 1

    return network


def merge_layer_blob(network):
    '''
        The 'layer' -> 'blob' pair of nodes is reduced to a single node
    '''
    net = {}
    for node in network:
        if is_caffe_layer(node):
            new = sanitize(node)
            if node not in net:
                net[new] = []
            for next in network[node]:
                nexts = map(sanitize, network[next])
                net[new].extend(nexts)
    return net


def reverse(network):
    '''
        Reverses a network
    '''
    rev = {}
    for node in network.keys():
        rev[node] = []
    for node in network.keys():
        for n in network[node]:
            rev[n].append(node)
    return rev


def remove_label_paths(network, starts, ends):
    '''
        Input Data -> Loss Layer connection(the label) is removed
    '''
    for start in starts:
        for end in ends:
            if end in network[start]:
                network[start].remove(end)
    return network


def get_inputs(reverse_network):
    '''
        Gets the starting points of the network
    '''
    inputs = ()
    for node in reverse_network:
        if reverse_network[node] == []:
            inputs += (node,)
    return inputs


def get_outputs(network):
    '''
        Gets the ending points of the network
    '''
    outputs = ()
    for node in network.keys():
        if network[node] == []:
            outputs += (node,)
    return outputs


def is_caffe_layer(node):
    '''
        The node an actual layer
    '''
    if node.startswith('caffe_layer_'):
        return True
    return False


def sanitize(string):
    '''
        removes the added identification prefix 'caffe_layer_'
    '''
    return int(string[12:])


def get_data_dim(layer):
    '''
        Finds the input dimension by parsing all data layers for image and image transformation details
    '''
    layer_type_nb = int(layer.type)
    if layer_type_nb == 5 or layer_type_nb == 12:
        # DATA or IMAGEDATA layers
        try:
            scale = layer.transform_param.scale
            if scale <= 0:
                scale = 1
        except AttributeError:
            pass

        try:
            side = layer.transform_param.crop_size * scale
            return [3, side, side]
        except AttributeError:
            pass

        try:
            height = layer.image_param.new_height * scale
            width = layer.image_param.new_width * scale
            return [3, height, width]
        except AttributeError:
            pass
    return []
