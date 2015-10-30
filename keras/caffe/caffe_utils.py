layer_num_to_name = {
            0: 'NONE',
            1: 'ACCURACY',
            2: 'BNLL',
            3: 'CONCAT',
            4: 'CONVOLUTION',
            5: 'DATA',
            6: 'DROPOUT',
            7: 'EUCLIDEANLOSS',
            8: 'FLATTEN',
            9: 'HDF5DATA',
            10: 'HDF5OUTPUT',
            11: 'IM2COL',
            12: 'IMAGEDATA',
            13: 'INFOGAINLOSS',
            14: 'INNERPRODUCT',
            15: 'LRN',
            16: 'MULTINOMIALLOGISTICLOSS',
            17: 'POOLING',
            18: 'RELU',
            19: 'SIGMOID',
            20: 'SOFTMAX',
            21: 'SOFTMAXWITHLOSS',
            22: 'SPLIT',
            23: 'TANH',
            24: 'WINDOWDATA',
            25: 'ELTWISE',
            26: 'POWER',
            27: 'SIGMOIDCROSSENTROPYLOSS',
            28: 'HINGELOSS',
            29: 'MEMORYDATA',
            30: 'ARGMAX',
            31: 'THRESHOLD',
            32: 'DUMMY_DATA',
            33: 'SLICE',
            34: 'MVN',
            35: 'ABSVAL',
            36: 'SILENCE',
            37: 'CONTRASTIVELOSS',
            38: 'EXP',
            39: 'DECONVOLUTION'}


def layer_type(layer):
    if type(layer.type) == int:
        typ = layer_num_to_name[layer.type]
    else:
        typ = str(layer.type)
    return typ.lower()


def parse_network(layers, phase):
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
    network = acyclic(network)  # Convert it to be truly acyclic
    network = merge_layer_blob(network)  # eliminate 'blobs', just have layers
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


def is_data_input(layer):
    return layer_type(layer) in ['data', 'imagedata', 'memorydata', 'hdf5data', 'windowdata']


def remove_label_paths(layers, network, inputs, outputs):
    '''
        Input Data -> Loss Layer connection(the label) is removed
    '''
    for input in inputs:
        for output in outputs:
            if output in network[input] and is_data_input(layers[input]):
                network[input].remove(output)
    return network


def get_inputs(network):
    '''
        Gets the starting points of the network
    '''
    reverse_network = reverse(network)
    return get_outputs(reverse_network)


def get_outputs(network):
    '''
        Gets the ending points of the network
    '''
    outputs = []
    for node in network.keys():
        if network[node] == []:
            outputs.append(node)
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
    if layer_type(layer) == 'data' or layer_type(layer) == 'imagedata':
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
