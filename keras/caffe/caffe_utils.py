def make_network(layers, phase):
	'''
		Construct a network from the layers by making all blobs and layers(operations) as nodes.
	'''
	nb_layers = len(layers)
	network = {}

	for l in range(nb_layers):
		included = False
		try:
			#try to see if the layer is phase specific
			if layers[l].include[0].phase == phase:
				included = True
		except IndexError:
			included = True

		if included:
			layer_key = 'caffe_layer_' + str(l)	# actual layers, special annotation to mark them
			if not network.has_key(layer_key):
				network[layer_key] = []
			top_blobs = map(str, layers[l].top)
			bottom_blobs = map(str, layers[l].bottom)
			blobs = top_blobs + bottom_blobs
			for blob in blobs:
				if not network.has_key(blob):
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
			continue	# actual layer
		i = 0
		while i < len(network[node]):
			out_node = network[node][i]
			if node in network[out_node]:
				# loop detected: -> node -> out_node -> node ->
				new_node = node + '_'
				# -> node -> out_node -> new_node ->
				network[node].remove(out_node)	#create a new_node that has all but the current loop
				network[new_node] = network[node]
				network[node] = [out_node]	# point old node to out_node only
				network[out_node] = [new_node]	# out_node points to new_node
				# update loops in new_node to point at new_node and not at node
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
			if not net.has_key(node):
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
		for n in network[node]:#edit
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

def get_starts(reverse_network):
	'''
		Gets the starting point of the network(inputs)
	'''
	starts = ()
	for node in reverse_network:
		if reverse_network[node] == []:
			starts += (node,)
	return starts

def get_ends(network):
	'''
		Gets the ending point of the network(outputs)
	'''
	ends = ()
	for node in network.keys():
		if network[node] == []:
			ends += (node,)
	return ends

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