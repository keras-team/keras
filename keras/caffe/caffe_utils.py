def is_layer(node):
	if node.startswith('__') and node.endswith('__'):
		return True
	return False

def clean_up(string):
	return int(string[2:-2])

def unfold(network):
	'''
		Takes in a normal graph and returns an unfolded version of it, by removing all node-node cycles
	'''
	for node in network.keys():
		if is_layer(node):
			continue # actual Layer
		i = 0
		while i < len(network[node]):
			out_node = network[node][i]
			if node in network[out_node]:
				# loop detected: -> node -> out_node -> node ->
				new_node = node + '_' + str(i)
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
			else:
				i += 1
	return network

def get_network(layers, phase):
	nb_layers = len(layers)
	network = {}
	for l in range(nb_layers):
		included = False

		try:	#try to see if the layer is phase specific
			if layers[l].include[0].phase == phase:
				included = True
		except IndexError:
			included = True

		if included:
			layer_key = '__' + str(l) + '__'	# actual layers, special annotation to mark them
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

def merge(network):
	#let us merge here
	net = {}
	for node in network:
		if node.startswith('__') and node.endswith('__'):
			new = clean_up(node)
			if not net.has_key(node):
				net[new] = []
			for next in network[node]:
				nexts = map(clean_up, network[next])
				net[new].extend(nexts)
	return net

def reverse_net(net):
	rev_net = {}
	for node in net.keys():
		rev_net[node] = []
	for node in net.keys():
		for n in net[node]:
			rev_net[n].append(node)
	return rev_net

def remove_label(net, start):
	#kick out the label
	for node in start:
		for end in ends:
			if end in net[node]:
				net[node].remove(end)
	return net

def get_start(rev_net):
	start = ()
	for node in rev_net:
		if rev_net[node] == []:
			start += (node,)
	return start

def get_ends(net):
	ends = ()
	for node in net.keys():
		if net[node] == []:
			ends += (node,)
	return ends