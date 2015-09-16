import pydot
# old pydot will not work with python3, must use one
# that works with python3 such as pydot2 or pydot


def plot(model, to_file='model.png'):

    graph = pydot.Dot(graph_type='digraph')
    if type(model) == Sequential:
        previous_node = None
        written_nodes = []
        n = 1
        for node in model.get_config()['layers']:
            # append number in case layers have same name to differentiate
            if (node['name'] + str(n)) in written_nodes:
                n += 1
            current_node = pydot.Node(node['name'] + str(n))
            written_nodes.append(node['name'] + str(n))
            graph.add_node(current_node)
            if previous_node:
                graph.add_edge(pydot.Edge(previous_node, current_node))
            previous_node = current_node
        graph.write_png(to_file)

    elif type(model) == Graph:
        # don't need to append number for names since all nodes labeled
        for input_node in model.input_config:
            graph.add_node(pydot.Node(input_node['name']))

        # intermediate and output nodes have input defined
        for layer_config in [model.node_config, model.output_config]:
            for node in layer_config:
                graph.add_node(pydot.Node(node['name']))
                # possible to have multiple 'inputs' vs 1 'input'
                if node['inputs']:
                    for e in node['inputs']:
                        graph.add_edge(pydot.Edge(e, node['name']))
                else:
                    graph.add_edge(pydot.Edge(node['input'], node['name']))

        graph.write_png(to_file)
