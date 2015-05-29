import pydot
from keras.layers.core import Merge
from keras.models import Model
from collections import Counter

class Grapher(object):

    def __init__(self):
        self.names = {}
        self.class_counts = Counter()

    def get_name(self, model):
        if hasattr(model, 'name'):
            return model.name
        cls = model.__class__.__name__
        if model not in self.names:
            self.class_counts[cls] += 1
            self.names[model] = cls + str(self.class_counts[cls])
        return self.names[model]

    def add_edge(self, f, t, graph):
        if f: graph.add_edge(pydot.Edge(f, t))
        return t

    def add_model(self, model, graph, parent=None):
        this = self.get_name(model)
        if isinstance(model, Model):
            parent = self.add_edge(parent, this, graph)
            for child in reversed(model.layers):
                parent = self.add_model(child, graph, parent)
        elif isinstance(model, Merge):
            for child in model.models:
                self.add_model(child, graph, this)
            return self.add_edge(parent, this, graph)
        else:
            return self.add_edge(parent, this, graph)

    def plot(self, model, to_file):
        graph = pydot.Dot(graph_type='graph')
        self.add_model(model, graph)
        graph.write_png(to_file)
