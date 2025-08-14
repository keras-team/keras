class MergeableGraph:
    """A graph that supports merging nodes."""

    def __init__(self):
        self._parent = {}
        self._edges = set()

    def get_root(self, node):
        if node not in self._parent:
            self._parent[node] = node
            return node
        if self._parent[node] == node:
            return node
        self._parent[node] = self.get_root(self._parent[node])
        return self._parent[node]

    def merge_nodes(self, node1, node2):
        root1 = self.get_root(node1)
        root2 = self.get_root(node2)
        if root1 != root2:
            self._parent[root1] = root2

    def add_edge(self, node1, node2):
        root1 = self.get_root(node1)
        root2 = self.get_root(node2)
        if root1 != root2:
            self._edges.add(tuple(sorted((root1, root2))))

    def get_edges(self):
        return self._edges
