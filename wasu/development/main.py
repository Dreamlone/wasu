from typing import Dict, Union


class WasuPipeline:
    """ Class for constructing pipelines for modelling """
    node_by_name = {}

    def __init__(self):
        self.nodes_to_execute = []

    def add_node(self, name: str, from_nodes: list, params: Union[Dict, None] = None):
        """ Adding block analysis """
        if params is None:
            node = self.node_by_name[name](from_nodes=from_nodes)
        else:
            node = self.node_by_name[name](from_nodes=from_nodes, **params)

        self.nodes_to_execute.append(node)

    def run(self):
        """ Launch constructed pipeline for analysis """
        pass
