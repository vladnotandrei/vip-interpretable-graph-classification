from torch_geometric.transforms import BaseTransform, Compose
import torch

### FEATURE MAPPINGS ###

def get_node_feature_label_mapping():
    # From Mutagenicity dataset README
    index_to_atom = {
        0: 'C',  1: 'O',  2: 'Cl',  3: 'H',  4: 'N',
        5: 'F',  6: 'Br', 7: 'S',   8: 'P',  9: 'I',
        10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'
    }
    return index_to_atom


def get_edge_feature_label_mapping():
    return {0: '1', 1: '2', 2: '3'}  # From Mutagenicity dataset README


def get_graph_label_mapping():
    return {0: 'mutagen', 1: 'nonmutagen'}  # From Mutagenicity dataset README

### OTHER FUNCTIONS ###

# def is_mutagen(data):
#     if data.graph_class_label == 'mutagen':
#         return True
#     elif data.graph_class_label == 'nonmutagen':
#         return False
#     else:
#         raise Exception('Unknown class label. Must be \'mutagen\' or \'nonmutagen\'')


### FEATURE TRANSFORMS ###

class MapNodeLabels(BaseTransform):
    def __init__(self):
        super().__init__()
        self.mapping = get_node_feature_label_mapping()

    def __call__(self, data):
        one_hot_indices = data.x.argmax(dim=1)  # Get feature idx for each node
        data.atoms = [self.mapping[idx.item()] for idx in one_hot_indices]
        return data


class MapEdgeLabels(BaseTransform):

    # def remove_duplicate_edges(data):
    #     """
    #     Removes duplicate edges from the data object by keeping only one direction,
    #     in order to represent as an undirected graph.
    #     """
    #     edge_index = data.edge_index.clone()
    #     edge_attr = data.edge_attr.clone()

    #     # Remove duplicate edges
    #     sorted_edges, _ = torch.sort(edge_index, dim=0)  # Sort so first tuple item <= second tuple item
    #     unique_edges, inverse_indices = torch.unique(sorted_edges, dim=1, sorted=False, return_inverse=True)
    #     unique_edge_attr = edge_attr[inverse_indices]  # Select corresponding bond attributes
    #     # #TODO: ^ this doesn't produce 16 edges as it should. It produces 32 still

    #     data.undir_edge_index = unique_edges
    #     data.undir_edge_attr = unique_edge_attr

    #     return data


    def __init__(self):
        super().__init__()
        self.mapping = get_edge_feature_label_mapping()


    def __call__(self, data):
        # data = MapEdgeLabels.remove_duplicate_edges(data)
        one_hot_indices = data.edge_attr.argmax(dim=1)  # Get feature idx for each node
        data.bonds = [self.mapping[idx.item()] for idx in one_hot_indices]
        return data


class MapGraphClassLabel(BaseTransform):
    def __init__(self):
        super().__init__()
        self.mapping = get_graph_label_mapping()

    def __call__(self, data):
        data.mutagenicity = self.mapping[data.y.item()]
        return data


def get_combined_mapping_transform():
    map_node_labels = MapNodeLabels()
    map_edge_labels = MapEdgeLabels()
    map_graph_class_labels = MapGraphClassLabel()
    combined_transform = Compose([map_node_labels, map_edge_labels, map_graph_class_labels])
    return combined_transform