from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import to_networkx
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Fragments
import pandas as pd

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
    return {0: 1, 1: 2, 2: 3}  # From Mutagenicity dataset README


def get_graph_label_mapping():
    return {0: 'mutagen', 1: 'nonmutagen'}  # From Mutagenicity dataset README


### FEATURE TRANSFORMS ###

class MapNodeLabels(BaseTransform):
    def __init__(self):
        super().__init__()
        self.mapping = get_node_feature_label_mapping()


    def __call__(self, data):
        one_hot_indices = data.x.argmax(dim=1)  # Get feature idx for each node
        data.atom = [self.mapping[idx.item()] for idx in one_hot_indices]
        return data


class MapEdgeLabels(BaseTransform):
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
    

class NumUndirectedEdges(BaseTransform):
    def __init__(self):
        super().__init__()


    def __call__(self, data):
        if data.num_edges % 2 != 0:
            raise Exception('Graph in dataset is not undirected. Cannot count number of undirected edges.')
        else:
            data.num_undirected_edges = int(data.num_edges / 2)
        return data


def get_combined_mapping_transform():
    map_node_labels = MapNodeLabels()
    map_edge_labels = MapEdgeLabels()
    map_graph_class_labels = MapGraphClassLabel()
    num_undirected_edges = NumUndirectedEdges()
    combined_transform = Compose([map_node_labels, map_edge_labels, map_graph_class_labels, num_undirected_edges])
    return combined_transform


### DATA PROCESSING ###

def raw_to_nx(data):
    G = to_networkx(data, to_undirected=True, edge_attrs=['bonds'], node_attrs=['atom'])
    return G


def nx_to_rdkit(G):
    """
    Convert a NetworkX graph G to an RDKit molecule.
    
    Assumes:
    - Each node has an attribute "atom" that contains the element symbol (e.g., "C", "O", etc.)
    - Each edge has an attribute "bond" that is an integer (1, 2, or 3) representing the bond order.
    """
    # Create an editable RDKit molecule
    rwmol = Chem.RWMol()
    
    # Map NetworkX nodes to RDKit atom indices
    node_to_idx = {}
    for node, data in G.nodes(data=True):
        atom_symbol = data.get('atom')
        if atom_symbol is None:
            raise ValueError(f'Node {node} does not have an \'atom\' attribute')
        atom = Chem.Atom(atom_symbol)
        idx = rwmol.AddAtom(atom)
        node_to_idx[node] = idx

    # Add bonds between atoms
    for u, v, data in G.edges(data=True):
        bond_order = data.get('bonds')
        if bond_order is None:
            raise ValueError(f'Edge ({u}, {v}) does not have a \'bonds\' attribute')
        if bond_order == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.rdchem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.rdchem.BondType.TRIPLE
        else:
            raise ValueError(f'Unsupported bond order {bond_order} on edge ({u}, {v})')
        
        # Add the bond using the mapped indices
        rwmol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Check if the atom is nitrogen with 4 bonds, and if it is, give it formal charge +1
    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() == 'N':
            total_bonds = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if total_bonds == 4:  # Overbonded N atom
                atom.SetFormalCharge(1)
    
    # Check if the atom is oxygen with 3 bonds, and if it is, give it formal charge +1
    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() == 'O':
            total_bonds = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if total_bonds == 3:  # Overbonded oxygen
                atom.SetFormalCharge(1)

    mol = rwmol.GetMol()  # Convert to a standard RDKit Mol object
    Chem.SanitizeMol(mol)  # Sanitize the molecule (this checks valences, computes aromaticity, etc.)
    return mol


def apply_all_fragments_dynamic(mol):
    # Discover all fragment-checking functions in the module (Should be 85)
    fragment_funcs = [func for func in dir(Fragments) if func.startswith('fr_') and callable(getattr(Fragments, func))]
    
    # Apply each function to the molecule
    results = {}
    for func_name in fragment_funcs:
        func = getattr(Fragments, func_name)
        try:
            result = func(mol)  # Counts occurences of func group, not including spatial permutations
            results[func_name] = result
        except Exception as e:
            results[func_name] = f"Error: {str(e)}"
    return results


def fragment_occurence_counts_onehot(mol, frag_func_names, count_list):
    """
    Input:
    mol: rdkit molecule
    frag_func_names, str: list of Rdkit.Chem.Fragments function names 
    count_list, list(list(int)): list of lists of counts (ints) to check for

    Output:
    onehot: list of 1 and -1, indicating True or False for occurrence count of a fragment respectively
    """
    onehot = []
    for func_name, counts in zip(frag_func_names, count_list):
        func = getattr(Fragments, func_name)
        result = func(mol)
        for c in counts:
            if result == c:
                onehot.append(1)
            else:
                onehot.append(-1)
    return onehot


def onehot_to_interpretable_dict(onehot):
    """
    Input:
    onehot: list of 1 and -1, indicating True or False for occurrence count of a fragment respectively
    frag_func_names, str: list of Rdkit.Chem.Fragments function names 
    count_list, list(list(int)): list of lists of counts (ints) to check for

    Output:
    frag_count_dict: interpreatable dict of count for each fragment type
    """
    df = pd.read_csv('./experiments/rdkit_querysets/queryset_1.csv')
    frag_func_names = df['frag_func_name'].to_list()
    count_list = [eval(n) for n in df['count_list'].to_list()]

    # Check if for size equivalencies between onehot and (frag_func_names * count_list items)
    num_frag_counts = 0
    for counts in count_list:
        num_frag_counts += len(counts)
    if num_frag_counts != len(onehot):
        raise Exception('Different number of frag counts than onehot elements. They must be the same.')

    frag_count_dict = {}
    i = 0
    for frag, counts in zip(frag_func_names, count_list):
        for c in counts:
            if onehot[i] == 1:
                frag_count_dict[frag] = c
            i += 1
    
    return frag_count_dict