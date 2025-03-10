from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import to_networkx
import torch

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Fragments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def get_fragment_names_and_counts():
    df = pd.read_csv('./experiments/rdkit_querysets/queryset_1.csv')
    frag_names = [name[3:] for name in df['frag_func_name'].to_list()]  # Remove the starting "fr_" at beginning of name
    count_list = [eval(n) for n in df['count_list'].to_list()]
    return frag_names, count_list


def onehot_to_interpretable_dict(onehot):
    """
    Input:
    onehot: query-answer list of 1 and -1, indicating True or False for occurrence count of a fragment respectively

    Output:
    frag_count_dict: interpreatable dict of count for each fragment type {str <frag_name>: int <count>}
    """
    frag_names, count_list = get_fragment_names_and_counts()

    # Check if for size equivalencies between onehot and (frag_func_names * count_list items)
    num_frag_counts = 0
    for counts in count_list:
        num_frag_counts += len(counts)
    if num_frag_counts != len(onehot):
        raise Exception('Different number of frag counts than onehot elements. They must be the same.')

    frag_count_dict = {}
    i = 0
    for frag, counts in zip(frag_names, count_list):
        for c in counts:
            if onehot[i] == 1:
                frag_count_dict[frag] = c
            i += 1
    return frag_count_dict


def get_query_name_list():
    frag_name, count_list = get_fragment_names_and_counts()
    query_names = []
    for frag, frag_counts in zip(frag_name, count_list):
        for count in frag_counts:
            q_name = f'{frag}={count}'
            query_names.append(q_name)
    return query_names


def onehot_to_query_name(onehot):
    q_idx = torch.argmax(onehot).item()
    q_name = get_query_name_list()[q_idx]
    return q_name


### FIGURES ###

def create_posterior_prob_heatmap(probs, queries, answers, y_true, y_pred_max, y_pred_ip, qry_need, threshold, sample_id=None, mol=None):
    """
    Input:
    probs: (num_queries, 2) tensor of class probabilities [0,1]
    queries: (num_queries, queryset_size) tensor of onehot query vectors
    answers: (queryset_size) tensor of all queries and answers no (-1) or yes (1) 
    y_true: int, 0 or 1
    y_pred_max: int, 0 or 1
    y_pred_ip: int, 0 or 1
    qry_need: int, number of queries needed to make prediction with IP for given probability threshold
    threshold: float, [0, 1]
    sample_id: int, id of sample in original Mutagenicity dataset (optional)
    mol: rdkit molecule object (optional). Include if you want image of molecule next to heatmap

    Output:
    fig, ax: matplotlib objects
    """

    row_labels, row_label_colours =  [], []
    for i, q in enumerate(queries):
        row_labels.append(f'{i+1}. {onehot_to_query_name(q)}')  # query names from onehot

        ans = answers[torch.argmax(q, dim=0)]
        if ans == 1:
            row_label_colours.append('green')
        elif ans == -1:
            row_label_colours.append('red')
        else:
            raise Exception('Invalid encoded answers. Must be -1 or 1.')

    col_labels = list(get_graph_label_mapping().values())  # class names from idx

    cmap = LinearSegmentedColormap.from_list("BlueRed", ["blue", "red"])  # colormap from blue (0) to red (1)
    
    ncols = 1 if mol == None else 2
    fig, axs = plt.subplots(figsize=(10, 6), ncols=ncols)

    ### HEAT MAP ###

    probs = probs.detach().cpu().numpy()
    queries = queries.detach().cpu().numpy()
    answers = answers.detach().cpu().numpy()

    ax = axs[0]
    im = ax.imshow(probs, cmap=cmap, vmin=0, vmax=1)  # Display data as heatmap with a fixed range (0 to 1)

    ax.set_xticks(np.arange(probs.shape[1]))
    ax.set_yticks(np.arange(probs.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Change the colour of each y-tick label for 'yes' (green) and 'no' (red)
    for tick_label, colour in zip(ax.get_yticklabels(), row_label_colours):
        tick_label.set_color(colour)

    # Rotate the x-axis tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create a colorbar with specific tick labels.
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_yticklabels([str(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])

    y_true_name = get_graph_label_mapping()[y_true]
    y_pred_max_name = get_graph_label_mapping()[y_pred_max]
    y_pred_ip_name = get_graph_label_mapping()[y_pred_ip]
    title = f"Posterior Probability:\nsample_id={sample_id}\ny_true={y_true_name}\ny_pred_max={y_pred_max_name}\ny_pred_ip={y_pred_ip_name}\nqry_need={qry_need}, threshold={threshold}\n"
    ax.set_title(title)

    ### Molecule Image ###
    
    if mol is not None:
        ax = axs[1]
        img = Draw.MolToImage(mol, size=(500, 500))
        img_array = np.array(img)
        # height = img_array.shape[0]
        # ax.imshow(img_array, extent=[0, 1, height, 0], aspect='auto')
        ax.imshow(img_array)
        ax.set_anchor('N')
        ax.axis("off")

        # Add text below molecule image (with functional group counts in the molecule)
        frag_count_dict = onehot_to_interpretable_dict(answers)
        text = ''
        for key, val in frag_count_dict.items():
            if val > 0:
                text += f'{key}: {val},\n'
        text = text[:-1]  # Remove last \n character
        ax.text(0.5, 0, text, ha="center", va="top", transform=ax.transAxes, fontsize=9)
        # ax.set_title(text)

    plt.tight_layout()
    # plt.close(fig)
    return fig, ax