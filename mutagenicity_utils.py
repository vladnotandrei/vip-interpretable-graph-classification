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


def nx_to_rdkit_old(G):
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


def nx_to_rdkit(G):
    """
    Convert a NetworkX graph G to an RDKit molecule with formal charge correction.
    
    Assumes:
      - Each node has an attribute "atom" (e.g., "C", "O", etc.)
      - Each edge has an attribute "bond" (integer: 1, 2, or 3)
      - No aromatic bonds or hydrogen atoms are added
      - Formal charges are inferred and assigned
    """
    # Default valences for common elements
    default_valences = {
        'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
        'Cl': 1, 'Br': 1, 'I': 1, 'S': 2, 'P': 3,
        'Na': 1, 'K': 1, 'Li': 1, 'Ca': 2
    }

    rwmol = Chem.RWMol()
    node_to_idx = {}

    # Map NetworkX nodes to RDKit atom indices
    for node, data in G.nodes(data=True):
        atom_symbol = data.get('atom')
        if atom_symbol is None:
            raise ValueError(f'Node {node} does not have an "atom" attribute')
        atom = Chem.Atom(atom_symbol)
        idx = rwmol.AddAtom(atom)
        node_to_idx[node] = idx

    # Add bonds
    for u, v, data in G.edges(data=True):
        bond_order = data.get('bonds')
        if bond_order is None:
            raise ValueError(f'Edge ({u}, {v}) does not have a "bonds" attribute')
        if bond_order == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.rdchem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.rdchem.BondType.TRIPLE
        else:
            raise ValueError(f'Unsupported bond order {bond_order} on edge ({u}, {v})')
        rwmol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)  # Add the bond using the mapped indices

    # Calculate and assign formal charges
    mol = rwmol.GetMol()
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        actual_valence = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
        expected_valence = default_valences[symbol]
        if symbol == 'P' and actual_valence == 5:
            atom.SetFormalCharge(+0)  # Explicitly allow it (RDKit requires SetFormalCharge even if it's zero)
        elif (symbol in default_valences) and expected_valence is not None:
            formal_charge = int(actual_valence - expected_valence)
            atom.SetFormalCharge(formal_charge)
        else:
            raise ValueError(f'Unsupported atom symbol {symbol} or unexpected valence in the graph')

    # Validate molecule without mutating it
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    except Exception as e:
        raise ValueError(f"Invalid molecule due to valence or structure: {e}")
    
    return mol


def raw_to_rdkit(data):
    return nx_to_rdkit(raw_to_nx(data))


def get_all_fragment_func_names():
    return [func for func in dir(Fragments) if func.startswith('fr_') and callable(getattr(Fragments, func))]


def apply_all_fragments_dynamic(mol):
    # Discover all fragment-checking functions in the module (Should be 85)
    fragment_funcs = get_all_fragment_func_names()
    
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


def mol_to_query_answers(mol, frag_query_map, device):
    """
    Input:
    mol: rdkit molecule
    frag_query_map: dict of {str <frag_name>: list of int <counts>} for each fragment type

    Output:
    onehot: list of 1 and -1, indicating True or False for occurrence count of a fragment respectively
    """
    qry_ans_vec = torch.full((1, sum(len(v) for v in frag_query_map.values())), -1, device=device)
    i = 0
    for frag_name, counts in frag_query_map.items():
        func_name = 'fr_' + frag_name  # Add 'fr_' prefix to match RDKit fragment function names
        func = getattr(Fragments, func_name)
        result = func(mol)
        for c in counts:
            if result == c:
                qry_ans_vec[0, i] = 1
            i += 1
    return qry_ans_vec


def answer_vec_to_interpretable_dict(qry_ans_vec, queryset_path):
    """
    Input:
    qry_ans_vec: query-answer np.array of 1 and -1, indicating True or False for occurrence count of a fragment respectively.

    Output:
    frag_count_dict: interpreatable dict of count for each fragment type {str <frag_name>: int <count>}
    """
    queries = pd.read_csv(queryset_path)
    unique_frag_names = queries['frag_name'].unique().tolist()

    frag_count_dict = {}
    for name in unique_frag_names:
        matched_idxs = queries[queries['frag_name'] == name].index.tolist()
        subvec = qry_ans_vec[matched_idxs]  # Should be a onehot vector

        argmax_idx = None
        for i, val in enumerate(subvec):
            if val == 1:
                if argmax_idx is not None:
                    raise Exception(f'Multiple 1 values found for fragment {name} in qry_ans_vec. Expected only one.')
                else:
                    argmax_idx = i
        if argmax_idx is None:
            raise Exception(f'No 1 value found for fragment {name} in qry_ans_vec. Expected at least one.')
        
        qry_idx = matched_idxs[argmax_idx]  # Get the index of the onehot vector in the original qry_ans_vec
        count = queries.iloc[qry_idx]['count']
        frag_count_dict[name] = count
    return frag_count_dict


### FIGURES ###

def create_posterior_prob_heatmap(mol, probs, queries, answers, threshold, show_funcgroups_in_mol=False, queryset_path=None, show_title=False, qry_need='', y_true='', y_pred_max='', y_pred_ip='', sample_id=''):
    """
    Input:
    mol: rdkit molecule object (optional). Include if you want image of molecule next to heatmap
    probs: (num_queries, 2) tensor of class probabilities [0,1]
    queries: (num_queries, queryset_size) tensor of onehot query vectors
    answers: (queryset_size) tensor of answers 'no'(-1) or 'yes' (1) to all queries for the given molecule
    threshold: float, [0, 1]
    show_funcgroups_in_mol: bool, if True, show functional groups in the molecule image below the heatmap
    queryset_path: path to the CSV file containing the query set
    show_title: bool, if True, show the title of the figure with sample information
    qry_need: int, number of queries needed to make prediction with IP for given probability threshold
    y_true: int, 0 or 1, true class label of the molecule
    y_pred_max: int, 0 or 1, class label predicted by the model
    y_pred_ip: int, 0 or 1, class label predicted by the model using IP
    sample_id: int, id of sample in original Mutagenicity dataset (optional)

    Output:
    fig, ax: matplotlib objects
    """
    queryset = pd.read_csv(queryset_path)
    row_labels, row_label_colours =  [], []
    for i, q_onehot in enumerate(queries):
        queryset_idx = torch.argmax(q_onehot).item()
        q = queryset.iloc[queryset_idx]
        row_labels.append(f'q{i+1}: {q["frag_name"]}={q["count"]}?')
        
        ans = answers[queryset_idx]
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

    # Convert from torch tensors to numpy arrays on CPU
    probs = probs.detach().cpu().numpy()
    queries = queries.detach().cpu().numpy()
    answers = answers.detach().cpu().numpy()

    ax = axs[0]
    im = ax.imshow(probs, cmap=cmap, vmin=0, vmax=1)  # Display data as heatmap with a fixed range (0 to 1)

    ax.set_xticks(np.arange(probs.shape[1]))
    ax.set_yticks(np.arange(probs.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Increase size of x-axis and y-axis labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Change the colour of each y-tick label for 'yes' (green) and 'no' (red)
    for tick_label, colour in zip(ax.get_yticklabels(), row_label_colours):
        tick_label.set_color(colour)

    # Rotate the x-axis tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create a colorbar with specific tick labels.
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_yticklabels([str(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    cbar.set_label('Class Probability', rotation=270, labelpad=20, fontsize=16)

    # Increase size of colour bar ticks
    cbar.ax.tick_params(labelsize=14)

    if show_title == True:
        # Set figure title
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

        if show_funcgroups_in_mol == True:
            # Add text below molecule image (with functional group counts in the molecule)
            frag_count_dict = answer_vec_to_interpretable_dict(answers, queryset_path)
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