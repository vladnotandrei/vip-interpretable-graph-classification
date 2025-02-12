import networkx as nx
import itertools


def node_match(node1, node2):
    """
    Input:
    dict node1: {'atom': atom_name}
    dict node2: {'atom': atom_name}

    Output:
    bool : True if node features are the same, False otherwise
    """
    return node1['atom'] == node2['atom']


def edge_match(edge1, edge2):
    """
    Input:
    dict edge1: {'bonds': bond_count}
    dict edge2: {'bonds': bond_count}

    Output:
    bool : True if edge features are the same, False otherwise
    """
    return edge1['bonds'] == edge2['bonds']


def unique_subgraph_isomorphisms(matcher):
    """Return one representative for each unique subgraph match."""
    seen_subgraphs = []  #set()
    unique_matches = []
    
    for mapping in matcher.subgraph_isomorphisms_iter():
        # The subgraph matched in G is just the set of G-nodes in 'mapping.keys()'.
        matched_nodes = set(mapping.keys())  #frozenset(mapping.keys())  # Convert to a frozenset so we can store it in a set.
        
        if matched_nodes not in seen_subgraphs:
            # seen_subgraphs.add(matched_nodes)  # if using a set
            seen_subgraphs.append(matched_nodes)
            unique_matches.append(mapping)
            
    return unique_matches


def unique_subgraph_isomorphisms_v2(matcher):
    """Return one representative for each unique subgraph match. Used sets instead of list"""
    seen_subgraphs = set()
    unique_matches = []
    
    for mapping in matcher.subgraph_isomorphisms_iter():
        # The subgraph matched in G is just the set of G-nodes in 'mapping.keys()'.
        matched_nodes = frozenset(mapping.keys())  # Convert to a frozenset so we can store it in a set.
        
        if matched_nodes not in seen_subgraphs:
            # seen_subgraphs.add(matched_nodes)  # if using a set
            seen_subgraphs.add(matched_nodes)
            unique_matches.append(mapping)
            
    return unique_matches


def count_node_induced_subgraph_isomorphisms(G, SG, use_v2=False):
    """
    Input:
    G: networkx graph to searh
    SG: networkx subgraph that will be search for in G

    Output:
    count: int, number of isomorphisms found 
    """

    isomatcher = nx.isomorphism.GraphMatcher(G, SG, node_match=node_match, edge_match=edge_match)
    if use_v2:
        unique_mappings = unique_subgraph_isomorphisms_v2(isomatcher)
    else:
        unique_mappings = unique_subgraph_isomorphisms(isomatcher)
    count = len(unique_mappings)

    # TODO: Return the nodes/edges in G that are isomorphic to SG

    return count, unique_mappings


### Naive slow subgraph enumeration algorihtm ###

def extract_all_motifs_in_graph(G, min_num_nodes_in_motif, max_num_nodes_in_motif):
    nodes = list(G.nodes())
    all_motifs = []
    for k in range(min_num_nodes_in_motif, max_num_nodes_in_motif + 1):
        print(k)
        for subset in itertools.combinations(nodes, k):
            SG = G.subgraph(subset)  # Induced subgraph
            if nx.is_connected(SG):  # Must be connected (No isolated nodes)
                all_motifs.append(SG)
    return all_motifs


### Faster subgraph enumeration algorithm ###
 
def recursive_local_expand(G, node_set, possible, excluded, results, max_size):
        """
        Recursive function to add an extra node to the subgraph being formed
        """
        results.append(node_set)
        if len(node_set) == max_size:  # Should this be max_size + 1?
            return
        for node in possible - excluded:
            new_node_set = node_set | {node}
            excluded = excluded | {node}
            new_possible = (possible | set(G.neighbors(node))) - excluded
            recursive_local_expand(G, new_node_set, new_possible, excluded, results, max_size)


def get_all_connected_subgraphs(G, min_motif_size, max_motif_size):
    """Get all connected subgraphs by a recursive procedure"""

    # Connected components sorted in decreasing order by number of nodes
    con_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]  
   
    results = []
    for cc in con_comp:
        max_size = max_motif_size  #len(cc)

        excluded = set()
        for node in G:
            excluded.add(node)
            node_set = {node}
            possible = set(G.neighbors(node)) - excluded
            recursive_local_expand(G, node_set, possible, excluded, results, max_size)

    results.sort(key=len)

    results_no_single_nodes = [motif for motif in results if len(motif) > 1]

    results_nx_objects = [G.subgraph(motif) for motif in results_no_single_nodes]

    return results_no_single_nodes, results_nx_objects