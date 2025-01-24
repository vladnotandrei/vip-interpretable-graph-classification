import networkx as nx


def node_match(node1, node2):
    """
    Input:
    dict node1: {'atom': atom_name}
    dict node2: {'atom': atom_name}

    Output:
    bool : True if node features are the same, False otherwise
    """
    return node1['atoms'] == node2['atoms']


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