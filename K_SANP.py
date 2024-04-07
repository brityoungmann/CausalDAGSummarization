import networkx as nx
import itertools
import Utils


def same_parents_and_children(dag, node1, node2):
    parents1 = set(dag.predecessors(node1))
    parents2 = set(dag.predecessors(node2))

    children1 = set(dag.successors(node1))
    children2 = set(dag.successors(node2))

    return parents1 == parents2 and children1 == children2
def k_snap(dag,k, similarity_df):
    if len(dag.nodes) <= k:
        return dag
    #first step
    G = merge_same_parents_children(dag,similarity_df)
    #second step
    while len(G.nodes) > k:
        G = merege_similar_clusters(G,dag,similarity_df)
    #Utils.show_dag(G)
    return G



def get_similarity(node1,node2,dag,similarity_df, G):
  
    nodes1 = node1.split('_')
    nodes2 = node2.split('_')
    sim = 0
    if Utils.a_valid_pair(node1,node2,dag,similarity_df, G):

        #(num edges from nodes1 to nodes2 + num edges from nodes2 to nodes1)/(|nodes1| + |nodes2|)
        num1 = count_edges_between_nodes(dag, nodes1, nodes2)
        num2 = count_edges_between_nodes(dag, nodes2, nodes1)
        sim = (num1 + num2)/float(len(nodes1) + len(nodes2))
    return sim


def count_edges_between_nodes(graph, nodes1, nodes2):
    edges_count = 0

    for node1 in nodes1:
        for node2 in nodes2:
            if graph.has_edge(node1,node2):
                edges_count = edges_count + 1

    return edges_count
def merege_similar_clusters(G,dag,similarity_df,verbos = False):
    node_pairs = itertools.combinations(G.nodes(), 2)
    max_sim = 0
    max_pair = []
    for pair in node_pairs:
        node1 = pair[0]
        node2 = pair[1]
        sim = get_similarity(node1,node2,dag,similarity_df, G)
        if verbos:
            print(pair,sim)
        if sim >= max_sim:
            max_sim = sim
            max_pair = pair
    if max_sim == 0:
        node_pairs = itertools.combinations(G.nodes(), 2)
        max_pair = []
        print("could not find a pair to merge, merging a random valid pair")
        for pair in node_pairs:
            node1 = pair[0]
            node2 = pair[1]

            if Utils.a_valid_pair(node1, node2, dag,similarity_df, G):
                max_pair = pair
    node1 = max_pair[0]
    node2 = max_pair[1]
    #print("choose to merge: ", node1, node2)

    G = nx.contracted_nodes(G, node1, node2, self_loops=False)
    new_node_name = node1 + '_' + node2
    G = nx.relabel_nodes(G, {node1: str(new_node_name), node2: str(new_node_name)})
    return G

def merge_same_parents_children(dag,similarity_df):
    nodes = dag.nodes()
    to_merge = []
    node_pairs = itertools.combinations(nodes, 2)
    for pair in node_pairs:
        n1 = pair[0]
        n2 = pair[1]
        if not n1 == n2:
            if Utils.semantic_sim(n1,n2,similarity_df):
                if same_parents_and_children(dag, n1, n2):
                    to_merge.append((n1, n2))
    G = dag.copy()
    for pair in to_merge:
        node1 = pair[0]
        node2 = pair[1]
        if node1 in G.nodes and node2 in G.nodes:
            G = nx.contracted_nodes(G, node1, node2, self_loops=False)
            new_node_name = node1 + '_' + node2
            G = nx.relabel_nodes(G, {node1: str(new_node_name), node2: str(new_node_name)})
        # already in a cluster
        else:
            node1, node2 = find_cluster_node(G, node1, node2)
            if node1 in G.nodes and node2 in G.nodes:
                G = nx.contracted_nodes(G, node1, node2, self_loops=False)
                new_node_name = f"{node1}_{node2}"
                G = nx.relabel_nodes(G, {new_node_name: (node1, node2)})
    return G


def find_cluster_node(G, node1, node2):
    if not node1 in G.nodes:
        for n in G.nodes:
            if node1 in n:
                node1 = n
                break
    if not node2 in G.nodes:
        for n in G.nodes:
            if node2 in n:
                node2 = n
                break
    return node1,node2


