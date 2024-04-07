import math
import networkx as nx
import itertools
import Utils
import random
import K_SANP
from itertools import combinations
from collections import defaultdict
import heapq

def is_special_pair(graph, node1, node2):
    # Check if there is an edge between node1 and node2
    if graph.has_edge(node1, node2):
        # Check if node1 has no outgoing edges and node2 has one incoming and one outgoing edge
        if graph.out_degree(node1) == 0 and graph.in_degree(node2) == 1 and graph.out_degree(node2) == 1:
            return True
        if graph.out_degree(node2) == 0 and graph.in_degree(node1) == 1 and graph.out_degree(node1) == 1:
            return True
        if graph.in_degree(node1) == 0 and graph.in_degree(node2) == 1 and graph.out_degree(node2) == 1:
            return True
        if graph.in_degree(node2) == 0 and graph.in_degree(node1) == 1 and graph.out_degree(node1) == 1:
            return True
    return False

def low_cost_merges(dag, similarity_df, not_valid):
    nodes = dag.nodes()
    to_merge = []
    node_pairs = itertools.combinations(nodes, 2)
    for pair in node_pairs:
        n1 = pair[0]
        n2 = pair[1]
        if not Utils.a_valid_pair(n1,n2, dag, similarity_df,dag):
            not_valid.add(pair)
            continue
        if not n1 == n2:
            if '_' in n1 or '_' in n2:
                continue
            if Utils.semantic_sim(n1, n2, similarity_df):
                if is_special_pair(dag, n1, n2):
                    to_merge.append((n1, n2))
                elif zero_cost(dag, n1, n2):
                    to_merge.append((n1, n2))
    G = dag.copy()
    for pair in to_merge:
        node1 = pair[0]
        node2 = pair[1]
        if node1 in G.nodes and node2 in G.nodes:
            G = nx.contracted_nodes(G, node1, node2, self_loops=False)
            new_node_name = node1 + '_' + node2
            G = nx.relabel_nodes(G, {node1: str(new_node_name), node2: str(new_node_name)})

    return G, not_valid




def zero_cost(G, n1, n2):
    if G.has_edge(n1, n2) or G.has_edge(n2,n1):
        parents1 = set(G.predecessors(n1))
        if n2 in parents1:
            parents1.remove(n2)
        parents2 = set(G.predecessors(n2))
        if n1 in parents2:
            parents2.remove(n1)

        children1 = set(G.successors(n1))
        if n2 in children1:
            children1.remove(n2)
        children2 = set(G.successors(n2))
        if n1 in children2:
            children2.remove(n1)
        return parents1 == parents2 and children1 == children2
    else:
        parents1 = set(G.predecessors(n1))
        parents2 = set(G.predecessors(n2))
        children1 = set(G.successors(n1))
        children2 = set(G.successors(n2))
        return parents1 == parents2 and children1 == children2



def greedy(dag,nodes, recursive_basis,k, similarity_df):
    if len(dag.nodes) <= k:
        return dag

    # first step
    not_valid = set()
    G, not_valid = low_cost_merges(dag, similarity_df, not_valid)

    # heap = []
    #
    # # Insert elements into the heap
    # heapq.heappush(heap, 4)

    cost_scores = {}
    while len(G.nodes) > k:
        #G,recursive_basis = merege_pair(G,recursive_basis,nodes,dag, similarity_df)
        G, not_valid, cost_scores = fast_merege_pair(G, nodes, dag, similarity_df,
                                                     not_valid, cost_scores)
        #print(cost_scores)

    return G, recursive_basis




def not_in_ci(ci, nodes1,nodes2):
    nodes = ci[0].copy()
    nodes.update(ci[1])
    if len(ci[2]) > 0:
        nodes.update(ci[2])
    for n in nodes1:
        if n in nodes:
            return False
    for n in nodes2:
        if n in nodes:
            return False
    return True


def all_in(nodes,s):
    for n in nodes:
        if not n in s:
            return False
    return True

def get_similarity(node1,node2,recursive_basis):
    nodes1 = node1.split('_')
    nodes2 = node2.split('_')
    num_preserved = 0
    for ci in recursive_basis:
        if all_in(nodes1,ci[0]) and all_in(nodes2,ci[0]):
            num_preserved = num_preserved + 1
        elif all_in(nodes1, ci[1]) and all_in(nodes2, ci[1]):
            num_preserved = num_preserved + 1
        elif all_in(nodes1, ci[2]) and all_in(nodes2, ci[2]):
            num_preserved = num_preserved + 1
        elif not_in_ci(ci, nodes1,nodes2):
            num_preserved = num_preserved + 1
    return num_preserved

def merege_pair(G,recursive_basis,nodes,dag,similarity_df, verbos = False):
    node_pairs = itertools.combinations(G.nodes(), 2)
    max_sim = 0
    max_pair = []
    for pair in node_pairs:
        node1 = pair[0]
        node2 = pair[1]
        if Utils.a_valid_pair(node1,node2,dag, similarity_df):
            sim = get_similarity(node1,node2,recursive_basis)
            if verbos:
                print(pair,sim)
            if sim > max_sim:
                max_sim = sim
                max_pair = pair
            elif sim == max_sim:
                if random.choice(['True', 'False']):
                    max_sim = sim
                    max_pair = pair
    if len(max_pair) == 0:
        print("could not find a pair to merge, merging a random valid pair")
        for pair in node_pairs:
            node1 = pair[0]
            node2 = pair[1]
            if Utils.a_valid_pair(node1, node2, dag):
                max_pair = pair
    node1 = max_pair[0]
    node2 = max_pair[1]
    print("choose to merge: ", node1,node2)
    G = nx.contracted_nodes(G, node1, node2, self_loops=False)
    new_node_name = node1 + '_' + node2
    G = nx.relabel_nodes(G, {node1: str(new_node_name), node2: str(new_node_name)})
    recursive_basis = Utils.get_recursive_basis(G,nodes)#update_recursive_basis(node1,node2,recursive_basis)
    return G, recursive_basis

def fast_merege_pair(G,nodes,dag,similarity_df,not_valid, cost_scores, verbos = False):
    node_pairs = itertools.combinations(G.nodes(), 2)
    min_cost =math.inf
    max_pair = []

    for pair in node_pairs:
        node1 = pair[0]
        node2 = pair[1]
        if pair in not_valid:
            continue
        valid = Utils.a_valid_pair(node1,node2,dag, similarity_df, G)
        if valid == False:
            not_valid.add(pair)
        else:
            if pair in cost_scores:
                cost = cost_scores[pair]
            else:
                cost = get_cost(node1,node2,G)
                cost_scores[pair] = cost
            if verbos:
                print(pair,cost)
            if cost < min_cost:
                min_cost = cost
                max_pair = pair
            elif cost == min_cost:
                if random.choice(['True', 'False']):
                    min_cost = cost
                    max_pair = pair
    if len(max_pair) == 0:
        print("could not find a pair to merge, merging a random valid pair")
        for pair in node_pairs:
            if pair in not_valid:
                continue
            node1 = pair[0]
            node2 = pair[1]
            if Utils.a_valid_pair(node1, node2, dag):
                max_pair = pair
    node1 = max_pair[0]
    node2 = max_pair[1]

    cost_scores = update_cost_scores(cost_scores, node1, node2, G)
    #print("choose to merge: ", node1,node2)
    G = nx.contracted_nodes(G, node1, node2, self_loops=False)
    new_node_name = node1 + '_' + node2
    G = nx.relabel_nodes(G, {node1: str(new_node_name), node2: str(new_node_name)})

    return G, not_valid, cost_scores

def update_cost_scores(cost_scores, node1, node2, G):
    nodes = [node1,node2]
    nodes = nodes + list(G.predecessors(node1)) + list(G.successors(node1))
    nodes = nodes + list(G.predecessors(node2)) + list(G.successors(node2))
    to_remove = []
    for k in cost_scores:
        for n in nodes:
            if n in k:
                to_remove.append(k)
                break
    filtered_dict = {key: value for key, value in cost_scores.items() if key not in to_remove}
    return filtered_dict
def get_cost(node1,node2,G):
    cost = 0
    nodes1 = node1.split('_')
    nodes2 = node2.split('_')
    #edges among the new cluster
    if not G.has_edge(node1, node2):
        cost = cost + len(nodes1)*len(nodes2)

    #edges to parents and children
    parents1 = set(G.predecessors(node1))
    parents2 = set(G.predecessors(node2))

    #unique parents of node1
    if node2 in parents1:
        parents1.remove(node2)
    parents1 = parents1 - parents2#[p for p in parents1 if not p in parents2]
    cost = cost + len(parents1)*len(nodes2)

    #parents1 = list(G.predecessors(node1))
    # unique parents of node2
    if node1 in parents2:
        parents2.remove(node1)
    parents2 = parents2-parents1#[p for p in parents2 if not p in parents1]
    cost = cost + len(parents2) * len(nodes1)

    children1 = set(G.successors(node1))
    children2 = set(G.successors(node2))
    # unique children of node1
    if node2 in children1:
        children1.remove(node2)
    children1 = children1 - children2#[p for p in children1 if not p in children1]
    cost = cost + len(children1) * len(nodes2)

    #children1 = list(G.successors(node1))
    # unique children of node2
    if node1 in children2:
        children2.remove(node1)
    children2 = children2 - children1#[p for p in children2 if not p in children1]
    cost = cost + len(children2) * len(nodes1)

    return cost

def update_recursive_basis(node1,node2,recursive_basis):
    updated_recursive_basis = []
    nodes1 = node1.split('_')
    nodes2 = node2.split('_')
    for ci in recursive_basis:
        if all_in(nodes1, ci[0]) and all_in(nodes2, ci[0]):
            updated_recursive_basis.append(ci)
        elif all_in(nodes1, ci[1]) and all_in(nodes2, ci[1]):
            updated_recursive_basis.append(ci)
        elif all_in(nodes1, ci[2]) and all_in(nodes2, ci[2]):
            updated_recursive_basis.append(ci)
        elif not_in_ci(ci, nodes1, nodes2):
            updated_recursive_basis.append(ci)
    return updated_recursive_basis



def check_edges_between_sets(graph, t1, t2):
    for edge in graph.edges():
        source, target = edge
        if (source in t1) and (target in t2):
            continue
        elif (source in t1) and (target in t1):
            continue
        elif (source in t2) and (target in t2):
            continue
        else:
            return False
    return True


def split_into_subsets(lst, threshold,dag):
    valid_splits = []
    for r in range(threshold , len(lst) - threshold):
        for combo in combinations(lst, r):
            complement = [item for item in lst if item not in combo]
            if len(combo) >= threshold and len(complement) >= threshold:
                valid = False
                if check_edges_between_sets(dag, list(combo), complement):
                    valid = True
                elif check_edges_between_sets(dag, complement,list(combo)):
                    valid = True
                if valid:
                    valid_splits.append((list(combo), complement))
    return valid_splits


def main():
    dag_edges = [('A', 'B'), ('A','C'),('C', 'D'),('B','D'), ('D', 'E')]
    dag = nx.DiGraph(dag_edges)
    k = 3
    nodes = ['A','B','C','D','E']
    recursive_basis = [(set(['C']), set(['B']), set(['A'])),
                       (set(['D']), set(['A']), set(['B','C'])),
                       (set(['E']), set(['A', 'B','C']), set('D'))]

    # summary_dag = greedy(dag,nodes, recursive_basis,k,None)
    # Utils.show_dag(summary_dag)
    import Examples
    from cProfile import Profile
    from pstats import SortKey, Stats
    dag, k, nodes, recursive_basis, similarity_df = Examples.random_dag(60, 0.2)
    k = 40

    #summary_dag_greedy, recursive_basis_greedy = greedy(dag, nodes, recursive_basis, k, similarity_df)
    with Profile() as profile:
        print(greedy(dag, nodes, recursive_basis, k, similarity_df))
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats()
        )
    # Example usage:


if __name__ == '__main__':
    main()