
import math

import networkx as nx
import itertools
import Utils
import pandas as pd
import numpy as np
import random
from sympy.utilities.iterables import multiset_partitions
import Main
from itertools import combinations

def is_valid(cluster, similarity_df, dag):
    for pair in combinations(cluster, 2):
        if not Utils.a_valid_pair(pair[0],pair[1],dag, similarity_df, dag):
            return False
    return True

def BF(dag,nodes, recursive_basis,k, similarity_df):
   G = dag.copy()
   recursive_basis = []
   for clustering in multiset_partitions(dag.nodes, k):
       if len(clustering) < 2 or len(clustering) > k:
           continue
       valid = True
       for cluster in clustering:
            if len(cluster) < 2:
                continue
            if not is_valid(cluster, similarity_df,dag):
                valid = False
                break
       if valid:
            #print(clustering)
            g_prime = dag.copy()
            for cluster in clustering:
                if len(cluster) < 2:
                    continue
                g_prime = contract_nodes(dag, cluster, g_prime)
            if not nx.is_directed_acyclic_graph(g_prime):
                continue
            recursive_basis_g = Utils.get_recursive_basis(g_prime,nodes)
            if Utils.check_if_contain(nodes,recursive_basis_g, recursive_basis):
                G = g_prime
                recursive_basis = recursive_basis_g
                #print("found a better summary dag: ", clustering)

   return G, recursive_basis


def contract_nodes(G, cluster, g_prime):
    new_node_name = cluster[0]
    parents = set(g_prime.predecessors(cluster[0]))
    children = set(g_prime.successors(cluster[0]))
    for node in cluster[1:]:
        # g_prime = nx.contracted_nodes(g_prime, new_node_name, node, self_loops=False)
        new_node_name = new_node_name + '_' + node
        parents_n = set(g_prime.predecessors(node))
        parents.update(parents_n)
        children_n = set(g_prime.successors(node))
        children.update(children_n)
        # g_prime = nx.relabel_nodes(G, {nodes[0]: str(new_node_name), node: str(new_node_name)})
    g_prime = nx.relabel_nodes(g_prime, {cluster[0]: new_node_name})
    for node in cluster[1:]:
        g_prime.remove_node(node)
    for p in parents:
        if g_prime.has_node(p):
            g_prime.add_edge(p, new_node_name)
    for c in children:
        if g_prime.has_node(c):
            g_prime.add_edge(new_node_name, c)
    return g_prime


def main():
    dag, k, nodes, recursive_basis, similarity_df = Main.example_1()#Main.flights()
    summary_dag_opt, recursive_basis_opt = BF(dag, nodes, recursive_basis, k, similarity_df)
    Utils.show_dag(summary_dag_opt, "summary_dag_opt")

if __name__ == '__main__':
    main()