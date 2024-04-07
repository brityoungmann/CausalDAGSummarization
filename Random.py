
import math

import networkx as nx
import itertools
import Utils
import pandas as pd
import numpy as np
import random
from sympy.utilities.iterables import multiset_partitions
import Main
import BruteForce
from itertools import combinations



def Random(dag,nodes, recursive_basis,k, similarity_df):
   G = dag.copy()
   for clustering in multiset_partitions(dag.nodes, k):
       if len(clustering) < 2 or len(clustering) > k:
           continue
       valid = True
       for cluster in clustering:
            if len(cluster) < 2:
                continue
            if not BruteForce.is_valid(cluster, similarity_df,dag):
                valid = False
                break
       if valid:
            if random.choice([True, False]):
                continue
            g_prime = dag.copy()
            for cluster in clustering:
                if len(cluster) < 2:
                    continue
                g_prime = BruteForce.contract_nodes(dag, cluster, g_prime)
            if not nx.is_directed_acyclic_graph(g_prime):
                continue
            if random.choice([True,False]):
                G = g_prime
   return G





def main():
    dag, k, nodes, recursive_basis, similarity_df = Main.example_1()#Main.flights()
    summary_dag_random = Random(dag, nodes, recursive_basis, k, similarity_df)
    Utils.show_dag(summary_dag_random, "summary_dag_random")

if __name__ == '__main__':
    main()