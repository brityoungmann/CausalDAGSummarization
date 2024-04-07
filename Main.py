import networkx as nx
import Utils
import Greedy
import K_SANP
import time
import BruteForce
import Examples
import Random


def main():
    dag, k, nodes, recursive_basis , similarity_df = Examples.flights()#Examples.random_dag(50,0.3)


    print(nx.is_directed_acyclic_graph(dag))
    # Utils.show_dag(dag, 'original')

    start = time.time()
    summary_dag_opt, recursive_basis_opt = BruteForce.BF(dag, nodes, recursive_basis, k, similarity_df)
    end = time.time()
    print("time opt: ", end - start)
    edges_opt = Utils.get_edges_count(summary_dag_opt, nodes, dag)
    Utils.show_dag(summary_dag_opt, "summary_dag_opt")



    start = time.time()
    summary_dag_random = Random.Random(dag, nodes, recursive_basis, k, similarity_df)
    end = time.time()
    print("time random: ", end - start)
    Utils.show_dag(summary_dag_random, "summary_dag_rand")
    edges_rand = Utils.get_edges_count(summary_dag_random, nodes, dag)
    recursive_basis_rand = Utils.get_recursive_basis(summary_dag_random, nodes)


    start = time.time()
    summary_dag_greedy, recursive_basis_greedy = Greedy.greedy(dag, nodes, recursive_basis, k, similarity_df)
    end = time.time()
    print("time greedy: ", end - start)
    #print(nx.is_directed_acyclic_graph(summary_dag_greedy))
    #Utils.show_dag(summary_dag_greedy, 'greedy')

    #recursive_basis_greedy = Utils.get_recursive_basis(summary_dag_greedy,nodes)
    edges_greedy = Utils.get_edges_count(summary_dag_greedy, nodes, dag)


    #
    # start = time.time()
    # summary_dag_k_snap = K_SANP.k_snap(dag, k, similarity_df)
    # end = time.time()
    # print("time k-snap: ", end - start)
    #Utils.show_dag(summary_dag_k_snap, 'k-snap')

    #print(recursive_basis_opt)

    #print("#############")
    #print(recursive_basis_greedy)
    #print('############')

    #recursive_basis_k_snap = Utils.get_recursive_basis(summary_dag_k_snap, nodes)
    # edges_ksnap = Utils.get_edges_count(summary_dag_k_snap, nodes, dag)
    #print(recursive_basis_k_snap)


    # print("greedy: ", edges_greedy)
    # print("k-snap: ", edges_ksnap)
    # print("opt: ", edges_opt)
    # print("random: ", edges_rand)
    # print('recursive basis opt: ')
    # print(recursive_basis_opt)
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # print('recursive basis greedy: ')
    # print(recursive_basis_greedy)
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # print('recursive basis k-snap: ')
    # print(recursive_basis_k_snap)
    # # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # # print('recursive basis rand: ')
    # # print(recursive_basis_rand)
    #
    # print('greedy -> k-snap: ',Utils.check_how_much_is_implied(nodes, recursive_basis_greedy,
    #                                                    recursive_basis_k_snap))
    # #
    # print('k-snap -> greedy: ', Utils.check_how_much_is_implied(nodes,
    #                                                    recursive_basis_k_snap,recursive_basis_greedy))
    # print('greedy -> opt: ', Utils.check_how_much_is_implied(nodes, recursive_basis_greedy,
    #                                                             recursive_basis_opt))
    #
    # print('opt -> greedy: ', Utils.check_how_much_is_implied(nodes,
    #                                                             recursive_basis_opt, recursive_basis_greedy))
    #
    # print('k-snap -> opt: ', Utils.check_how_much_is_implied(nodes, recursive_basis_k_snap,
    #                                                          recursive_basis_opt))
    #
    # print('opt -> k-snap: ', Utils.check_how_much_is_implied(nodes,
    #                                                          recursive_basis_opt, recursive_basis_k_snap))
    # #
    # print('opt-> rand: ', Utils.check_how_much_is_implied(nodes, recursive_basis_opt,
    #                                                             recursive_basis_rand))
    #
    # print('rand -> opt: ', Utils.check_how_much_is_implied(nodes,
    #                                                             recursive_basis_rand, recursive_basis_opt))
    #
    # print('opt-> rand: ', Utils.check_if_contain(nodes, recursive_basis_opt,
    #                                                       recursive_basis_rand))
    #
    # print('rand -> opt: ', Utils.check_if_contain(nodes,
    #                                                        recursive_basis_rand, recursive_basis_opt))

if __name__ == '__main__':
    main()