# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import networkx as nx
import matplotlib.pyplot as plt
SEMANTIC_THRESHOLD = 0.5



def check_if_contain(nodes,recursive_basis1,recursive_basis2):
    if len(recursive_basis1) == 0:
        if len(recursive_basis2) == 0:
            return True, None
        return True, recursive_basis2[0]
    G = build_DAG_from_basis(nodes, recursive_basis1)
    #show_dag(G)

    for ci in recursive_basis2:
        ans = nx.d_separated(G,ci[0],ci[1],ci[2])
        if ans == True:
            return True, ci
    return False, None

def check_how_much_is_implied(nodes,recursive_basis1,recursive_basis2):
    count = 0
    if len(recursive_basis1) == 0:
        if len(recursive_basis2) == 0:
            return True, None
        return False, recursive_basis2[0]
    G = build_DAG_from_basis(nodes, recursive_basis1)
    #show_dag(G)

    for ci in recursive_basis2:
        ans = nx.d_separated(G,ci[0],ci[1],ci[2])
        if ans == True:
           count = count + 1
    return count/len(recursive_basis2)



def build_DAG_from_basis(nodes, recursive_basis1):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
        for n in G.nodes():
            if n == node:
                continue
            edge = True
            for ci in recursive_basis1:
                if (node in ci[0] and n in ci[1]) or (node in ci[1] and n in ci[0]):
                    edge = False
                    continue
            if edge:
                G.add_edge(n, node)
    return G

def replace_underscore(label):
    return label.replace('_', '\n')
def show_dag(dag,name):
    mapping = {node: replace_underscore(node) for node in dag.nodes}

    # Relabel nodes
    dag = nx.relabel_nodes(dag, mapping)
    pos = nx.circular_layout(dag, scale=0.5)  # Adjust the scale parameter
    nx.draw(dag, pos, with_labels=True, font_weight='bold',
            font_size=14, node_size=3000, node_color='white', edgecolors='black', arrowsize=30)  # Adjust the k parameter

    # Save and display the plot
    plt.savefig(name + '.png')
    # plt.savefig(name + '.pdf', format='pdf')
    plt.show()


def get_grounded_dag(summary_dag,nodes):
    G = summary_dag.copy()
    for n in summary_dag.nodes:
        if '_' in n:
            new_nodes = n.split('_')
            node_to_split = n

            # Identify parents and children of the original node
            parents = list(G.predecessors(node_to_split))
            children = list(G.successors(node_to_split))

            # Add edges between new nodes and parents/children
            for parent in parents:
                for new_node in new_nodes:
                    G.add_edge(parent, new_node)

            for child in children:
                for new_node in new_nodes:
                    G.add_edge(new_node, child)

            for new_node in new_nodes:
                for node_before in nodes:
                    if node_before not in n:
                        continue
                    if node_before == new_node:
                        break  # Stop connecting nodes once we reach the target node
                    G.add_edge(node_before, new_node)
            # Remove the original node
            G.remove_node(node_to_split)
    #show_dag(G,'grounded_dag')
    return G


def semantic_sim(n1, n2, similarity_df):
    if similarity_df is not None:
        sim = max(similarity_df[n1][n2], similarity_df[n2][n1])
        if sim < SEMANTIC_THRESHOLD:
            return False
    return True

def check_semantic(node1, node2, similarity_df):
    nodes1 = node1.split('_')
    nodes2 = node2.split('_')
    # print("check valid pair: ", nodes1,nodes2)
    for n1 in nodes1:
        for n2 in nodes2:
            if similarity_df is not None:
                sim = max(similarity_df[n1][n2], similarity_df[n2][n1])
                if sim < SEMANTIC_THRESHOLD:
                    return False
    return True

def a_valid_pair(node1,node2,dag, similarity_df, summary_dag):
    if not check_semantic(node1, node2,similarity_df):
        return False
    G = summary_dag.copy()
    if summary_dag.has_edge(node1,node2):

        G.remove_edge(node1, node2)
    elif summary_dag.has_edge(node2,node1):

        G.remove_edge(node2, node1)
    if nx.has_path(G, node1, node2):
            length =  nx.shortest_path_length(G, node1, node2)
            if length >= 2:
            # paths = nx.all_simple_paths(summary_dag, node1, node2)
            # path_exists = any(len(path) >= 3 for path in paths)
            # if path_exists:
                return False
    elif nx.has_path(G, node2, node1):
        length = nx.shortest_path_length(G, node2, node1)
        if length >= 2:
            # #and nx.shortest_path_length(dag, node1, node2) >= 2:
            # paths = nx.all_simple_paths(summary_dag, node2, node1)
            # path_exists = any(len(path) >= 3 for path in paths)
            # if path_exists:
                return False
    return True



    #
    #         if nx.has_path(dag, n1, n2):
    #
    #             #and nx.shortest_path_length(dag, node1, node2) >= 2:
    #             paths = nx.all_simple_paths(dag, n1, n2)
    #             path_exists = any(len(path) >= 3 for path in paths)
    #             if path_exists:
    #                 return False
    #         if nx.has_path(dag, n2, n1):
    #             #and nx.shortest_path_length(dag, node1, node2) >= 2:
    #             paths = nx.all_simple_paths(dag, n2, n1)
    #             path_exists = any(len(path) >= 3 for path in paths)
    #             if path_exists:
    #                 return False
    # if nx.has_path(summary_dag, node1, node2):
    # #and nx.shortest_path_length(dag, node1, node2) >= 2:
    #             paths = nx.all_simple_paths(summary_dag, node1, node2)
    #
    #             # Check if there is a path of length >= 2
    #             path_exists = any(len(path) >= 3 for path in paths)
    #             if path_exists:
    #                 return False
    # if nx.has_path(summary_dag, node2, node1):
    # #and nx.shortest_path_length(dag, node1, node2) >= 2:
    #             paths = nx.all_simple_paths(summary_dag, node2, node1)
    #
    #             # Check if there is a path of length >= 2
    #             path_exists = any(len(path) >= 3 for path in paths)
    #             if path_exists:
    #                 return False
    #
    # return True


def update_order(summary_dag,nodes):
    new_order = []
    dag_nodes = summary_dag.nodes()
    for node in nodes:
        if node in dag_nodes:
            new_order.append(node)
        else:
            for n in dag_nodes:
                if node in n:
                    if (node == n[-1]):
                        # cluster_nodes = n.split('_')
                        # for nn in cluster_nodes:
                        new_order.append(node)
    return new_order

def get_edges_count(summary_dag, nodes, dag, verbose = False):
    G = get_grounded_dag(summary_dag, nodes)
    if verbose:
        print("grounded dag: ", len(G.nodes), len(G.edges))
        print("original dag: ", len(dag.nodes), len(dag.edges))
        print("summary dag: ", len(summary_dag.nodes), len(summary_dag.edges))
    edges = G.edges
    num = len(edges)
    for (u,v) in edges:
        if v in dag.successors(u):
            num = num -1
    return num

def get_recursive_basis(summary_dag, nodes):
    G = get_grounded_dag(summary_dag,nodes)

    nodes = list(nx.topological_sort(G))#update_order(summary_dag,nodes)
    # show_dag(G, 'grounded_dag')
    recursive_basis = []
    for i in range(0,len(nodes)):
        if i == 0:
            continue
        n = nodes[i]
        if already_considered(n, recursive_basis):
            continue
        parents = list(G.predecessors(n))
        rest_of_the_garph = nodes[:i]
        rest_of_the_garph = [item for item in rest_of_the_garph if item not in parents]
        if len(rest_of_the_garph) == 0:
            continue
        ci = (set([n]), set(rest_of_the_garph), set(parents))
        ci = update_ci(ci, summary_dag.nodes)
        recursive_basis.append(ci)
    return recursive_basis


def already_considered(n, recursive_basis):
    for ci in recursive_basis:
        for c in ci:
            if n in c:
                return True
    return False
def update_ci(ci, nodes):
    new_ci = []
    for s in ci:
        x = set()
        for n in s:
            if n in nodes:
                x.add(n)
            else:
                for node in nodes:
                    if n in node:
                        cluster_nodes = node.split('_')
                        x.update(cluster_nodes)
        new_ci.append(x)
    return tuple(new_ci)

def main():
    # # dag1_edges = [('E1', 'AF'), ('E2', 'AF'),('E2','CE'),('E3','CE'),('E3','BD'),('E6','CE'),('E5','BD'),('E4','BD'),('E4','AF')]
    # # dag2_edges = [('E2', 'ABC'), ('E3', 'ABC'), ('E4', 'ABC'), ('E1', 'ABC'), ('E1', 'F'), ('E5', 'ABC'), ('E5', 'E'),
    # #              ('E6', 'ABC'), ('E6', 'D')]
    # # dag1 = nx.DiGraph(dag1_edges)
    # # dag2 = nx.DiGraph(dag2_edges)
    # #
    # # recursive_basis1 = [(set(['E2']), set(['E1']), set()), (set(['E3']), set(['E1', 'E2']), set()),
    # #                     (set(['E4']), set(['E1', 'E2', 'E3']), set()),
    # #                     (set(['E5']), set(['E1', 'E2', 'E3', 'E4']), set()),
    # #                     (set(['E6']), set(['E1', 'E2', 'E3', 'E4', 'E5']), set()),
    # #                     (set(['BD']), set(['E1', 'E2', 'E6']), set(['E3','E4','E5'])),
    # #                     (set(['CE']), set(['E1', 'E4', 'E5', 'E6','BD']), set(['E2','E3','E6'])),
    # #                     (set(['AF']), set(['E3', 'E5', 'E6', 'CE', 'BD']), set(['E1','E2','E4']))]
    # #
    # # recursive_basis2 =  [(set(['E2']), set(['E1']),set()), (set(['E3']), set(['E1','E2']),set()),
    # #                      (set(['E4']),set(['E1','E2','E3']),set()),
    # # (set(['E5']),set(['E1','E2','E3','E4']), set()), (set(['E6']),set(['E1','E2','E3','E4','E5']),set()),
    # #                      (set(['F']),set(['E2','E3','E4','E5','E6']), set(['E1'])),
    # # (set(['E']) ,set(['E1','E2','E3','E4','E6','F']),set(['E5'])),
    # #                      (set(['D']),set(['E1','E2','E3','E4','E5','F','E']), set(['E6'])),
    # #                      (set(['ABC']), set(['D','E','F']) ,set(['E1','E2','E3','E4','E5','E6']))]
    # #
    # # nodes = ['A','B','C','D','E','F','E1','E2','E3','E4','E5','E6']
    #
    # dag1_edges = [('A', 'BC'), ('BC', 'D'),('D','E')]
    # dag2_edges = [('A', 'C'), ('A', 'BD'), ('C', 'BD'), ('BD', 'E')]
    # dag1 = nx.DiGraph(dag1_edges)
    # dag2 = nx.DiGraph(dag2_edges)
    #
    # recursive_basis1 = [(set(['D']), set(['A']), set(['B','C'])), (set(['E']), set(['A', 'B','C']), set('D'))]
    #
    # recursive_basis2 =  [(set(['E']), set(['A','C']),set(['B','D']))]
    #
    # recursive_basis3 = [(set(['E']), set(['A','B', 'C']), set(['D']))]

    # nodes = ['A','B','C','D','E']
    #
    # G = build_DAG_from_basis(nodes, recursive_basis3)
    # show_dag(G)

    #print(check_if_contain(nodes,recursive_basis3,recursive_basis2))

    dag_edges = [('1', '2_3_6'), ('1', '5'), ('4', '2_3_6'), ('4', '5'), ('5', '2_3_6')]
    dag = nx.DiGraph(dag_edges)
    print(get_recursive_basis(dag, ['1','2','3','4','5','6']))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
