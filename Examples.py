import networkx as nx
import Utils
import pandas as pd
import numpy as np
import random
import pickle

from dowhy import CausalModel
import dowhy.datasets
import pydot
# Avoid printing dataconversion warnings from sklearn and numpy
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def random_dag_with_df(k):
    m = int(k/4)
    data = dowhy.datasets.linear_dataset(beta=10,
                                         num_common_causes=m,
                                         num_instruments=m,
                                         num_effect_modifiers=m,
                                         num_samples=50,
                                         treatment_is_binary=False,
                                         stddev_treatment_noise=10,
                                         num_discrete_common_causes=m)
    df = data["df"]
    dot_string = data['dot_graph']
    graph = pydot.graph_from_dot_data(dot_string)

    # Convert the pydot graph to a NetworkX DiGraph
    nx_graph = nx.DiGraph()
    for edge in graph[0].get_edges():
        source, target = edge.obj_dict['points']
        nx_graph.add_edge(source, target)

    print(len(nx_graph.nodes))
    print(len(nx_graph.edges))
    k = len(nx_graph.nodes)/2
    nodes = list(nx.topological_sort(nx_graph))
    recursive_basis = Utils.get_recursive_basis(nx_graph, nodes)
    return nx_graph, k,nodes , recursive_basis, None, df


def example_1():
    dag_edges = [('A', 'B'), ('A', 'C'), ('C', 'D'), ('B', 'D'), ('D', 'E')]
    dag = nx.DiGraph(dag_edges)
    k = 3
    nodes = ['A', 'B', 'C', 'D', 'E']
    recursive_basis = [(set(['C']), set(['B']), set(['A'])),
                       (set(['D']), set(['A']), set(['B', 'C'])),
                       (set(['E']), set(['A', 'B', 'C']), set('D'))]
    similarity_df = None

    print(list(dag.predecessors('C')) + list(dag.successors('C')))
    return dag, k, nodes, recursive_basis, similarity_df

def example_2():
    dag_edges = [('1', '2'), ('1', '3'), ('1', '5'), ('2', '6'), ('3', '6'),
                 ('4','5'),('4','6'), ('5','6')]
    dag = nx.DiGraph(dag_edges)
    k = 4
    nodes = ['1', '2', '3', '4', '5','6']
    recursive_basis = [(set(['3']), set(['2']), set(['1'])),
                       (set(['4']), set(['1','2','3']), set()),
                       (set(['5']), set(['2', '3']), set(['1','4'])),
                       (set(['6']),set(['1']), set(['2','3','4','5']))]
    return dag, k, nodes, recursive_basis, None



def generate_random_dag(nodes, density):
    # Generate a random directed graph
    graph = nx.gnp_random_graph(nodes, density, directed=True)

    # Ensure the graph is acyclic
    while not nx.is_directed_acyclic_graph(graph):
        graph = nx.gnp_random_graph(nodes, density, directed=True)

    # Remove edges to achieve the desired density
    edges_to_remove = int((1 - density) * graph.number_of_edges())
    edges = list(graph.edges())
    random.shuffle(edges)
    graph.remove_edges_from(edges[:edges_to_remove])

    nodes = list(nx.topological_sort(graph))

    recursive_basis = Utils.get_recursive_basis(graph, nodes)

    return graph, len(nodes)/2, nodes, recursive_basis, None



def random_dag(n, density, read = False):

    if read:
        #read_graph = nx.read_gpickle('random_dag.gpickle')
        with open('test.gpickle', 'rb') as f:
            dag = pickle.load(f)
    else:
        dag = nx.DiGraph()
        dag.add_node('1')  # Node 1 has no incoming edges

        for i in range(2, n + 1):
            dag.add_node(str(i))
            possible_parents = list(range(1, i))
            for parent in possible_parents:
                if random.random() < density:#.choice([True, False, False,False,False]):
                    dag.add_edge(str(parent), str(i))

        #Utils.show_dag(dag)
        #nx.write_gpickle(dag, 'random_dag.gpickle')
        with open('test.gpickle', 'wb') as f:
            pickle.dump(dag, f, pickle.HIGHEST_PROTOCOL)

    # Read the graph back from the file

    k = n/2
    nodes = list(range(1, n + 1))
    nodes = [str(n) for n in nodes]

    recursive_basis = Utils.get_recursive_basis(dag,nodes)

    return dag,k,nodes,recursive_basis, None

def flights():
    dag_edges = [('State', 'City'), ('City', 'Airport'), ('City', 'Pop'),
                 ('City', 'Temp'), ('City', 'Humidity'),('Humidity', 'Temp'),
                 ('Humidity', 'Prec'), ('Pop', 'Traffic'), ('Airport', 'Traffic'),
                 ('Airline','Fleet'), ('Fleet', 'Delay'), ('Airline', 'Delay'),
                 ('Traffic', 'Delay'), ('Temp', 'Delay'), ('Prec', 'Delay')]
    dag = nx.DiGraph(dag_edges)
    nodes = ['State', 'City', 'Airport', 'Pop', 'Temp', 'Humidity'
        , 'Prec', 'Traffic', 'Airline', 'Fleet', 'Delay']

    variables = ['State', 'City', 'Humidity', 'Airport', 'Pop',
                'Temp', 'Traffic', 'Airline', 'Prec', 'Fleet', 'Delay']

    # # Create an example 10x10 similarity matrix
    similarity_matrix = np.array([
        [1.0, 0.9, 0.4, 0.5, 0.7, 0.6, 0.5, 0.7, 0.5, 0.3,0.2],
        [0, 1.0, 0.5, 0.5, 0.6, 0.7, 0.6, 0.7, 0.4, 0.5, 0.3],
        [0,0 , 1.0, 0.3, 0.5, 0.9, 0.4, 0.5, 0.8, 0.2, 0.1],
        [0, 0, 0, 1.0, 0.5, 0.4, 0.8, 0.8, 0.2, 0.8, 0.8],
        [0, 0, 0, 0, 1.0, 0.4, 0.8, 0.6, 0.3, 0.4, 0.5],
        [0, 0, 0, 0, 0, 1.0, 0.3, 0.2, 0.9, 0.3, 0.5],
        [0, 0, 0, 0, 0, 0, 1.0, 0.8, 0.5, 0.4, 0.8],
        [0, 0, 0, 0, 0, 0, 0, 1.0, 0.4, 0.8, 0.9],
        [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.2, 0.4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    ])
    #

    # # Create a DataFrame with variable names and the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=variables, columns=variables)
    # print(similarity_df['Traffic']['Prec'], similarity_df['Prec']['Traffic'])
    # print(similarity_df)
    recursive_basis = Utils.get_recursive_basis(dag, nodes)
    return dag, 7, nodes, recursive_basis, similarity_df


def covid():
    dag_edges = [('climate', 'COVID19RecoveredCases'), ('climate', 'COVID19Spread'),
                 ('climate', 'newCOVID19Cases'),
                 ('country', 'climate'), ('country', 'COVID19RecoveredCases'),
                 ('country', 'COVID19Spread'),
                 ('country', 'economy'), ('country', 'newCOVID19Cases'),
                 ('country', 'officialLanguage'),
                 ('country','populationSize'), ('country', 'timeZone'),
                 ('COVID19Spread', 'COVID19RecoveredCases'),
                 ('COVID19Spread', 'newCOVID19Cases'), ('economy', 'COVID19RecoveredCases'),
                 ('ethnicGroups', 'country'),('ethnicGroups','officialLanguage'),
                 ('geography','climate'), ('geography', 'country'),
                 ('geography', 'ethnicGroups'), ('geography','timeZone'),
                 ('newCOVID19Cases', 'COVID19RecoveredCases'),
                 ('populationSize','COVID19Spread'), ('populationSize','newCOVID19Cases')]


    dag = nx.DiGraph(dag_edges)
    print(len(dag.nodes), len(dag.edges))

    print(nx.is_directed_acyclic_graph(dag))
    print(dag.nodes)
    print(list(nx.topological_sort(dag)))

    # t = nx.has_path(dag, 'geography', 'timeZone')
    # y = nx.has_path(dag, 'timeZone','geography')
    nodes = ['geography', 'ethnicGroups', 'country', 'climate', 'economy',
             'officialLanguage', 'populationSize', 'timeZone', 'COVID19Spread',
             'newCOVID19Cases', 'COVID19RecoveredCases']

    variables = nodes

    # # Create an example 10x10 similarity matrix
    similarity_matrix = np.array([
        [1.0,0.4,0.8,0.8,0.7,0.5,0.6,0.9,0.2,0.2,0.2],
        [0, 1.0, 0.8, 0.4, 0.4,
             0.8, 0.6, 0.5, 0.4,
             0.3,0.3],
        [0,0 , 1.0, 0.8, 0.6,
             0.8, 0.7, 0.8, 0.2,
             0.2, 0.3],
        [0, 0, 0, 1.0, 0.5,
             0.3, 0.5, 0.8, 0.4,
             0.3, 0.4],
        [0, 0, 0, 0, 1.0, 0.7, 0.7, 0.6, 0.4,0.3,0.2],
        [0, 0, 0, 0, 0, 1.0, 0.6, 0.8, 0.4,0.4,0.4],
        [0, 0, 0, 0, 0, 0, 1.0, 0.5, 0.4,0.3,0.2 ],
        [0, 0, 0, 0, 0, 0, 0, 1.0, 0.4, 0.4, 0.4],
        [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.8, 0.8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    ])
    #

    # # Create a DataFrame with variable names and the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=variables, columns=variables)
    # print(similarity_df['Traffic']['Prec'], similarity_df['Prec']['Traffic'])
    # print(similarity_df)
    recursive_basis = Utils.get_recursive_basis(dag, nodes)
    return dag, 6, nodes, recursive_basis, similarity_df

def adult():
    data = [
        ('age', 'education_num'),
        ('age', 'hours_per_week'),
        ('age', 'income'),
        ('age', 'income'),
        ('age', 'marital_status'),
        ('age', 'marital_status'),
        ('age', 'occupation'),
        ('age', 'relationship'),
        ('age', 'work_class'),
        ('education_num', 'education'),
        ('education_num', 'hours_per_week'),
        ('education_num', 'income'),
        ('education_num', 'income'),
        ('education_num', 'occupation'),
        ('education_num', 'work_class'),
        ('education_num', 'work_class'),
        ('hours_per_week', 'income'),
        ('hours_per_week', 'relationship'),
        ('income', 'education'),
        ('marital_status', 'education_num'),
        ('marital_status', 'hours_per_week'),
        ('marital_status', 'income'),
        ('marital_status', 'occupation'),
        ('marital_status', 'work_class'),
        ('marital_status', 'work_class'),
        ('native_country', 'education'),
        ('native_country', 'education_num'),
        ('native_country', 'education_num'),
        ('native_country', 'hours_per_week'),
        ('native_country', 'income'),
        ('native_country', 'marital_status'),
        ('native_country', 'occupation'),
        ('native_country', 'race'),
        ('native_country', 'work_class'),
        ('occupation', 'education'),
        ('occupation', 'income'),
        ('occupation', 'occupation_category'),
        ('occupation_category', 'education'),
        ('occupation_category', 'work_class'),
        ('race', 'education_num'),
        ('race', 'hours_per_week'),
        ('race', 'income'),
        ('race', 'income'),
        ('race', 'marital_status'),
        ('race', 'occupation'),
        ('race', 'work_class'),
        ('race', 'work_class'),
        ('sex', 'education_num'),
        ('sex', 'hours_per_week'),
        ('sex', 'income'),
        ('sex', 'income'),
        ('sex', 'marital_status'),
        ('sex', 'occupation'),
        ('sex', 'race'),
        ('sex', 'relationship'),
        ('sex', 'work_class'),
        ('sex', 'work_class'),
        ('work_class', 'income'),
    ]

    # Remove '_' from all strings in the list of tuples
    data_no_underscore = [(row[0].replace('_', ''), row[1].replace('_', '')) for row in data]
    dag = nx.DiGraph(data_no_underscore)
    print(len(dag.nodes), len(dag.edges))

    print(nx.is_directed_acyclic_graph(dag))
    print(len(dag.nodes), len(dag.edges))
    print(list(nx.topological_sort(dag)))
    variables = ['age', 'nativecountry', 'sex', 'race', 'maritalstatus', 'educationnum',
                 'hoursperweek', 'occupation', 'relationship', 'occupationcategory',
                 'workclass', 'income', 'education']

    # Create a DataFrame with NaN values
    similarity_matrix = pd.DataFrame(0, index=variables, columns=variables)

    # Subjective assessment of semantic similarity
    similarity_matrix.loc['age', 'nativecountry'] = 0.2
    similarity_matrix.loc['age', 'sex'] = 0.8
    similarity_matrix.loc['age', 'race'] = 0.8
    similarity_matrix.loc['age', 'maritalstatus'] = 0.8
    similarity_matrix.loc['age', 'educationnum'] = 0.7
    similarity_matrix.loc['age', 'hoursperweek'] = 0.5
    similarity_matrix.loc['age', 'occupation'] = 0.5
    similarity_matrix.loc['age', 'relationship'] = 0.7
    similarity_matrix.loc['age', 'occupationcategory'] = 0.6
    similarity_matrix.loc['age', 'workclass'] = 0.4
    similarity_matrix.loc['age', 'income'] = 0.6
    similarity_matrix.loc['age', 'education'] = 0.6

    similarity_matrix.loc['nativecountry', 'sex'] = 0.4
    similarity_matrix.loc['nativecountry', 'race'] = 0.8
    similarity_matrix.loc['nativecountry', 'maritalstatus'] = 0.2
    similarity_matrix.loc['nativecountry', 'educationnum'] = 0.1
    similarity_matrix.loc['nativecountry', 'hours_perweek'] = 0.1
    similarity_matrix.loc['nativecountry', 'occupation'] = 0.1
    similarity_matrix.loc['nativecountry', 'relationship'] = 0.1
    similarity_matrix.loc['nativecountry', 'occupationcategory'] = 0.1
    similarity_matrix.loc['nativecountry', 'workclass'] = 0.1
    similarity_matrix.loc['nativecountry', 'income'] = 0.4
    similarity_matrix.loc['nativecountry', 'education'] = 0.1

    similarity_matrix.loc['sex', 'race'] = 0.8
    similarity_matrix.loc['sex', 'maritalstatus'] = 0.5
    similarity_matrix.loc['sex', 'educationnum'] = 0.1
    similarity_matrix.loc['sex', 'hoursperweek'] = 0.1
    similarity_matrix.loc['sex', 'occupation'] = 0.5
    similarity_matrix.loc['sex', 'relationship'] = 0.8
    similarity_matrix.loc['sex', 'occupationcategory'] = 0.5
    similarity_matrix.loc['sex', 'workclass'] = 0.1
    similarity_matrix.loc['sex', 'income'] = 0.4
    similarity_matrix.loc['sex', 'education'] = 0.1

    similarity_matrix.loc['race', 'maritalstatus'] = 0.5
    similarity_matrix.loc['race', 'educationnum'] = 0.1
    similarity_matrix.loc['race', 'hoursperweek'] = 0.1
    similarity_matrix.loc['race', 'occupation'] = 0.5
    similarity_matrix.loc['race', 'relationship'] = 0.7
    similarity_matrix.loc['race', 'occupationcategory'] = 0.5
    similarity_matrix.loc['race', 'workclass'] = 0.1
    similarity_matrix.loc['race', 'income'] = 0.4
    similarity_matrix.loc['race', 'education'] = 0.6

    similarity_matrix.loc['maritalstatus', 'educationnum'] = 0.1
    similarity_matrix.loc['maritalstatus', 'hoursperweek'] = 0.4
    similarity_matrix.loc['maritalstatus', 'occupation'] = 0.5
    similarity_matrix.loc['maritalstatus', 'relationship'] = 0.8
    similarity_matrix.loc['maritalstatus', 'occupationcategory'] = 0.5
    similarity_matrix.loc['maritalstatus', 'workclass'] = 0.1
    similarity_matrix.loc['maritalstatus', 'income'] = 0.4
    similarity_matrix.loc['maritalstatus', 'education'] = 0.3

    similarity_matrix.loc['educationnum', 'hoursperweek'] = 0.8
    similarity_matrix.loc['educationnum', 'occupation'] = 0.8
    similarity_matrix.loc['educationnum', 'relationship'] = 0.3
    similarity_matrix.loc['educationnum', 'occupationcategory'] = 0.8
    similarity_matrix.loc['educationnum', 'workclass'] = 0.8
    similarity_matrix.loc['educationnum', 'income'] = 0.8
    similarity_matrix.loc['educationnum', 'education'] = 0.9

    similarity_matrix.loc['hoursperweek', 'occupation'] = 0.4
    similarity_matrix.loc['hoursperweek', 'relationship'] = 0.3
    similarity_matrix.loc['hoursperweek', 'occupationcategory'] = 0.4
    similarity_matrix.loc['hoursperweek', 'workclass'] = 0.3
    similarity_matrix.loc['hoursperweek', 'income'] = 0.8
    similarity_matrix.loc['hoursperweek', 'education'] = 0.3

    similarity_matrix.loc['occupation', 'relationship'] = 0.3
    similarity_matrix.loc['occupation', 'occupationcategory'] = 0.9
    similarity_matrix.loc['occupation', 'workclass'] = 0.8
    similarity_matrix.loc['occupation', 'income'] = 0.8
    similarity_matrix.loc['occupation', 'education'] = 0.9

    similarity_matrix.loc['relationship', 'occupationcategory'] = 0.6
    similarity_matrix.loc['relationship', 'workclass'] = 0.3
    similarity_matrix.loc['relationship', 'income'] = 0.3
    similarity_matrix.loc['relationship', 'education'] = 0.3

    similarity_matrix.loc['occupationcategory', 'workclass'] = 0.9
    similarity_matrix.loc['occupationcategory', 'income'] = 0.7
    similarity_matrix.loc['occupationcategory', 'education'] = 0.7

    similarity_matrix.loc['workclass', 'income'] = 0.7
    similarity_matrix.loc['workclass', 'education'] = 0.7

    similarity_matrix.loc['income', 'education'] = 0.8

    recursive_basis = Utils.get_recursive_basis(dag, variables)
    return dag, 7, variables, recursive_basis, similarity_matrix


def german():
    data = [
        ('age', 'amount'),
        ('age', 'credit_history'),
        ('age', 'credit_risk'),
        ('age', 'duration'),
        ('age', 'housing'),
        ('age', 'other_debtors'),
        ('age', 'savings'),
        ('age', 'status'),
        ('amount', 'credit_risk'),
        ('amount', 'status'),
        ('credit_history', 'credit_risk'),
        ('credit_history', 'duration'),
        ('duration', 'credit_risk'),
        ('foreign_worker', 'installment_rate'),
        ('housing', 'credit_history'),
        ('housing', 'credit_risk'),
        ('housing', 'telephone'),
        ('installment_rate', 'credit_history'),
        ('job', 'amount'),
        ('number_credits', 'credit_history'),
        ('number_credits', 'installment_rate'),
        ('number_credits', 'status'),
        ('other_debtors', 'installment_rate'),
        ('other_installment_plans', 'housing'),
        ('other_installment_plans', 'property'),
        ('people_liable', 'amount'),
        ('people_liable', 'credit_history'),
        ('personal_status_sex', 'amount'),
        ('personal_status_sex', 'credit_history'),
        ('personal_status_sex', 'credit_risk'),
        ('personal_status_sex', 'duration'),
        ('personal_status_sex', 'housing'),
        ('personal_status_sex', 'savings'),
        ('personal_status_sex', 'status'),
        ('present_residence', 'credit_risk'),
        ('present_residence', 'employment_duration'),
        ('present_residence', 'property'),
        ('property', 'status'),
        ('purpose', 'personal_status_sex'),
        ('savings', 'credit_risk'),
        ('status', 'credit_risk'),
        ('telephone', 'other_debtors'),
        ('telephone', 'people_liable')
    ]

    # Remove '_' from all strings in the list of tuples
    data_no_underscore = [(row[0].replace('_', ''), row[1].replace('_', '')) for row in data]
    dag = nx.DiGraph(data_no_underscore)
    print(len(dag.nodes), len(dag.edges))
    print(nx.is_directed_acyclic_graph(dag))
    print(len(dag.nodes), len(dag.edges))
    print(list(nx.topological_sort(dag)))
    variables = ['age', 'foreignworker', 'job', 'numbercredits', 'otherinstallmentplans',
                 'presentresidence', 'purpose', 'employmentduration', 'property',
                 'personalstatussex', 'housing', 'savings', 'telephone', 'otherdebtors',
                 'peopleliable', 'installmentrate', 'amount', 'credithistory', 'status',
                 'duration', 'creditrisk']


    similarity_df = pd.read_csv('german_sematic.csv', index_col=0)
    # similarity_df.set_index(variables, inplace=True)
    similarity_df = similarity_df.fillna(0)
    print(similarity_df.head())
    recursive_basis = Utils.get_recursive_basis(dag, variables)
    return dag, 11, variables, recursive_basis, similarity_df

def accidents():
    file_path = 'traffic.edgelist'  # Replace with the actual path to your file

    # Read non-empty lines from the file
    with open(file_path, 'r') as file:
        rows = [line.strip().split() for line in file if line.strip()]

    #print(rows)
    # Create a list of tuples
    data = [(row[0], row[1]) for row in rows]

    data_no_underscore = [(row[0].replace('_', ''), row[1].replace('_', '')) for row in data]
    dag = nx.DiGraph(data_no_underscore)

    print(nx.is_directed_acyclic_graph(dag))
    print(len(dag.nodes), len(dag.edges))
    print(list(nx.topological_sort(dag)))

    variables = ['country', 'starttime', 'timezone', 'state', 'endtime',
                 'sunrisesunset', 'county', 'civiltwilight', 'city',
                 'nauticaltwilight', 'zipcode', 'astronomicaltwilight',
                 'street', 'startlatitude', 'startlongitude', 'endlatitude',
                 'endlongitude', 'temperaturefahrenheit', 'pressureinches',
                 'visibilitymiles', 'winddirection', 'precipitationinches',
                 'amenity', 'bump', 'giveway', 'noexit', 'railway', 'roundabout',
                 'station', 'stop', 'windchillfahrenheit', 'humiditypercentage',
                 'windspeedmilesperhour', 'trafficcalming', 'junction', 'crossing',
                 'weathercondition', 'trafficsignal', 'turningloop', 'distancemiles', 'severity']


    similarity_df = pd.read_csv('traffic_sematic.csv', index_col=0)
    # similarity_df.set_index(variables, inplace=True)
    similarity_df = similarity_df.fillna(0)
    #print(similarity_df.head())
    recursive_basis = Utils.get_recursive_basis(dag, variables)
    return dag, 21, variables, recursive_basis, similarity_df





if __name__ == '__main__':
    #covid()
    #adult()
    #german()
    #accidents()
    #example_1()
    random_dag_with_df()