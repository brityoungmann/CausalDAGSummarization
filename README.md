# Causal DAG Summarization

This repository provides an implementation of the **CaGres** algorithm for summarizing Causal Directed Acyclic Graphs (DAGs).

The core algorithm is implemented in the `Greedy.py` file. To run the code, follow the example usage in `Main.py` or explore the examples provided in the `Examples.py` file.

In the `Main.py` file, you'll find examples of calls for generating summary causal DAGs using the Random, Brute-Force, and K-Snap baselines.

In the  `Examples.py` file, you will find the input causal DAGs examined in our paper. 


## How It Works

The **CaGres** algorithm summarizes causal DAGs by selecting node-pairs for contraction. 

1. **Input**: A causal DAG, a bound on the number of nodes in the summary DAG k, the recursive basis of the input causal DAG (which can be generated by the function called get_recursive_basis in Utils.py), and possibly a semantic similarity matrix representing the semantic similarity between node pairs.
2. **Output**: A summarized causal DAG with no more than k nodes and its associated recursive basis.


## Algorithm Implementation

The algorithm implementation is contained within the `Greedy.py` file. Here's a brief breakdown of how the algorithm is structured:

1. **low_cost_merges**: The algorithm selects to contract node pairs in which their cost is fixed and known (e.g., node pairs with the same parents and children)

2. **fast_merege_pair**: While the number of nodes in the summary DAG is more than k, the algorithm picks the next best node-pair to contract, by counting the number of additional edges this contraction entails. The algorithm uses cach to avoid redundant cost computations. 




## Example Usage

```python
# Use other examples in the Example file
dag, k, nodes, recursive_basis, similarity_df = Examples.random_dag(50, 0.3)

# Get a summary causal DAG using the greedy algorithm
summary_dag_greedy, recursive_basis_greedy = Greedy.greedy(dag, nodes, recursive_basis, k, similarity_df)

# Display the summarized DAG
Utils.show_dag(summary_dag_greedy, 'greedy')
```







## Requirements

To run the code, make sure you have the following Python dependencies installed:

- contourpy==1.2.0
- cycler==0.12.1
- dowhy==0.8
- fonttools==4.45.1
- joblib==1.3.2
- kiwisolver==1.4.5
- matplotlib==3.8.2
- mpmath==1.3.0
- networkx==3.2.1
- numpy==1.26.2
- packaging==23.2
- pandas==2.1.3
- patsy==0.5.4
- Pillow==10.1.0
- pydot==1.4.2
- pyparsing==3.1.1
- python-dateutil==2.8.2
- pytz==2023.3.post1
- scikit-learn==1.3.2
- scipy==1.11.4
- six==1.16.0
- statsmodels==0.14.1
- sympy==1.12
- threadpoolctl==3.2.0
- tzdata==2023.3


