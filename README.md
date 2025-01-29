# Causal DAG Summarization

This repository provides an implementation of the **CaGres** algorithm for summarizing Causal Directed Acyclic Graphs (DAGs).

The core algorithm is implemented in the `Greedy.py` file. To run the code, follow the example usage in `Main.py` or explore the examples provided in the `Examples.py` file.

## Example Usage

```python
# Use other examples in the Example file
dag, k, nodes, recursive_basis, similarity_df = Examples.random_dag(50, 0.3)

# Get a summary causal DAG using the greedy algorithm
summary_dag_greedy, recursive_basis_greedy = Greedy.greedy(dag, nodes, recursive_basis, k, similarity_df)

# Display the summarized DAG
Utils.show_dag(summary_dag_greedy, 'greedy')
```

