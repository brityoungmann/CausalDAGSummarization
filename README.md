# Causal DAG Summarization
Causal DAG Summarization: This code provides an implementation of our proposed CaGres algorithm. To run the code, use the example usage in the main.py file. 

The code of our algorithm is given in the file called Greedy.py

# Example usage:
//use other examples in the Example file

dag, k, nodes, recursive_basis , similarity_df = Examples.random_dag(50,0.3)

//get a summary causal DAG

summary_dag_greedy, recursive_basis_greedy = Greedy.greedy(dag, nodes, recursive_basis, k, similarity_df)

Utils.show_dag(summary_dag_greedy, 'greedy')



# Requirements
contourpy==1.2.0
cycler==0.12.1
dowhy==0.8
fonttools==4.45.1
joblib==1.3.2
kiwisolver==1.4.5
matplotlib==3.8.2
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.2
packaging==23.2
pandas==2.1.3
patsy==0.5.4
Pillow==10.1.0
pydot==1.4.2
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
scikit-learn==1.3.2
scipy==1.11.4
six==1.16.0
statsmodels==0.14.1
sympy==1.12
threadpoolctl==3.2.0
tzdata==2023.3



