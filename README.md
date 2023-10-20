# Linking Theories and Methods of Cognitive Control

> This is the official repository for the paper [Linking Theories and Methods in Cognitive Sciences via Joint Embedding of the Scientific Literature: The Example of Cognitive Control](https://arxiv.org/abs/2203.11016).

We performed automated text analyses on a large body of scientific texts (385705 scientific abstracts) and created a joint representation of cognitive control tasks and constructs.

Abstracts were first mapped into an embedding space using GPT-3 and Top2Vec models. Document embeddings were then used to identify a task-construct graph embedding that grounds constructs on tasks and supports nuanced meaning of the constructs by taking advantage of constrained random walks in the graph.


## Setup

We recommend [Conda/Mamba](https://mamba.readthedocs.io/en/latest/) and [DVC](https://dvc.org) to set up a clean environment and download the data. You can create and activate the `cogtext` environment and automatically download the required data from [CogText dataset on HuggingFace](https://huggingface.co/datasets/morteza/cogtext) by running:


```bash
mamba env create --file environment.yml  # or use `conda`
mamba activate cogtext                   # activate the environment
dvc pull                                 # download the data
```

## Notebooks

The main entry point of the project is the `notebooks/` folder.

Note that Jupyter notebooks contain relative paths and are supposed to be run from the root of the project.


- **[1 Data Collection (2023)](notebooks/1%20Data%20Collection%20(2023).ipynb)** uses the [EFO ontology](https://huggingface.co/datasets/morteza/cogtext/blob/main/ontologies/efo.owl) to search PubMed, aggregates abstracts as a single dataset, and stores the results in a compressed CSV file. If you already downloaded the [CogText dataset](https://huggingface.co/datasets/morteza/cogtext/blob/main/pubmed/abstracts_2023.csv.gz), you can skip this step. Simply copy your downloaded file to `data/pubmed/abstracts_2023.csv.gz`.

- **[2 Descriptive Statistics](notebooks/2%20Descriptive%20Statistics.ipynb)** computes some basic statistics such as the number of tasks and constructs, co-occurrences, articles per each task or construct, etc. This notebook requires the `data/pubmed/abstracts_2023.csv.gz` file.

- **[3 Document Embedding](notebooks/3%20Document%20Embedding.ipynb)** uses GPT-3 Embedding API (Ada) to transform the raw abstracts to vectorized embeddings.

- **[4 Topic Embedding](notebooks/4%20Topic%20Embedding.ipynb)** projects embeddings into a more interpretable topic space. The topic embedding uses UMAP and HDBSCAN to calculate the topic weights (as in Top2Vec).

- **[5 Hypernomy](notebooks/5%20Hypernomy.ipynb)** visualizes *construct hypernomy*: inconsistent definitions of cognitive constructs across cognitive fields.

- **[6 Hypergraph Visualization](notebooks/6%20Hypergraph%20Visualization.ipynb)** plots the task-construct hypergraph.

- **[7 Link Prediction](notebooks/7%20Link%20Prediction.ipynb)** predicts the edges of the task-constructs hypergraph and learns Metapath2vec embedding of the graph nodes.


# Acknowledgements

This research was supported by the Luxembourg National Research Fund (ATTRACT/2016/ID/11242114/DIGILEARN
and INTER Mobility/2017-2/ID/11765868/ULALA).

# Citation

To cite the paper use the following entry:

```
@misc{cogtext2022,
  author = {Morteza Ansarinia and
            Paul Schrater and
            Pedro Cardoso-Leite},
  title = {Linking Theories and Methods in Cognitive Sciences via Joint Embedding of the Scientific Literature: The Example of Cognitive Control},
  year = {2022},
  url = {https://arxiv.org/abs/2203.11016}
}
```
