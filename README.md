# Linking Theories and Methods of Cognitive Control

> This is the official repository for the paper [Linking Theories and Methods in Cognitive Sciences via Joint Embedding of the Scientific Literature: The Example of Cognitive Control](https://arxiv.org/abs/2203.11016).

We performed automated text analyses on a large body of scientific texts (385705 scientific abstracts) and created a joint representation of cognitive control tasks and constructs.

Abstracts were first mapped into an embedding space using GPT-3 and Top2Vec models. Document embeddings were then used to identify a task-construct graph embedding that grounds constructs on tasks and supports nuanced meaning of the constructs by taking advantage of constrained random walks in the graph.


## Setup

We recommend conda/mamba to set up a clean environment for this project. All the required packages are listed in the `environment.yml` file. You can create and activate the `cogtext` environment by running:

```bash
conda create --file environment.yml  # or use mamba/micromamba
conda activate cogtext
```

## Data

The required dataset for the analysis can be downloaded from the [CogText dataset on HuggingFace](https://huggingface.co/datasets/morteza/cogtext). The dataset contains a collection of PubMed abstracts, along with their GPT-3 embeddings and topic embeddings.

One convenient place to keep the dataset is the `../cogtext_data/` folder.

## Notebooks

The main entry point of the project is the `notebooks/` folder.

Note that Jupyter notebooks contain relative paths and are supposed to be run from the root of the project.


- **<kbd>1</kbd> [Data Collection](notebooks/1%20Data%20Collection.ipynb)**: searches PubMed, aggregates abstracts as a single dataset, and stores the results in a single CSV file. If you already downladed the CogText dataset, you can skip this step.

- **<kbd>2</kbd> [Descriptive Statistics](notebooks/2%20Descriptive%20Statistics.ipynb)** computes some basic statistics such as number of tasks per article, co-occurrences, articles per each task or construct, etc. The notebook does not need the texts and relies only on the PMIDs of the articles.

- **<kbd>3</kbd> [Document Embedding](notebooks/3%20Document%20Embedding.ipynb)** uses GPT-3 Embedding API (Ada) to embed the raw abstracts.

- **<kbd>4</kbd> [Topic Embedding](notebooks/4%20Topic%20Embedding.ipynb)** projects document embeddings into a more interpretable topic space. The topic embedding uses UMAP and HDBSCAN to calculate the topic weights (as in Top2Vec).

- **<kbd>5</kbd> [Hypernomy](notebooks/5%20Hypernomy.ipynb)** visualizes construct hypernomy, inconsistent definitions of constructs across cognitive  fields.

- **<kbd>6</kbd> [Hypergraph Visualization](notebooks/6%20Hypergraph%20Visualization.ipynb)** plots the task-construct hypergraph.

- **<kbd>7</kbd> [Link Prediction](notebooks/7%20Link%20Prediction.ipynb)** predicts the graph edges of the task-constructs and learns Metapath2vec embedding of the graph nodes.


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
