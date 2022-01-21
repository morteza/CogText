# MODELS

This folder contains trained models, where they are saved as checkpoint files.

`sbert/`

  - `abstracts_all-MiniLM-L6-v2.npz`: Document embeddings of all documents using pretrained SBert `all-MiniLM-L6-v2`, a 384 dimensional embedding. Data are sorted by the the ordering as the unprocessed data provided by `PubMedDataLoader`.
  - `abstracts_UMAP5d.npz`: Mapping of the `abstracts_all-MiniLM-L6-v2` document vectors to a 5-dimensional UMAP embedding. Documents are sorted by the same ordering.
  - `abstracts_clusters.csv.gz`:
    Topic clusters of each document indexed by `pmid`. Noise documents are marked withe the `nan` cluster.
  - `abstracts_weights.npz`:
    Cluster weights of each document sorted by the the same order as in `abstracts_clusters.csv.gz`.
  - `abstracts_metapath2vec.pkl`: Serialized Word2Vec model of type `gensim.models.word2vec.Word2Vec`. Assumes Gensim v4 when deserializing the pickled file.
  - `multivariate_normal_kl_losses.csv`: A square matrix indexed by lexicon terms, where $M_{ij}$ is the KL-divergence of the respective multivariate normal distributions of the labels.

`usev4/`

  - `abstracts_USEv4.npz`: Embedding of all documents using pretrained `USEv4`, a 768 dimensional embedding. Data are sorted by the the same ordering as the unprocessed data provided by `PubMedDataLoader(preprocessed=False).load()`.


`gpt3/`

  - `abstracts_gpt3ada.npz`: GPT-3 (Ada) embedding of those abstracts with less than 2048 tokens in a 1024 dimensional space. PMIDs of each are provide in the `abstracts_gpt3ada_pmids.csv` file.
  - `abstracts_gpt3ada_pmids.csv`: PMIDs of the documents in the `abstracts_gpt3ada.npz`. Original indices of the documents (as seen in the dataset provided by `PubMedDataLoader(preprocessed=False).load()`) are also provided in the `original_index` column.
  - `abstracts_gpt3ada.nc`: A single XArray NetCDF4 dataset that includes both GPT-3 Ada embeddings of size 1024 dimensions, and PMIDs of the documents int the original PubMed abstracts dataset.

