# Executive Functions Text Analysis


## Method Pipeline

> :warning: This is a rough draft figure. It's being updated.


![method pipeline](docs/pipeline.drawio.png)


### Notebooks

**Note**: Notebooks are stored in the [notebooks/](notebooks/) folder.

- **[NB1] [Data Collection.ipynb](notebooks/1%20Data%20Collection.ipynb)**: searches PubMed, aggregates abstracts as a single dataset, and stores the results in a single CSV file.

- **[NB2] [Preprocessing.ipynb](notebooks/)**: performs tokenizing, stripping, stop words removal, word stemming, lemmatizing, and n-gram phrase detection (e.g., working_memory will be a single token instead of two words).

- **[NB3] [NB3](notebooks/)**: TODO

- **[NB4] [NB4](notebooks/)**: TODO
- **[NB5] [NB5](notebooks/)**: TODO

### Outputs

- **[D0] [data/ontologies/efo.owl](data/ontologies/efo.owl)**: executive functions ontology (i.e. EFO), with the following IRI: `http://xcit.org/ontologies/2021/executive-functions-ontology`.
- **[D1] [data/pubmed_abstracts.csv.gz](data/pubmed_abstracts.csv.gz)**: PubMed abstracts dataset of cognitive tasks and constructs; compressed in gzip format.
- **[D1] [data/pubmed_journals.csv](data/pubmed_journals.csv)**: TODO
- **[D3] [data/pubmed_abstracts_preprocessed.csv.gz](data/pubmed_abstracts_preprocessed.csv.gz)**: TODO
- **[D3] [data/pubmed_coappearances.csv.gz](data/pubmed_coappearances.csv.gz)**: TODO
- **[D5] [...]()**: TODO
- **[D6] [...]()**: TODO
