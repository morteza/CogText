# Executive Functions Text Analysis


## Method Pipeline

> :warning: This is a rough draft figure. It's being updated.


![method pipeline](docs/pipeline.drawio.png)

### Ouptuts

- **[a] `data/ontologies/efo.owl`**: executive functions ontology (i.e. EFO), with the following IRI: `http://xcit.org/ontologies/2021/executive-functions-ontology`.
- **[b] `data/pubmed_abstracts.csv.gz`**: PubMed abstracts dataset of cognitive tasks and constructs; use gzip to decompress.

### Notebooks

**Note**: Notebooks are stored in the `notebooks/` folder.

- **`[1] Data Collection.ipynb`**: searches PubMed, aggregates abstracts as a single dataset, and stores the results in a single CSV file.
