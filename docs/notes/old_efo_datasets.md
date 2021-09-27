# EFO Datasets

Below is a description of corpus we collect for the EF project. All datasets, versioned with date, are available in the `data/` directory.

## PubMed Cognitive Tests Abstracts Corpus
- [ ] TODO

## PubMed Cognitive Tests Search Hits

Contains number of PubMed search hits for cognitive tests. It also includes number of hits when the task name was queried alongside the "Executive Function" or "Cognitive Control" terms.

Terms are extracted from the prior reviews.

The following table describes the columns of this dataset:

| Column  | Description |
|---      |---|
| term            | term used to represent the test in search queries |
| timestamp       | query time in ISO format |
| hits            | total number of search hits    |
| ef_hits         | number of search hits when query was combined with additional "Executive Function" or "Cognitive Control" terms) |
| cc_hits         | number of search hits when query was combined with an additional "Cognitive Control" term)  |


## PubMed Cognitive Constructs Search Hits

Stored as `efo_pubmed_concepts_hits.<date>.csv`, PubMed Concepts Hits dataset includes the number of search hits for each constructs (i.e., cognitive concepts). The terms are extracted from the EFO ontology with `rdfs:label` annotations of `efo:Construct` as the search terms.

The following table describes the columns of this dataset.

| Column  | Description |
|---      |---|
| concept | search term represented the construct name |
| timestamp_ms    | querying time in millis |
| concept_hits    | number of search hits across all contexts whether EF or not |
| concept_ef_hits | number of search hits only in EF context ( query was constrained with an additional "Executive Function" term) |

## PubMed Hits
- [ ] TODO

### PubMed Hits - Tidy
- [ ] TODO
