---
instruction: Compile using MARP and save the output in the `outputs/colloqium2021/` folder.
title: EFO (Uni.lu LUCET Colloquium; July 14, 2021)
marp: true
theme: gaia
# class: lead
paginate: true
_paginate: false  # skip page number of the first slide
emoji:
    - shortcode: true
    - twemoji: false
---
<!-- _class: lead -->

Clarifying Cognitive Constructs
by Automated Text Mining 
of the Literature
===

Morteza Ansarinia
Pedro Cardoso-Leite


![h:160](../../docs/colloquium2021/assets/unilu_logo.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![h:160](../../docs/colloquium2021/assets/xcit_logo.png)

---
<!-- _class: lead -->

<!-- This is a comment -->
OR

# "Executive Functions" Text Mining
<br />

*COSA-LUCET Colloquium*
*July 14, 2021*

---

## Background
### Executive Functions
- One of the quirks of human cognition is the ability to coordinate complex behaviors in pursuit of goals.
- It has been the focus of research in many disciplines, including psychology, neuroscience,and artificial intelligence.

`![FIGURE: control disciplines triangle]()`

:pizza: One example of such complex behaviors would be, for instance, cooking a pizza.

---

## Background
### Executive Functions

`![FIGURE: a complex task such as cooking]()`

---
# Problem


Cognitive scientists have created numerous constructs.

- To name just a few, executive functions (<mark>ref</mark>), executive attention (<mark>ref</mark>), executive control (<mark>ref</mark>), attention control (<mark>ref</mark>), attentional control (<mark>ref</mark>), cognitive control (<mark>ref</mark>), fluid intelligence (<mark>ref</mark>), fluid cognition (<mark>ref</mark>), working memory (<mark>ref</mark>), updating, shifting and inhibition (<mark>ref</mark>).

- To what extent those terms refer to different phenomena?
- To what extent those terms are distinct, synonymous or polysemous?

---

`![FIGURE](models of EF)`

---

- Current state of affairs makes it hard to understand past results and limits scientific progress;

- it also makes it hard to develop effective interventions: there is a great need for more conceptual clarity.
    - Example: Targeting Attentional Control vs. Cognitive Control in cognitive training regimes with action video game.

`![FIGURE](CC vs AC brain activities)`

---

To gain clarity, we can:

1. extensively reading, synthesizing and criticizing the literature and writing reviews or reports describing their understanding of the field
    - it's biased
    - sheer volume of papers published every year (6 EFs paper per day on PubMed)

2. Automatic analysis of the scientific texts

---
## Current project

We aim use text-based methods to gain clarity on the meaning of cognitive constructs and the measures provided by cognitive tests.

Why?
it is clear that psychological constructs are not fully distinct since the same cognitive test may be used to characterize different constructs.

More generally, and analogous to some recommender systems, two tasks are similar to the extent that they similarly relate to constructs, and constructs are similar to the extent that they are measured using the same tasks.

---
# Research questions

- Which tests help to operationalize cognitive constructs by cognitive tests
- Which cognitive processes are useful as constructs
- evolution over time

---
# Executive Functions Ontology

- operationalize -> Marr's 3rd level
- conceptualize -> constructs/algorithms
- observe -> brain and behavior

`TODO: A figure of the EFO`

---
## Data Collection
### Pipeline

---
## Data Collection
### Knowledge Model

- Improving the ontology via manual knowledge mining of highly cited papers
    - main classes: task, cognitive processes, brain regions
    - some statistics of the ontology
    - references: diamond, miyake, baggetta, <mark>enkavi</mark>
    - other resources: CogAt, CogPo



---
## Data Collection
### PubMed Abstracts

- Improving the ontology via manual knowledge mining of highly cited papers

---
<!-- _class: lead -->
# Descriptive Results
- Frequency of tasks/constructs

---

![bg vertical](#fff)
![bg fit](../../outputs/descriptive_plots/1_tests_corpus_size.png)
![bg fit](../../outputs/descriptive_plots/1_constructs_corpus_size.png)

---
- number of tasks per paper (x), percentage of papers (y)
- how many papers used more than one task
- co-occurrence of tasks

---
[same for the constructs]


---
evolution over time

- frequency given first appearance
- development of new tasks and constructs

---

- co-appearance of task-construct
- specificity of tasks and constructs

![bg](#fff)
![bg fit right:60%](../../outputs/matrix_factorization/1_test_construct_coappearance_heatmap.png)

---
# Popular tasks and constructs

A subset of previous heat map

![bg](#fff)
![bg fit right:60%](../../outputs/matrix_factorization/1_top_test_construct_coappearance_heatmap.png)

---
<!-- _class: lead -->
topic modeling descriptive results


--- 
<!-- _class: lead -->
# Latent Space

- Method 1: factorize the probability matrix of co-appearance
- Method 2: topic modeling


---
Cognitive tests
similarity map

![bg](#fff)
![bg fit right:60%](../../outputs/matrix_factorization/5_tests_similarity_map.png)


---
Cognitive constructs
similarity map

![bg](#fff)
![bg fit right:60%](../../outputs/matrix_factorization/6_constructs_similarity_map.png)



---
Topic evolution over time


---
# Conclusion

- need for more rigorous methods to avoid confusion (e.g., ontology, constrained definition of constructs)
a theory about tasks
instead of focusing on confirmatory analysis, we can focus on tasks
a battery of tasks that covers most cognitive processes

---
# Limitations and Future Works

- a measure of coherency that works for machines


limitations:
implicit decisions during data collection and processing
did not take into account that how often papers are cited
... (we are aware!)

## future works
questionnaires
- a website
- manually annotate part of the corpus
- ecologically valid tasks (models that do not involve standard lab-tasks)

---
# Reproducibility and Open Science

- EF ontology,
- collected PubMed corpus,
- notebooks, and codes

are all available on a private Uni.lu GitLab. Contact pedro@xcit.org for access.