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
---

Morteza Ansarinia
Pedro Cardoso-Leite


![h:120](../../docs/colloquium2021/assets/unilu_logo.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![h:120](../../docs/colloquium2021/assets/xcit_logo.png)


*COSA-LUCET Colloquium*
*July 14, 2021*

---
<!-- _class: lead -->

# :warning:

`
This is an ongoing study. Results are still inconclusive.
`

---
## Cognitive constructs are ambiguous

Cognitive scientists have created numerous constructs.

To name just a few, executive functions (<mark>ref</mark>), executive attention (<mark>ref</mark>), executive control (<mark>ref</mark>), attention control (<mark>ref</mark>), attentional control (<mark>ref</mark>), cognitive control (<mark>ref</mark>), fluid intelligence (<mark>ref</mark>), fluid cognition (<mark>ref</mark>), working memory (<mark>ref</mark>), updating, shifting and inhibition (<mark>ref</mark>).

:question: To what extent those terms refer to different phenomena?
:question: To what extent those terms are synonymous or polysemous?


--- 
## Example: Executive Functions (EFs)

The ability to coordinate complex behaviors in pursuit of goals.

Focus of research in many disciplines, including psychology, neuroscience, and artificial intelligence.

<center>

![h:200](../../docs/colloquium2021/assets/botvinick_triangle.png)

</center>

<!-- _footer: Image reproduced from <mark>Botvinick</mark> at <mark>Triangulating Intelligence (Oct 2020)</mark>. -->
---

## Executive Functions (EFs)

:pizza: One example of such complex behaviors would be, for instance, cooking a pizza.

![bg](#fff)

<center>

![w:650](../../docs/colloquium2021/assets/pizza.png)

</center>

---

## Models of EFs

![bg](#fff)
![bg fit 95%](../../docs/colloquium2021/assets/ef_bavelier2019.jpg)
![bg fit 95%](../../docs/colloquium2021/assets/ef_miyake2017.jpg)

<!-- _footer: Images from <mark>Bavelier2019</mark> and <mark>Miyake2017</mark>. -->

---

![bg vertical](#fff)
![bg fit 90%](../../docs/colloquium2021/assets/ef_dosenbach2007_corbetta2008.png)

<!-- _footer: Images from <mark>Dosenbach2007</mark> and <mark>Corbetta2008</mark>. -->
---

- Current state of affairs makes it hard to understand past results and limits scientific progress;

- it also makes it hard to develop effective interventions: there is a great need for more conceptual clarity.
    - Example: Targeting Attentional Control vs. Cognitive Control in cognitive training regimes with action video game.
    - EF models (Miyake) vs Attention models (Posner).

---

### To gain clarity, we can...

:feather: **Manually** read, synthesize, and criticize the literature to write reviews describing our understanding of the field.
- :thumbsdown: it's biased.
- :exploding_head: sheer volume of papers published every year (200 EFs paper per month on PubMed).

:robot: **Automatic** analysis of the scientific texts.

---
## Current project

<!--fit-->
This study aims a text-based method to gain clarity on the meaning of cognitive constructs and the measures provided by cognitive tests.
- Psychological constructs are not fully independent because the same tests may be used to characterize different constructs.
- Cognitive tests are similar to the extent that their similarly relate to constructs.

- :question: Which tests help operationalize executive function constructs?
- :question: Which constructs are useful, observable, and unambiguous?

---

## Method

<!-- _class: gaia -->

- Develop a knowledge model of what we know about EF-related constructs and tests.
- Collect articles related to the constructs and tests.
- preprocessing
- mining the corpus:
    1. descriptive statistics analysis
    2. latent space modeling
    3. topic modeling

---
## Executive Functions Ontology

A machine-readable graph-based model of what we know about *Executive Functions*.

`TODO: EFO ontology figure`

---
## Data Collection
### Pipeline

`TODO: figure for data collection and preprocessing pipeline`

---
## Data Collection
### Knowledge Model

`TODO`

- Developed an ontology by manually mining three highly cited reviews (<mark>diamond</mark>, <mark>miyake</mark>, <mark>baggetta</mark>, <mark>enkavi</mark>).

    - main classes: cognitive test, cognitive construct, brain mechanism, cognitive model
    - `TODO` some statistics of the ontology
    - other resources: CogAt, CogPo



---
## Data Collection
### PubMed Abstracts

`TODO`

- Improving the ontology via manual knowledge mining of highly cited papers
- ...

---

<!-- _class: lead -->

## Results

---
### Frequency of tasks/constructs

![bg vertical](#fff)
![bg fit right:60%](../../outputs/descriptive_plots/1_tests_corpus_size.png)
![bg fit 80%](../../outputs/descriptive_plots/1_constructs_corpus_size.png)


what is the message:
- many tasks but few are used (+power law)


---
- number of tasks per paper (x), percentage of papers (y)
- how many papers used more than one task
- co-occurrence of tasks

message:
- ???

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

message: ??? (if you show a figure, what is the message?)
 - some are specific
 - some are generic for many constructs
---
# Popular tasks and constructs

A subset of previous heat map

![bg](#fff)
![bg fit right:60%](../../outputs/matrix_factorization/1_top_test_construct_coappearance_heatmap.png)

---
<!-- _class: lead -->

`TODO: topic modeling descriptive results`


--- 
<!-- _class: lead -->
# Latent Space

- Method 1: factorize the probability matrix of co-appearance
  - goal: ??? remind them that what is the goal. Answer should be the figure we are showing afterwards.
  - show the bavelier2019 as an example of what we want to do, but driven by data.
- [SKIP] Method 2: topic modeling


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

<!-- _class: lead -->

`TODO: Topic evolution over time`

`leave aside for the presentation`

---
# Conclusion

- need for more rigorous methods to avoid confusion (e.g., ontology, constrained definition of constructs)
a theory about tasks
instead of focusing on confirmatory analysis, we can focus on tasks
a battery of tasks that covers most cognitive processes

---
## Limitations

- a measure of coherency that works for machines


limitations:
implicit decisions during data collection and processing
did not take into account that how often papers are cited
... (we are aware!)

---
## Future works
- questionnaires
- manually annotate part of the corpus
- a website for interactive visualizations
- ecologically valid tasks (models that do not involve standard lab-tasks)

---
## Reproducibility and open science

- EF ontology,
- collected PubMed corpus,
- notebooks, and codes

Currently are all the materials are available on *Uni HPC GitLab*. In the future it will be openly available on GitHub.

---
# References and Citations

<!-- class: gaia -->
`TODO`

- Page 4 image reproduced from Botvinick's talk.
- Page 6 image from Peterson2016 (reproduced in Bavelier2019) and Miyake2017
- Page 7 image from Corbetta2008


---

<!-- _paginate: false -->
<!-- _class: lead gaia -->
