---
title: EFO
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


![h:160](assets/unilu_logo.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![h:160](assets/xcit_logo.png)

---
<!-- _class: lead -->

<!-- This is a comment -->
# Executive Functions Text Mining
<br />

Morteza Ansarinia
Pedro Cardoso-Leite

<br />

*July 14, 2021*

---

# Background

## Executive Functions
One of the quirks of human cognition is the ability to coordinate complex behaviors in pursuit of goals. This ability has been the focus of a substantial body of research in many disciplines, including psychology, neuroscience,and artificial intelligence.

- One example of such complex tasks would be, for instance, cooking. Imagine we want to make Pizza.

---
# Problem


- Cognitive scientists have created numerous constructs like executive functions, executive attention, executive control, attention control, attentional control, cognitive control, fluid intelligence, fluid cognition, working memory, updating, shifting and inhibition, to cite just a few.

- It's not always clear  to what extent those terms refer to different phenomena and b) are distinct, synonymous or polysemous.


- This state of affairs makes it hard to understand past results and limits scientific progress; it also makes it hard to develop effective interventions: there is a great need for more conceptual clarity.


---

# Problem (2)

To gain clarity:

1. extensively reading, synthesizing and criticizing the literature and writing reviews or reports describing their understanding of the field
    - biase
    - sheer volume of papers published every year  (6 EFs paper per day on PubMed)


---
# Current project

To use text-based methods to gain clarity on the meaning of cognitive constructs and the measures provided by cognitive tests.

Why?
it is clear that psychological constructs are not fully distinct since the same cognitive test may be used to characterize different constructs.

More generally, and analogous to some recommender systems, two tasks are similar to the extent that they similarly relate to constructs, and constructs are similar to the extent that they are measured using the same tasks.

---
# Research questions


---
# Executive Functions Ontology

- operationalize -> Marr's 3rd level
- conceptualize -> constructs/algorithms
- observe -> brain and behavior


TODO: A figure of the EFO


---
# Data Collection
## Knowledge Model

- Improving the ontology via manual knowledge mining of highly cited papers
    - main classes: task, cognitive processes, brain regions
    - some statistics of the ontology
    - references: diamond, miyake, baggetta, <mark>enkavi</mark>
    - other resources: CogAt, CogPo



---
# Data Collection
## PubMed Abstracts

- Improving the ontology via manual knowledge mining of highly cited papers

---
<!-- _class: lead -->
# Descriptive Results

---
frequency of tasks/constructs

follows power law

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

---
<!-- _class: lead -->
topic modeling descriptive results


--- 
<!-- _class: lead -->
# Latent Space

- Method 1: factorize the probability matrix of co-appearance
- Method 2: topic modeling


---
# Similarity map


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
ecologically valid tasks (models that do not involve standard lab-tasks)
