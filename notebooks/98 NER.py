# %% [markdown]

# This notebook demonstrates a method that extracts named entities (e.g., cognitive test names) from a corpus. It will be used during processing and more advanced NLP processing.


# %%
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# %%

# %%

cognitive_tests = ['digit span', 'stroop', 'n-back']
patterns = [nlp.make_doc(text) for text in cognitive_tests]
matcher.add('CognitiveTest', patterns)

texts = [
    'we used Stroop, n-back, and Digit span tasks to measure executive functions skill.',
    'but we only used TMT and Stroop for cognitive control measurements',
]

docs = nlp.pipe(texts)

# %%

for doc in docs:
    matches = matcher(doc)
    print(matches)


# %% user BERT to tokenize
from transformers import BertTokenizer

texts = [
    'we used Stroop, n-back, and Digit span tasks to measure executive functions skill.',
    'but we only used TMT and Stroop for cognitive control measurements',
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenizer.tokenize(texts[0])

# Conclusion: BERT-based tokenizer not really fit here in this project yet.
