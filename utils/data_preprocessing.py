import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    doc = nlp(text)
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(clean_tokens)