import pickle
import spacy
import en_core_web_sm
import pandas as pd

with open("dataset/doc-with-authid.p",'rb') as f:
    docs = pickle.load(f)

print([token.has_vector for token in docs[1]])
