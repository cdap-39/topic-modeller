from gensim import similarities
from gensim.models import ldamodel
import gensim.corpora as corpora
import pandas as pd

lda_model = ldamodel.LdaModel.load('ldamodel')
index = similarities.MatrixSimilarity.load('index')

df = pd.read_json('https://raw.githubusercontent.com/cdap-39/data/master/newsfirst_hirunews.json')

import os
os.environ.update({'MALLET_HOME': r'C:\\mallet-2.0.8\\mallet-2.0.8\\'})
mallet_path = 'C:\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet'

query = "Police have arrested seven people from the village of Ambalakulam, three kilometres away from Kilinochchi town, in connection with the leopard-killing."

id2word = corpora.Dictionary.load('id2')
vec_bow = id2word.doc2bow(query.lower().split())

vec_lda = lda_model[vec_bow]
sims = index[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[:10])

print('\n################################################################################')
print(df['data'][25])
print(df['data'][19])
print(df['data'][7])
print(df['data'][13])
print(df['data'][8])
print(df['data'][33])
print(df['data'][3])
print(df['data'][24])
print(df['data'][108])
print(df['data'][49])
