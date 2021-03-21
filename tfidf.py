import faiss                   # make faiss available
import engine
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import normalize
import pickle

stop = stopwords.words('english')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
wnl = WordNetLemmatizer()
RegTokenizer = RegexpTokenizer(r'\w+')
df = pd.read_csv('articles.csv')
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'publication', 'author', 'date', 'year', 'month', 'url'])
df = df.dropna()

def text2sequence(text):
    text = wnl.lemmatize(text.lower())
    
    text = RegTokenizer.tokenize(text)
    text = [item for item in text if item not in stop]
        
    sequence = tokenizer.texts_to_matrix([text], mode='tfidf')
    
    return sequence.astype(np.float32)

engine = engine.SearchEngine(text2sequence)
engine.importencoded('sequences.npy')
engine.importdf(df)
engine.normalizeencoded()
engine.buildindex()

print(engine.searchquery('queen elizabeth'))
