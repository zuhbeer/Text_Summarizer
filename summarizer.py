import numpy as np
import pandas as pd 
import requests, re, spacy
nlp = spacy.load('en')
from bs4 import BeautifulSoup as BS
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from attention import AttentionLayer
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")



reviews=pd.read_csv("data/Reviews.csv",nrows=100000)

reviews.drop_duplicates(subset=['Text'],inplace=True)

reviews.dropna(axis=0,inplace=True)

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


stop = set(stopwords.words('english')) 

def clean_line(text,num):
        '''
        clean_line

        input: line (str) - a line from the script
        return: list of cleaned, lemetized tokens

        This does all the NLP preprocessing:
            * lower text
            * split on special characters, while removing annotations
            * remove stop words
            * perform lemmetization
            *** MIGHT REMOVE IMPORTANT WORDS IDK MAN
        '''
        line = text.lower()
        line = BS(line, "lxml").text
        line = re.sub(r'\([^)]*\)', '', line)
        line = re.sub('"','', line)
        line = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in line.split(" ")])    
        line = re.sub(r"'s\b","",line)
        line = re.sub("[^a-zA-Z]", " ", line) 
        line = re.sub('[m]{2,}', 'mm', line)
        if(num==0):
            tokens = [w for w in line.split() if not w in stop]
        else:
            tokens=line.split()
        long_words=[]
        for i in tokens:
            if len(i)>1:                                                
                long_words.append(i)   
        return (" ".join(long_words)).strip()

cleaned_text = []
for t in reviews['Text']:
    cleaned_text.append(clean_line(t,0))

cleaned_sum = []
for t in reviews['Summary']:
    cleaned_sum.append(clean_line(t,1))