import nltk
import nltk.translate.gleu_score as gleu
import pandas as pd
import numpy
import os

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

def lists(filepath):
    df = pd.read_csv(filepath)

    df['predicted']=df['predicted'].fillna("")
    df['reviews']=df['reviews'].fillna("")
    df['true']=df['true'].fillna("")

    tru = df.true
    summ = df.predicted
    rev = df.reviews

    tru = tru.tolist()
    summ = summ.tolist()
    rev = rev.tolist()

    tru = [x.strip(' ') for x in tru]
    summ = [x.strip(' ') for x in summ]
    rev = [x.strip(' ') for x in rev]

    return tru, summ, rev


def G_bleu_score(tru, summ, rev):
    
    actual = []
    predicted = []
    review = []
    for i in range(len(tru)):
        actual.append(tru[i].split(' '))
        predicted.append(summ[i].split(' '))
        review.append(rev[i].split(' '))
    
    gleu_actual = []
    gleu_predicted = []
    gleu_pred_to_actual = []
    for i in range(len(actual)):
        gleu_actual.append(gleu.sentence_gleu(actual[i], ' '.join(review[i])))
        gleu_predicted.append(gleu.sentence_gleu(predicted[i], ' '.join(review[i])))
        gleu_pred_to_actual.append(gleu.sentence_gleu(predicted[i], ' '.join(actual[i])))

    ar = np.mean(gleu_actual)
    pr = np.mean(gleu_predicted)
    ap = np.mean(gleu_pred_to_actual)

    print('actual GLEU score: ',ar)
    print('predicted GLEU score: ',pr)
    print('actual to predicted GLEU score: ',ap)
    
    


