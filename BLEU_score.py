import pandas as import pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

df = pd.read_csv('Summarized')
tru = df.true
summ = df.predicted

tru = tru.tolist()
summ = summ.tolist()


def bleu_score(tru, summ):
    '''minor cleaning and formatting 
    to feed into BLEU scorer'''
    tru = [x.strip(' ') for x in tru]
    summ = [x.strip(' ') for x in summ]
    actual = []
    summary = []
    for i in range(len(tru)):
        actual.append(tru[i].split(' '))
        summary.append(summ[i].split(' '))
    
    oneg = []
    twog = []
    threeg = []
    fourg = []
    for i in range(len(actual)):
        reference = [actual[i]]
        candidate = summary[i]
        oneg.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        twog.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        threeg.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        fourg.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
    
    one_gram = np.asarray(oneg)
    two_gram = np.asarray(twog)
    three_gram = np.asarray(threeg)
    four_gram = np.asarray(fourg)

    #It is common to report the cumulative BLEU-1 to BLEU-4 scores 
    # when describing the skill of a text generation system
    print("One Gram Score: ", np.mean(one_gram))
    print("Four Gram Score: ", np.mean(four_gram)))

bleu_score(tru, summ)

