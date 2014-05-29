import nltk

'''
@todo add pos tags
@todo add lemmas
@todo add stems
@todo add liwc classes
@todo add polarity scores
'''

def corpusFrequentWords(corpus, stopwords = []):
    '''blub
    '''
    all_words = nltk.FreqDist(w.lower() for w in corpus.words() if w not in stopwords)
    frequentWords = all_words.keys()[:50] 
    return {"frequentWords": frequentWords}
