def corpusFrequentWords(corpus, stopwords = []):
    '''
    '''
    all_words = nltk.FreqDist(w.lower() for w in corpus.words() if w not in stopwords)
    frequentWords = all_words.keys()[:50] 
    return {"frequentWords": frequentWords}
