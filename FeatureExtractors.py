"""@file DocumentFeatures.py

The module Document Features contains a set of configurable classes that
generate features from a single documentCas structure. Each class is given a set
of parameters in the constructor.  
"""

import nltk
import numpy as np

def featMeanSentsLength(fileid, corpus, stopwords, **kwargs):
    '''This is blub
    @param fileid
    @param corpus
    @param stopwords
    @param **kwargs
    @retval none
    '''
    features = {}
    doc = corpus.sents(fileids=fileid)
    return {'MeanSentsLength':np.mean([len(s) for s in doc])}


def featMeanWordLength(fileid, corpus, stopwords, **kwargs):
    '''
    '''
    features = {}
    doc = corpus.words(fileids=fileid)
    return {'MeanWordLength': np.mean([len(w) for w in doc])}


def featNumberOfStopwords(fileid, corpus, stopwords, **kwargs):
    '''
    '''
    features = {}
    doc = corpus.words(fileids=fileid)
    return {'NumberOfStopwords': len([w for w in doc if w in stopwords])}


def featureFrequentWords(fileid, corpus, frequentWords = [], **kwargs):
    '''
    '''
    features = {}
    doc = corpus.words(fileids=fileid)
    features['length'] = len(doc)
    document_words = set(doc) 
    for word in frequentWords:
        features['contains (%s)', word] = (word in document_words)
    return features


class DocumentLength:
    ''' Computes the number of tokens of a given text'''

    def __init__(self, inputRawTextLabel = "raw", 
                 outputLabel = "FeatureTextLength"):
        ''' Constructor

        @param inputRawText = "raw"
        @param outputLabel = "FeatureTextLength"
        '''
        self.inputRawTextLabel = inputRawTextLabel
        self.outputLabel = outputLabel

    def process(self, features, **kwargs):
        '''
        '''
        features.update({self.outputLabel: len(kwargs[self.inputRawTextLabel])})
