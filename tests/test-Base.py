import numpy as np
import random
import unittest
import pyrl.base as rl
import math
import pyuima
import pandas as pd
import nltk
from nltk.corpus.reader import CategorizedPlaintextCorpusReader as reader



class TestPipeline(unittest.TestCase):
    ''' Test functions for the basic pipeline

    '''

    def setUp(self):
        # reading in the questionnaire data
        psychData = pd.read_csv("/home/david/info/master_thesis/Data/"
                                "dataset2/analysis/dataset2.csv")
        tmp1 = [x.replace('EG_1', 'EG1').replace('EG_2', 'EG2').strip() + '.txt'
                for x in psychData['code']]
        psychData['fileids'] = tmp1

        # loading text resources
        textData = reader("/home/david/nltk_data/corpora/master_dataset2/",
                          '.*\.txt', cat_pattern = r'(.*)_.*\.txt', 
                          encoding = 'utf-8')
        stopwords = nltk.corpus.stopwords.words('german')
        testFile = textData.words(fileids=textData.fileids()[0])

    def test_mdp(self): 
       

if __name__ == '__main__':
    unittest.main()
