import unittest
import pyuima
from pyuima import DocumentAnnotators as da
import pandas as pd
import nltk
from nltk.corpus.reader import CategorizedPlaintextCorpusReader as reader
import nltk.corpus as cor



class TestWordTokenize(unittest.TestCase):
    ''' Test functions for the basic pipeline

    '''

    def setUp(self):
        self.testText = "Dies ist ein Testsatz. Und dies ist noch einer."

    def test_WordTokenize(self): 
        # no parameters
        wt = da.WordTokenize()
        r = {}
        wt.process(r, text = self.testText)
        self.assertEqual(r['tokens'][1], 'ist')

        # different names for input and output
        wt = da.WordTokenize(rawTextLabel = 'irgendwas', 
                            outputLabel = 'super')
        r = {}
        wt.process(r, irgendwas = self.testText)
        self.assertEqual(r['super'][3], 'Testsatz.')
        
        # different tokenizer
        wt = da.WordTokenize(rawTextLabel = 'irgendwas', 
                            outputLabel = 'super',
                            tokenizer = nltk.tokenize.TreebankWordTokenizer)
        r = {}
        wt.process(r, irgendwas = self.testText)
        self.assertEqual(r['super'][3], 'Testsatz.')

if __name__ == '__main__':
    unittest.main()
