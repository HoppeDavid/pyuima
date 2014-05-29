import nltk

class WordTokenize:
    '''
    '''
    def __init__(self, rawTextLabel = 'text', outputLabel = 'tokens',
                 tokenizer = nltk.tokenize.PunktWordTokenizer):
        '''
        '''
        self.rawTextLabel = rawTextLabel
        self.outputLabel = outputLabel
        self.tokenizer = tokenizer
        
    def process(self, res, **kwargs):
        tokenize = self.tokenizer()
        tokens = tokenize.tokenize(kwargs[self.rawTextLabel])
        res.update({self.outputLabel:tokens})
