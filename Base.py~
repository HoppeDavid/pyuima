import nltk
import pandas as pd

'''
@todo add check function for pipeline

'''

class Pipeline():
    ''' Basic text processing pipeline

    This class implements a rudimentary UIMA pipeline. The set-up comprises
    three steps: First, a text corpus is initialized. In addition, optional
    annotations can be specified through a pandas DataFrame. Second, functions
    that make use of the whole corpus are set. Third, functions that extract
    certain features are given. Finally, the pipeline can be run. The results
    can be obtained by several functions according to the preferred format.   

    @todo  split corpus to training, test, evaluation
    @todo change 'fileids' to a flexible parameter
    '''

    def __init__(self, corpus, data = '', fileIdLabel = 'fileId',
                 contentLabel = 'text'):
        ''' Initialize the pipeline by specifying the underlying data

        @param corpus    a nltk corpus. Can be read in through one of the
        CorpusReaders

        @param data   an optional pandas DataFrame containing additional
        information about the documents in the corpus. If provided, a column
        named 'fileids' has to be included. The column must contain a mapping to
        the fileids in the corpus.
        '''
        self.__fileIdLabel = fileIdLabel
        self.__contentLabel = contentLabel
        self.__corpus = corpus
        self.__data = data
        self.__args = {}
        self.__initCas()

    def __initCas(self):
        self.__casDoc = {}
        for idIter in self.__corpus.fileids():
            d = {}
            d[self.__fileIdLabel] = idIter
            d[self.__contentLabel] = self.__corpus.raw(idIter)
            self.__casDoc[idIter] = d
            
        self.__casCorpus = {}
        self.__features = []

    def initDocumentAnnotators(self, *documentAnnotators):
        self.__docAnnotators = list(documentAnnotators)
    
    def initCorpusAnnotators(self, *corpusAnnotators):
        self.__corpusAnnotators = list(corpusAnnotators)
        
    def initFeatureExtractors(self, *featureExtractors):
        self.__FeatureExtractors = list(featureExtractors)

    def run(self):
        '''
        '''
        # Preprocessing
        for k in self.__casDoc.keys():
            [anno.process(self.__casDoc[k], **self.__casDoc[k]) 
             for anno in self.__docAnnotators]
        
        # corpus features
        tmp = nltk.defaultdict(list)
        for item in self.__casDoc.values():
            for k,v in item.items():
                tmp[k].append(v)
                
        for anno in self.__corpusAnnotators:
            anno.process(self.__casCorpus, **tmp)
            
        # document features
        for k in self.__casDoc.keys():
            f = {}
            pars = {}
            pars.update(self.__casDoc[k])
            pars.update(self.__casCorpus)
            
            for anno in self.__FeatureExtractors:
                anno.process(f, **pars) 
            self.__features.append(f)
        
        


    def getResDf(self, resCols = []):
        ''' Returns the results of the pipeline as a pandas DataFrame

        @resCols a list of column names from the additional data (default []).
        If no additional data was specified in the constructor, only the
        features are returned as a dataframe

        @retval returns the data as a pandas DataFrame '''
        resText = self.__toDataFrame(self.__features)
        
        if resCols == 'all':
            resData = self.__data
        elif not len(resCols) == 0:
            resCols.append('fileids')
            resCols = list(set(resCols))
            resData = self.__data[resCols]
        else:
            resData = False


        if resCols:
            return pd.merge(resText, resData, left_on='fileids',
                            right_on='fileids')
        else:
            return resText
        
    def getResCl(self, resCol = ''):
        ''' Returns the results of the pipeline for classification

        The format of the data returned by this functions is compatable with the
        nltk functions for classification. It is a list of tuples containing two
        entries: a dict containing the features as key value pairs and a string
        or a numeric label used for classfication or other statistical tasks.

        @param resCol name of a column, whose values should be used as labels
        for classification.

        @retval returns the data suitable for the classification functions in
        nltk. [({},_), ...].

        '''
        dd = nltk.defaultdict(str)
        tmpLabel = zip(self.__data['fileids'].tolist(), self.__data[resCol])
        for k,v in tmpLabel:
            dd[k] = v
        f = self.__features
        featTupels = []
        for featIter in range(0,len(f)):
            featTupels.append((f[featIter], dd[f[featIter]['fileids']]))
        return featTupels
    
    def __toDataFrame(self, features):
        '''


        '''
        d = nltk.defaultdict(list)
        for row in list(features):
            for k in row.keys():
                d[k].append(row[k])
        return pd.DataFrame(d)
        

