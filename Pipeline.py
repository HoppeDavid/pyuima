class FeaturePipeline():
    ''' Basic text processing pipeline

    uiaeu eudiarne uiern uiern udien u
    ie udier u
    iea 
    uiae
     uiae
      uiae u
      iae

    
    '''
    def __init__(self, corpus, data = ''):
        ''' Constructor

        @param corpus    a nltk corpus
        @param data   optional pandas DataFrame
        @retval    none
        '''

        self.__corpus = corpus
        self.__data = data
        self.__args = {}
    
    def initCorpusProcess(self, *funs, **args):
        for fun in funs:
            self.__args.update(fun(self.__corpus, **args))
        
    def initDocumentProcess(self, *funs, **args):
        self.__docFuns = funs
        self.__args.update(args)
    
    def runFeaturePipeline(self, fileids = ''):
        
        if type(self.__data) == 'pandas.core.frame.DataFrame':
            features = [self.__singleDocFeatures(fileid) for fileid in self.__data['fileids']]
        else:
            features = [self.__singleDocFeatures(fileid) for fileid in self.__corpus.fileids()]
        
        self.__features = features
        
    def getResDf(self, resCols = []):
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
            return pd.merge(resText, resData, left_on='fileids',right_on='fileids')
        else:
            return resText
        
    def getResCl(self, resCol = ''):
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
        d = nltk.defaultdict(list)
        for row in list(features):
            for k in row.keys():
                d[k].append(row[k])
        return pd.DataFrame(d)
        
    def __singleDocFeatures(self, fileid):
        features = {}
        features['fileids'] = fileid
        for fun in self.__docFuns:
            features.update(fun(fileid, self.__corpus, **self.__args))
        return features
