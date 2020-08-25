# Basic function to clean the text

def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

class yearpredictor(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return np.array([int(year) for year in X['year']]).reshape(-1,1)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

class textpredictor(TransformerMixin):

    def __init__(self, cat):
        self.cat=cat

    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X[self.cat]]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def my_preprocessor(doc):
    return(unescape(doc).lower())
def my_tokenizer(doc):
    tokens = nlp(doc)
    return([token.lemma_ for token in tokens if token.is_stop == False and token.text.isalpha() == True])


def modelprediction(tstr,df,knmodel,rmodel):

    vect=nlp(tstr).vector.reshape(1,-1)
    x,y=knmodel.kneighbors(vect,return_distance=True)
    mdf=df.copy()
    mdf=mdf.iloc[y.flatten(),]
    rmodel.fit(np.array(mdf['year']).reshape(-1,1),mdf['interest'])

    dyears=np.unique(mdf['year'])

    pred=rmodel.predict(dyears.reshape(-1,1))
    se=2*math.sqrt(mean_squared_error(rmodel.predict(np.array(mdf['year']).reshape(-1,1)),mdf.interest)/len(mdf.interest))

    return dyears,pred,mdf,se


def wordmodel(mdf):
    ts=textpredictor('description')
    wt=CountVectorizer(tokenizer=my_tokenizer,preprocessor=my_preprocessor,stop_words=STOP_WORDS,ngram_range=(1,2),min_df=.1,max_df=.8)
    yp=yearpredictor()

    desc=Pipeline([('selector',ts),('vect',wt)])
    year=Pipeline([('sel',yp)])

    feats=FeatureUnion([('desc',desc),('year',year)])

    int_est=Pipeline([('features',feats),('reg',RandomForestRegressor(max_depth=5))])
    int_est.fit(mdf,mdf['interest'])

    featslist=int_est['features'].transformer_list[0][1].named_steps['vect'].get_feature_names()
    featslist.append('year')
    feat_importances = pd.Series(int_est['reg'].feature_importances_, index=featslist)
    FI=feat_importances.drop(['year'])
    keywords=FI.nlargest(10).index.tolist()

    return keywords
