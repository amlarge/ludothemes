from flask import Flask,request, render_template
import requests
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVR
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
import numpy as np
import bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import pickle
import sys
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
import string
import spacy
#from spacy.lang.en.stop_words import STOP_WORDS
#from spacy.lang.en import English
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import base, svm
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from bokeh.models import Band, ColumnDataSource
from bokeh.models import HoverTool, Legend, LegendItem,Span, DatetimeTickFormatter, ColumnDataSource, NumeralTickFormatter, Band
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from bokeh.embed import components
from bokeh.layouts import row
from bokeh.layouts import column
from sklearn import tree
from wtforms import Form, TextAreaField, validators
from html import unescape
import enchant
from spacy.lang.en.stop_words import STOP_WORDS
from bokeh.models import Div
from sklearn.tree import DecisionTreeRegressor
app = Flask(__name__)

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

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


nlp=spacy.load('en_core_web_md')
d = enchant.Dict("en_US")
nlp.Defaults.stop_words |= {"player","players","de","de la","o","y","publishers","publisher","und","por","puntos","play","description","designer","end","board","card","game","edition","new","like","set","try","way","rule","design","feature","box"}
def my_preprocessor(doc):
    return(unescape(doc).lower())
def my_tokenizer(doc):
    tokens = nlp(doc)
    return([token.lemma_ for token in tokens if token.is_stop == False and token.text.isalpha() == True and d.check(token.text)==True])

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

def setupplot(datayears,pred,df,se):
    source = ColumnDataSource({
        'base':datayears.tolist(),
        'lower':pred-se,
        'upper':pred+se,
        })

    df['intyear']=pd.to_numeric(df.year)

    gameinfo=ColumnDataSource({
    'name':df.name,
    'interest':df.interest,
    'year':df.intyear
    })

    return source,gameinfo


def makeplot(textinput,Dy,P,games,se,color):

    pred,gameinfo=setupplot(Dy,P,games,se)

    TOOLTIPS = """
    <div>
        <div>
            <span style="font-size: 2em; font-weight: bold;">Game</span>
            <span style="font-size: 2em; color: #966;">@name</span>
        </div>
        <div>
            <span style="font-size: 2em;">Interest</span>
            <span style="font-size: 2em; color: #696;">@interest</span>
        </div>
    </div>
    """
    Plot1= figure(tools="pan,wheel_zoom,reset",plot_width=1250, plot_height=750,x_range=(1989,2021),y_range=(-3,3))
    pline1=Plot1.line(Dy.tolist(),P,line_color=color,line_width=2)
    pc1=Plot1.circle('year','interest',size=12,source=gameinfo,name='gamelist',fill_color=color,line_width=0)
    Plot1.add_tools(HoverTool(renderers=[pc1],tooltips=TOOLTIPS))
    shade1=Plot1.varea(source=pred,x='base',y1='lower',y2='upper',fill_alpha=.3,fill_color=color)
    hline1 = Plot1.line(list(range(1980,2031)), [0]*50,line_color='red', line_width=1,line_dash='dashed')

    legend1=Legend(items=[
    LegendItem(label="Raw Data",renderers=[pc1]),
    LegendItem(label="Prediction",renderers=[pline1]),
    LegendItem(label="2*SE",renderers=[shade1]),
    LegendItem(label="Average Interest",renderers=[hline1])
    ],location='bottom_left', label_text_font_size='1.6em',glyph_height=20,glyph_width=20)


    Plot1.add_layout(legend1)
    Plot1.add_layout(hline1)

    Plot1.sizing_mode = 'scale_width'
    Plot1.title.text = "Game Interest For: " + string.capwords(str(textinput))
    Plot1.title.align='center'
    Plot1.title.text_font_size='2.5em'
    Plot1.xgrid[0].grid_line_color=None
    Plot1.xaxis.axis_label = 'Year'
    Plot1.xaxis.axis_label_text_font_size='3.5em'
    Plot1.yaxis.axis_label_text_font_size='3.5em'
    Plot1.xaxis.major_label_text_font_size='1.75em'
    Plot1.yaxis.major_label_text_font_size='1.75em'
    Plot1.yaxis.axis_label = 'Interest'
    Plot1.yaxis.major_label_overrides = {0: 'Average', 3: 'High', -3: 'Low'}
    Plot1.yaxis.ticker = [ -3, 0, 3 ]

    return Plot1
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

engine = create_engine('sqlite:///games_big.db', echo=True)
FDF=pd.read_sql_query("SELECT * FROM games;",engine)

def suggestwordsdiv(keywords,root):
    out=''
    for i in range(len(keywords)): # loop through all elements of the list in the "lines" list using the index 'i'
        out += '<h5>'+str(i+1)+ '. '+'<a class="suggestions" href="/graphs/' +root +' ' +keywords[i]+'">' + keywords[i]+'</a></h5>'
    t="""Want to improve your description? Try these words: <p> """
    t+=out

    div = Div(text=t,width=300, height=500,background='white',style={'font-size': '150%', 'color': 'black'})

    return div

nn=pickle.load(open('nearestneighbors','rb'))

sf=SVR(kernel='rbf',C=50)

cols=['#af8dc3','#7fbf7b']
#
# class HelloForm(Form):
# 	sayhello = TextAreaField('')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

@app.route('/graphs',methods=['POST'])
@app.route('/graphs/<w1>',methods=['GET'])
def graphs(w1='Dinosaur'):
    # form = HelloForm(request.form)


    if request.method == 'GET':
        Dy,P,games,se=modelprediction(w1,FDF,nn,sf)
        P1=makeplot(w1,Dy,P,games,se,cols[0])

        k=wordmodel(games)
        suggestions=suggestwordsdiv(k,w1)
        metaplot=row(P1,suggestions)
        pscript,pdiv = components(metaplot)
        #P2=makeplot(w2,FDF,nn,sf,cols[1])
        #metaplot=column(P1,P2,sizing_mode='stretch_both')
        #pscript,pdiv = components(P1)
        return render_template('graphs.html',plot_script=pscript,plot_div=pdiv,w1=w1)

    else:
        word1 = request.form['theme1']
        if word1 == None:
            word1="cowboy"

        w1=word1
        #dataframe=update_figure(re.split(',',word.lower()))


        Dy,P,games,se=modelprediction(word1,FDF,nn,sf)

        P1=makeplot(word1,Dy,P,games,se,cols[0])
        #P2=makeplot(word2,FDF,nn,sf,cols[1])
        k=wordmodel(games)
        suggestions=suggestwordsdiv(k,word1)
        metaplot=row(P1,suggestions,sizing_mode='stretch_both')
        pscript,pdiv = components(metaplot)
        return render_template('graphs.html',plot_script=pscript,plot_div=pdiv,word1=word1)


#@app.route('/about')
#def about():
#  return render_template('about.html')

if __name__ == '__main__':
  app.run(port=5000, debug=True)
