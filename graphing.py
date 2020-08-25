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
    ],location='bottom_left', label_text_font_size='1em',glyph_height=10,glyph_width=10)


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

def suggestwordsdiv(keywords,root):
    out=''
    for i in range(len(keywords)): # loop through all elements of the list in the "lines" list using the index 'i'
        out += '<h5>'+str(i+1)+ '. '+'<a class="suggestions" href="/graphs/' +root +' ' +keywords[i]+'">' + keywords[i]+'</a></h5>'
    t="""Want to improve your description? Try these words: <p> """
    t+=out

    div = Div(text=t,width=300, height=500,background='white',style={'font-size': '150%', 'color': 'black'})

    return div
