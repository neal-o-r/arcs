import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Div
import arcs
import numpy as np


t = "Portrait of the Artist as a Young Man"
output_file("poa.html", title=t)

directory = 'texts/'
book = 'portrait_of_the_artist.txt' 

raw_text = arcs.de_gutenberger(directory + book)

df = arcs.sentences(raw_text)
df = arcs.sentence_sentiment(df)

arc = arcs.get_arc(df, 0)

df_sum = arcs.summary_frame(df, 20, 20)
df_sum['Sentiment'] = arc[
	np.floor((df_sum.Locs.values * (len(df)-1) / 100)).astype(int)]

desc = Div(text=open("description.html", 'r').read(), width=650)
source = ColumnDataSource(data=dict(x=[], y=[], summary=[]))

source.data = dict(
                y=df_sum['Sentiment'],
                x=df_sum['Locs'],
                summary1=df_sum.Summaries.apply(lambda x: x[0]),
                summary2=df_sum.Summaries.apply(lambda x: x[1]),
                summary3=df_sum.Summaries.apply(lambda x: x[2]),
        )
hover = HoverTool(tooltips="""
        <div>
	<p><font size="1">@summary1{safe}</font></p>
	<p><font size="1">@summary2{safe}</font></p>
	<p><font size="1">@summary3{safe}</font></p>
        </div>
""")


p = figure(plot_width=800, plot_height=650,
           title=t, toolbar_location="below", tools=[hover])

p.line(np.linspace(0, 100, len(df)), arc, line_width=2)
p.circle(df_sum.Locs, df_sum.Sentiment, size=20, color="navy", alpha=0.5,
	source=source)


p.yaxis.axis_label = "Plot"
p.xaxis.axis_label = "% of Book"

show(p)


