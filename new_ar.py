import pandas as pd
import numpy as np
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go
from ipywidgets import widgets
import plotly.express as px

import sys

files_list = ['研究大樓','北棟病床梯','南棟電梯']

#for range and bar width
max_diff = 500
split_num = 1

df_arr = pd.DataFrame()
 

for i in files_list:
    print(i)
    data = pd.read_csv(i+'_ar.csv')
    data['building'] = i
    print(i)
    print(data.shape)
    
    df_arr = df_arr.append(data) 


df_arr.dropna(inplace=True)
ts = df_arr['timestamp'].to_numpy() 
ts = ts % 1000000
df_arr['timestamp'] = ts

df_arr.dropna(inplace=True)
ts = df_arr['timestamp'].to_numpy() 
ts = ts % 1000000
df_arr['timestamp'] = ts

is_b = []
for fl in df_arr['floor']:
    if type(fl) == type('asdf') and 'B' in fl:
        is_b.append(1)
    else:
        is_b.append(0)
df_arr['is_b'] = is_b

df_arr.sort_values(['is_b','floor','direction','timestamp'],ascending = [0,1,0,1],inplace = True)

ts1 = np.array(df_arr['timestamp'])
ts2 = np.concatenate([[ts1[0]],ts1[:-1]])
df_arr['time_diff'] = np.minimum(np.maximum(ts1-ts2,0),max_diff+2 * split_num)

print(df_arr['time_diff'])

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()
app.layout = html.Div([
    
    #Inter arrival time
    html.H1('Inter arrival time'),
    html.H3('By Floor'),
    dcc.Dropdown(
        id="building_interarr",
        options=[
            {'label': '研究大樓', 'value': '研究大樓'},
            {'label': '北棟病床梯', 'value': '北棟病床梯'},
            {'label': '南棟電梯', 'value': '南棟電梯'}
        ],
        value='研究大樓'
    ),
    dcc.Graph(id="graph_interarr"),
    

])


@app.callback(
    Output('graph_interarr', 'figure'),
    [Input('building_interarr', 'value')])
def update_figure(selected_building):
    
    fig = px.histogram(data_frame=df_arr[df_arr.building == selected_building], 
             x = "time_diff",histnorm='probability density',
             facet_col="direction",color="direction",
             animation_frame="floor",
             labels={"count":"probability",
                     "time_diff":"inter arrival time(s)"},nbins = max_diff/split_num + 2)
    #fig.update_xaxes(nticks=10)
    fig.update_xaxes(range=[0,max_diff])
    fig.update_yaxes(range=[0, 0.04])
    fig.update_layout(title_text='Inter arrival time of Elevator Calls by Time')
    return fig


app.run_server(debug=True, use_reloader=False)

