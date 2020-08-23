import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import plotly
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import plotly.express as px
import numpy as np
import time
from flask_main import flask_app
import os
from keras.models import load_model

#--------------------------------------

# link to external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,requests_pathname_prefix='/coffee_machine/',external_stylesheets=external_stylesheets)
#flask_app =app.server

#data processing
df=pd.read_sql_table('coffee_machine_test', 'sqlite:///ml_test.db')
df_half = pd.DataFrame ({'Wh':(df.groupby(pd.Grouper(freq='30T',key='time'))['kWh'].sum().ffill()*1000)})
df_half ['Hour'] = df_half.index.hour
df_half['Hour'] = df_half.Hour.map(str) + ':00 ' # format hour string
df_half ['Week'] = df_half.index.week
df_half ['Week'] = df_half.Week.map(lambda x: 'Week'+' '+ str(x))  # format week string
df_half ['Day'] = df_half.index.day_name()
df_half ['Month'] = df_half.index.strftime('%b')

# load coffee machine ann model
model = load_model('coffee_machine_ann.h5')
#df_half ['month'] = df_half['month']apply(lambda x: x )
# data process for model--------------------------------------

# manipulate data to variable

def create_dataset(dataset, look_back=96):
    dataX = []
    for q in range(len(dataset)-look_back-1):
        a = dataset[q:(q+look_back), 0]
        dataX.append(a)
    return np.array(dataX)



maxlength =48

input_dataset = create_dataset(df_half[['Wh']].values,look_back=96)
next_x_data = model.predict(input_dataset)
Yp_list0=next_x_data.tolist() # transform to list

Yp_list=[]
for w in Yp_list0:
    for n in w:
        Yp_list.append(n)

#Yp.append(next_x_data)
#yp_list = [0]*maxlength
Yp = deque(Yp_list[:maxlength],maxlen= maxlength) # prediction array
#------------------------------------

fig = px.sunburst(df_half, path=['Month','Week', 'Day', 'Hour'],values='Wh')
fig.update_layout(
    title={
        'text': "Proportion of Electricity Consumption",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig1= px.bar(data_frame=df_half,
             x='Month',
             y='Wh',
             color='Month',
             barmode='stack',
             )
fig1.update_layout(
    title={
        'text': "Monthly Electricity Consumption",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})





y_list = df_half['Wh'].values.tolist()
x_list = df_half.index.tolist()
###https://www.geeksforgeeks.org/plot-live-graphs-using-python-dash-and-plotly/

i =0
start_point = 96
a = x_list[start_point:start_point+maxlength]
X = deque(a,maxlen= maxlength)
#X.append(a)
b = y_list[start_point:start_point+maxlength]
Y = deque(b,maxlen= maxlength)
#Y.append(b)





app.layout = html.Div(
     [html.Div([      # live updated graph
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1200,
            n_intervals=maxlength
        )]),


         html.Div([dcc.Graph(id='sun-upgraph',    #sunburst graph
                             figure=fig
                             )]),

        html.Div([dcc.Graph(id='bar-upgraph',     # bar chart
                figure=fig1
    )]),

    daq.ToggleSwitch(                             # toggle
        id='my-toggle-switch',
        label='Switch to See the Instruction',
        value=False
    ),
    html.Div(id='toggle-switch-output')

    ]
)



@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)

def update_graph_scatter(n):

    global df_half
    global y_list
    global x_list
    global i
    global Yp_list
    i +=1

    X.append(x_list[start_point+maxlength+i])
    Yp.append(Yp_list[maxlength + i])



    data1 = plotly.graph_objs.Scatter(
        x=np.array(X),
        y=list(Yp),
        name='Prediction',
        mode='lines+markers'
    )

    #time.sleep(1.5)

    Y.append(y_list[start_point + maxlength+ i])

    data = plotly.graph_objs.Scatter(
        x=np.array(X),
        y=list(Y),

        name='Actual',
        mode='lines+markers'
    )


    return  {'data': [data1, data],
                'layout': go.Layout(title='Live Updated Coffee Machine Electricity Consumption and ANN Prediction Model',xaxis=dict(range=[df_half.index[start_point+i], df_half.index[start_point+maxlength+i]]), yaxis=dict(range=[(0), max(df_half['Wh'])]))
             }


@app.callback(dash.dependencies.Output('toggle-switch-output', 'children'),
    [dash.dependencies.Input('my-toggle-switch', 'value')])
def update_toggle(Value):
    if Value is True:

        return html.Div([ html.P('This chart shows the proportion'),
           html.P('of electricity consumption '),
           html.P('based on the timeframe displayed on the centre. '),
           html.P('By Clicking on the centre, the timeframe will change. ')],
                        style={'marginLeft': 5,'marginBottom':25})
    else:
        return ''


'''
@app.callback(dash.dependencies.Output('toggle-switch1-output', 'children'),
    [dash.dependencies.Input('my-toggle-switch1', 'value')])
def update_toggle1(Value1):
    if Value1 is True:

        return html.Div([ html.P('This bar chart shows'),
           html.P('monthly electricity consumption, '),
           html.P('the bar is piled up by every half-hour consumption'),
           html.P('from 1th to last day of the month.'),
           html.P('The dark the colour is, the higher the consumption is.')])
    else:
        return ''
'''
'''
if __name__ == '__main__':
    app.run_server()
'''