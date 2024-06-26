import dash
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import rtmidi
import mido
import logging
import time
import random

midi_in = rtmidi.MidiIn()
midi_in_ports = midi_in.get_ports()
midi_out_ports = mido.get_output_names()
unicorns = []

# DASHBOARD
theme = dbc.themes.BOOTSTRAP
css = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
app = dash.Dash(external_stylesheets=[theme, css])

'''for style consult Cascading Style Sheets(CSS)'''

title = dbc.Row(dbc.Col([html.Br(),
                         html.H1("Affective AI Music Improvised"),
                         html.Br(),
                         html.H5("Developed by Marco Bortolotti"),
                         html.Br()]),
                className="text-center",
                style={'color': 'Black', 'background-color': 'White'})  # 'color': 'AliceBlue', 'background-color': 'Teal'}

dropdown_midi_in = html.Div([html.H6("MIDI INPUT:",style={'display':'inline-block', 'color': 'black', 'margin-right':16, 'margin-top':10}),
                             dcc.Dropdown(midi_in_ports, id='dropdown_midi_in', style={'display':'inline-block', 'width': '200px', 'textAlign': 'center',  'vertical-align': 'middle'})])

dropdown_midi_out = html.Div([html.H6("MIDI OUTPUT:",style={'display':'inline-block', 'color': 'black', 'margin-right':13, 'margin-top':10}),
                              dcc.Dropdown(midi_out_ports, id='dropdown_midi_out', style={'display':'inline-block', 'width': '200px', 'textAlign': 'center', 'vertical-align': 'middle'})])

dropdown_unicorn = html.Div([html.H6("EEG DEVICE:",style={'display':'inline-block', 'color': 'black', 'margin-right':18, 'margin-top':10}),
                              dcc.Dropdown(unicorns, id='dropdown_eeg', style={'display':'inline-block', 'width': '200px', 'textAlign': 'center', 'vertical-align': 'middle'})])


start_button = dbc.Button("Start", color="primary", id="button_start", style={'width': '300px','textAlign': 'center'})
stop_button = dbc.Button("Stop", color="danger", id="button_stop", style={'width': '300px','textAlign': 'center'})

selection = dbc.Row([dbc.Col([html.Br(), start_button, stop_button], width=4, style={'background-color': 'White'}),
                     dbc.Col([html.Br(), dropdown_midi_in, dropdown_midi_out, dropdown_unicorn], width=3)],
                    className="text-center",
                    justify="center",  
                    style={'color': 'gray'})

output = dbc.Row([dbc.Container(id='output_img_table')],
                 className="text-center",
                 style={'color': 'gray'})  

live_update = dcc.Interval(id='live_update',
                           interval=0.5*1000, # in milliseconds
                           n_intervals=0)   

app.layout = dbc.Container(fluid=True, 
                           children=[title,
                                     selection,
                                     html.Br(),
                                     output,
                                     live_update])



def output(data):
    fig_fit = go.Figure(data=data,
                        layout = go.Layout(title={'text':'<b>Fitness</b>', 'xanchor': 'center', 'y':0.9, 'x':0.5, 'yanchor': 'top'}, 
                                            title_font_color='black')) 
    graph_fit = dcc.Graph(id="graph", figure=fig_fit)
    children = dbc.Container([dbc.Row([dbc.Col(graph_fit)])])
    return children


@app.callback(Output('output_img_table', 'children'),
              Input('button_start', 'n_clicks'),
              Input('button_stop', 'n_clicks'),
              Input('dropdown_midi_in', 'value'),
              Input('dropdown_midi_out', 'value'),
              Input('dropdown_eeg', 'value'),
              Input('live_update', 'n_intervals'))

def update_output(start, stop, midi_in, midi_out, eeg, n_intervals):
    
    global data_start, EXIT_APPLICATION

    if ctx.triggered_id == 'button_start':
        print('START')
        EXIT_APPLICATION = False
   
    elif ctx.triggered_id == 'button_stop':
        print('STOP')
        EXIT_APPLICATION = True

    if not EXIT_APPLICATION:

        value = random.randint(0, 5)
        data.append(go.Scatter(x=[data_start, data_start+3], y=[value, value], marker = {'color' : 'blue'}, showlegend=False))
        data_start+=3

    return output(data)
    
    
    

# MAIN
if __name__ == '__main__':
    global data_start, data, EXIT_APPLICATION 

    EXIT_APPLICATION = True
    data_start = 0
    data = []
    data.append(go.Scatter(x=[data_start, data_start+3], y=[4,4], marker = {'color' : 'blue'}, showlegend=False))

    app.run_server()


