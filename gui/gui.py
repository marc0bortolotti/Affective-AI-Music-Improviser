import dash
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import rtmidi
import mido
from app.application import run_application, close_application, APPLICATION_STATUS
from OSC.osc_connection import Client_OSC, REC_MSG
import threading
import logging
import time
from PIL import Image


# avoid verbose of dash server
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# PORTS
midi_in_ports = rtmidi.MidiIn().get_ports()
midi_out_ports = mido.get_output_names()
unicorns = ['UNICORN']

# GLOBAL VARIABLES
RUN_APPLICATION = False
IMAGE_EXCITED_PATH = 'gui_images/excited.jpg'
IMAGE_RELAX_PATH = 'gui_images/relax.png'


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

dropdown_midi_in = html.Div([html.H6("MIDI INPUT:",style={'display':'inline-block', 'color': 'black', 'margin-right':36, 'margin-top':10}),
                             dcc.Dropdown(midi_in_ports, id='dropdown_midi_in', placeholder=midi_in_ports[-1], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center',  'vertical-align': 'middle'})])

dropdown_midi_out = html.Div([html.H6("MIDI OUTPUT:",style={'display':'inline-block', 'color': 'black', 'margin-right':20, 'margin-top':10}),
                              dcc.Dropdown(midi_out_ports, id='dropdown_midi_out', placeholder=midi_out_ports[2], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center', 'vertical-align': 'middle'})])

dropdown_unicorn = html.Div([html.H6("EEG DEVICE:",style={'display':'inline-block', 'color': 'black', 'margin-right':38, 'margin-top':10}),
                             dcc.Dropdown(unicorns, id='dropdown_eeg', placeholder=unicorns[-1], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center', 'vertical-align': 'middle'})])


start_button = dbc.Button("Start Application", color="success", id="button_start", disabled=False, style={'width': '250px','textAlign': 'center'})
stop_button = dbc.Button("Stop Application", color="danger", id="button_stop", disabled=False, style={'width': '250px','textAlign': 'center'})
training_button = dbc.Button("Start Training", color="primary", id="button_training", style={'width': '250px','textAlign': 'center'})


buttons = [dbc.Row([training_button], justify="center"), dbc.Row([start_button], justify="center"), dbc.Row([stop_button], justify="center")]   
dropdowns = [dbc.Row([dropdown_midi_in], justify="center"), dbc.Row([dropdown_midi_out], justify="center"), dbc.Row([dropdown_unicorn], justify="center")]

selection = dbc.Row([dbc.Col(buttons, width=4, style={'background-color': 'White'}),
                     dbc.Col(dropdowns, width=4)],
                    className="text-center",
                    justify="center",  
                    style={'color': 'gray'})

output = dbc.Row([dbc.Container(id='output')],
                 className="text-center",
                 style={'color': 'gray'})  

live_update = dcc.Interval(id='live_update',
                           interval=3*1000, # in milliseconds
                           n_intervals=0)   

app.layout = dbc.Container(fluid=True, 
                           children=[title,
                                     selection,
                                     html.Br(),
                                     output,
                                    #  live_update
                                    ])



def output(path):
    if path is not None:
        # fig_fit = go.Figure(data = data,       
        #                     layout = go.Layout(title={'text':'<b>MIDI INPUT</b>', 'xanchor': 'center', 'y':0.9, 'x':0.5, 'yanchor': 'top'}, 
        #                                         title_font_color='black',
        #                                         xaxis_range = [0, 50]))
        #                                     #    rangeslider = {'visible': True}
        # graph_fit = dcc.Graph(id="graph", figure=fig_fit)

        img = Image.open(path)
        img = html.Img(src=img, style={'height':'200px', 'width':'200px'}),
        children = dbc.Container([dbc.Row([dbc.Col(img)])])
        return children


@app.callback(Output('output', 'children'),
              Input('button_start', 'n_clicks'),
              Input('button_stop', 'n_clicks'),
              Input('dropdown_midi_in', 'value'),
              Input('dropdown_midi_out', 'value'),
              Input('dropdown_eeg', 'value'),
            #   Input('live_update', 'n_intervals')
              )
  
def update_output(start_clicks, stop_clicks, midi_in_port_name, midi_out_port_name, eeg): #, n_intervals):

    global RUN_APPLICATION, midi_in_data

    if ctx.triggered_id == 'button_start':
        print('START')

        if(midi_in_port_name == None): 
            midi_in_port_name = midi_in_ports[-1]
        if(midi_out_port_name == None): 
            midi_out_port_name = midi_out_ports[2]
        if(eeg == None): 
            eeg = unicorns[-1]

        # open midi ports
        midi_in_port = midi_in.open_port(midi_in_ports.index(midi_in_port_name))
        midi_out_port = mido.open_output(midi_out_port_name)

        # start application
        global thread_app
        thread_app = threading.Thread(target=run_application, args=(midi_in_port, midi_out_port))
        thread_app.start()

        while not application_status()['READY']:
            time.sleep(0.1)

        RUN_APPLICATION = True
   

    elif ctx.triggered_id == 'button_stop':
        print('STOP')
        close_application()
        thread_app.join()
        RUN_APPLICATION = False
        


    if RUN_APPLICATION:
        path = IMAGE_EXCITED_PATH
        return output(path)
    
    

# MAIN
if __name__ == '__main__':
    app.run()


