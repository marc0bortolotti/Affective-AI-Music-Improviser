import dash
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import rtmidi
import bluetooth
import threading
import logging
from PIL import Image
import re
import sys
import os
sys.path.append('..')
from app.application import AI_AffectiveMusicImproviser

# EEG PARAMETERS
WINDOW_OVERLAP = 0.875 # percentage
WINDOW_DURATION = 4 # seconds

STARTING_MOOD = {'RELAXED': 0, 'EXCITED': 1}
GENERATION_TYPE = {'RHYTHM': 0, 'MELODY': 1}


gen_type = 'rhythm' if GENERATION_TYPE['MELODY'] else 'melody'

# TRAINING AND VALIDATION PARAMETERS
TRAINING_SESSIONS = 1
TRAINING_TIME = 10 # must be larger than 2*WINDOW_DURATION (>8sec)
VALIDATION_TIME = 5

MODELS = {  'MT' : {'module': 'generative_model/architectures/musicTransformer.py', 'class' : 'MusicTransformer', 'param':f'generative_model/runs/MT_{gen_type}'},
            'TCN': {'module': 'generative_model/architectures/tcn.py', 'class' : 'TCN', ' param':f'generative_model/runs/TCN_{gen_type}'},}

MODEL = MODELS['MT']

PROJECT_PATH = os.path.dirname(__file__)
MODEL_PARAM_PATH = os.path.join(PROJECT_PATH,  MODEL['param'])
MODEL_MODULE_PATH = os.path.join(PROJECT_PATH, MODEL['module'])
MODEL_CLASS_NAME = MODEL['class']

start_mood = 'RELAXED' if STARTING_MOOD['RELAXED'] else 'CONCENTRATED'
simulation_track_path = os.path.join(PROJECT_PATH, f'generative_model/dataset/{gen_type}/{gen_type}_{start_mood}.mid')
ticks_per_beat = 12 if GENERATION_TYPE['RHYTHM'] else 4


# avoid verbose of dash server
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    synthetic_devices = [('00:00:00:00:00:00', 'Synthetic Board', '0000')]
    ant_neuro_devices = [('ANT_NEURO_225', 'ANT Neuro 225', '0000'), ('ANT_NEURO_411', 'ANT Neuro 411', '0000')]
    lsl_device = [('LSL', 'LSL', '0000')]
    all_devices = synthetic_devices + unicorn_devices + enophone_devices + ant_neuro_devices + lsl_device
    return all_devices


def retrieve_midi_ports():
    available_input_ports = []    
    for port in rtmidi.MidiIn().get_ports():
        available_input_ports.append(port)
    available_output_ports = []
    for port in rtmidi.MidiOut().get_ports():
        available_output_ports.append(port)
    return available_input_ports, available_output_ports


eeg_devices = retrieve_eeg_devices()
midi_in_ports, midi_out_ports = retrieve_midi_ports()

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

dropdown_midi_in = html.Div([html.H6("MIDI INPUT INSTRUMENT:",style={'display':'inline-block', 'color': 'black', 'margin-right':36, 'margin-top':10}),
                             dcc.Dropdown(midi_in_ports, id='dropdown_midi_in', placeholder=midi_in_ports[-1], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center',  'vertical-align': 'middle'})])

dropdown_midi_out_rhythm = html.Div([html.H6("MIDI OUTPUT RHYTHM:",style={'display':'inline-block', 'color': 'black', 'margin-right':20, 'margin-top':10}),
                              dcc.Dropdown(midi_out_ports, id='dropdown_midi_out', placeholder=midi_out_ports[2], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center', 'vertical-align': 'middle'})])

dropdown_midi_out_melody = html.Div([html.H6("MIDI OUTPUT MELODY:",style={'display':'inline-block', 'color': 'black', 'margin-right':20, 'margin-top':10}),
                              dcc.Dropdown(midi_out_ports, id='dropdown_midi_out', placeholder=midi_out_ports[2], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center', 'vertical-align': 'middle'})])


dropdown_unicorn = html.Div([html.H6("EEG DEVICE:",style={'display':'inline-block', 'color': 'black', 'margin-right':38, 'margin-top':10}),
                             dcc.Dropdown(eeg_devices, id='dropdown_eeg', placeholder=eeg_devices[-1], style={'display':'inline-block', 'width': '250px', 'textAlign': 'center', 'vertical-align': 'middle'})])


start_button = dbc.Button("Start Application", color="success", id="button_start", disabled=False, style={'width': '250px','textAlign': 'center'})
stop_button = dbc.Button("Stop Application", color="danger", id="button_stop", disabled=False, style={'width': '250px','textAlign': 'center'})
training_button = dbc.Button("Start Training", color="primary", id="button_training", style={'width': '250px','textAlign': 'center'})


buttons = [dbc.Row([training_button], justify="center"), dbc.Row([start_button], justify="center"), dbc.Row([stop_button], justify="center")]   
dropdowns = [dbc.Row([dropdown_midi_in], justify="center"), dbc.Row([dropdown_midi_out_rhythm], justify="center"), dbc.Row([dropdown_midi_out_rhythm], justify="center"), dbc.Row([dropdown_unicorn], justify="center")]

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
              Input('dropdown_midi_out_rhythm', 'value'),
              Input('dropdown_midi_out_melody', 'value'),
              Input('dropdown_eeg', 'value'),
            #   Input('live_update', 'n_intervals')
              )
  
def update_output(start_clicks, stop_clicks, midi_instrument_port_name, midi_rhythm_port_name, midi_melody_port_name, eeg_device): #, n_intervals):

    if ctx.triggered_id == 'button_start':
        print('START')

        if(midi_instrument_port_name == None): 
            instrument_in_port_name = midi_in_ports[-1]
        if(midi_rhythm_port_name == None): 
            midi_rhythm_port_name = midi_out_ports[2]
        if(midi_melody_port_name == None): 
            midi_melody_port_name = midi_out_ports[2]
        if(eeg_device == None): 
            eeg_device = eeg_devices[-1]

        global app
        app = AI_AffectiveMusicImproviser(  instrument_in_port_name = instrument_in_port_name, 
                                            instrument_out_port_name = midi_rhythm_port_name,
                                            generation_play_port_name = midi_melody_port_name,
                                            eeg_device_type = eeg_device,
                                            window_duration = WINDOW_DURATION,
                                            model_param_path = MODEL_PARAM_PATH,
                                            model_module_path = MODEL_MODULE_PATH,
                                            model_class_name = MODEL_CLASS_NAME,
                                            init_track_path = simulation_track_path,
                                            ticks_per_beat = ticks_per_beat,
                                            generate_melody = GENERATION_TYPE['MELODY'],
                                            parse_message=True)

        # start application
        global thread_app
        thread_app = threading.Thread(target=app.run, args=())
        thread_app.start()
   

    elif ctx.triggered_id == 'button_stop':
        print('STOP')
        app.close_application()
        thread_app.join()
    
# MAIN
if __name__ == '__main__':
    app.run()


