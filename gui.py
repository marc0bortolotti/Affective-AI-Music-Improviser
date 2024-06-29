import dash
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import rtmidi
import mido
from application import run_application, close_application, get_notes_played, application_status
from OSC.osc_connection import Client_OSC, REC_MSG
import threading
import logging
import time

# avoid verbose of dash server
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)



# PORTS
midi_in = rtmidi.MidiIn()
midi_in_ports = midi_in.get_ports()
midi_out_ports = mido.get_output_names()
unicorns = ['UNICORN']



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
                             dcc.Dropdown(midi_in_ports, id='dropdown_midi_in', placeholder=midi_in_ports[-1], style={'display':'inline-block', 'width': '200px', 'textAlign': 'center',  'vertical-align': 'middle'})])

dropdown_midi_out = html.Div([html.H6("MIDI OUTPUT:",style={'display':'inline-block', 'color': 'black', 'margin-right':13, 'margin-top':10}),
                              dcc.Dropdown(midi_out_ports, id='dropdown_midi_out', placeholder=midi_out_ports[2], style={'display':'inline-block', 'width': '200px', 'textAlign': 'center', 'vertical-align': 'middle'})])

dropdown_unicorn = html.Div([html.H6("EEG DEVICE:",style={'display':'inline-block', 'color': 'black', 'margin-right':18, 'margin-top':10}),
                             dcc.Dropdown(unicorns, id='dropdown_eeg', placeholder=unicorns[-1], style={'display':'inline-block', 'width': '200px', 'textAlign': 'center', 'vertical-align': 'middle'})])


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
                           interval=50, # in milliseconds
                           n_intervals=0)   

app.layout = dbc.Container(fluid=True, 
                           children=[title,
                                     selection,
                                     html.Br(),
                                     output,
                                     live_update])



def output(data):
    if data is not None:
        fig_fit = go.Figure(data = data,       
                            layout = go.Layout(title={'text':'<b>MIDI INPUT</b>', 'xanchor': 'center', 'y':0.9, 'x':0.5, 'yanchor': 'top'}, 
                                                title_font_color='black',
                                                xaxis_range = [0, 50]))
                                            #    rangeslider = {'visible': True}

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

def update_output(start_clicks, stop_clicks, midi_in_port_name, midi_out_port_name, eeg, n_intervals):
    
    global EXIT_APPLICATION

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

        EXIT_APPLICATION = False
   

    elif ctx.triggered_id == 'button_stop':
        print('STOP')
        close_application()
        thread_app.join()
        EXIT_APPLICATION = True
        


    if not EXIT_APPLICATION:

        notes = get_notes_played()
        if len(notes) > 0:
            for note in notes:
                pitch = int(note['pitch'])
                velocity = note['velocity']
                start = note['start']
                end = note['end']
                color = f'rgb(0,0,{velocity})'

                print(start, end, color)

                data.append(go.Scatter(x=[start, end], y=[pitch, pitch], marker = {'color' : color}, showlegend=False))
            
    return output(data)
    
    

# MAIN
if __name__ == '__main__':
    global data, EXIT_APPLICATION

    EXIT_APPLICATION = True
    data = []

    app.run()


