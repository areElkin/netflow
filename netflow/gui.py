# import tkinter as tk
# import tkinter.ttk as ttk

# window = tk.Tk()
# frame = tk.Frame()

# greeting = ttk.Label(master=frame, text="Hello, Tkinter", width=30, foreground="green")  # Set the text color to white
# greeting.pack()
# button = tk.Button(master=frame, text="Click me!", width=20, heigh=2, fg="green")
# # button.pack()

# entry = tk.Entry(master=frame, fg="yellow", bg="blue", width=50)
# # entry.pack()

# label = ttk.Label(text="Name", master=frame)
# entry = ttk.Entry()

# # label.pack()
# # entry.pack()

# name = entry.get()
# entry.delete(0, 4)

# text_box = tk.Text(master=frame)
# # text_box.pack()
# text_box_input = text_box.get("1.0", tk.END)

# frame.pack()
# window.mainloop()

import networkx as nx
import tkinter as tk
import tkinter.ttk as ttk
import threading
import webbrowser
import random

import plotly.graph_objs as go
# import dash_html_components as html

# from dash import Dash, dcc
# from dash.dependencies import Output, Input

from dash import Dash, dcc, html, Input, Output, callback
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

class DashThread(threading.Thread):
    """ CREATING A DASH APPLICATION THREAD - We’ll be running the Dash application in a separate thread.
    This allows the Tkinter GUI and Dash app to run simultaneously. To do this, we create a new DashThread
    class that inherits from Python’s threading.Thread
    """
    def __init__(self, data_list, G):
        threading.Thread.__init__(self)
        self.data_list = data_list
        self.G = G

        self.app = Dash(__name__,
                        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

        # Initialize an empty graph
        pos = nx.layout.spring_layout(self.G, k=10)
        nodes = [{'data' : {'id': g},
                  'position': {'x': pos[g][0], 'y': pos[g][1]}} for g in self.G]
        edges = [{'data': {'source': edge[0], 'target': edge[1],
                           'weight': 1},
                  } for edge in self.G.edges()]
        elements = nodes + edges
        
        self.app.layout = html.Div(
            [
                html.H1(children="<PUT TITLE HERE>",
                        style={'textAlign': 'left'},
                        ),
                dcc.Graph(id="live-graph", animate=True),
                dcc.Interval(
                    id="graph-update",
                    interval=1 * 1000,
                ),
                html.Div([
                    html.Div([
                        cyto.Cytoscape(id="full_graph",
                                       layout={'name': 'preset'},
                                       style={'width': "{}px".format(700), 'height': "{}px".format(500)},
                                       stylesheet=[
                                           # Group selectors
                                           {
                                               'selector': 'node',
                                               'style': {
                                                   'content': 'data(label)',
                                                   'font-size': 3, # 226,
                                                   'text-halign': 'right', # left, center, or right
                                                    'text-valign': 'center', # top, center, or bottom
                                                    'width': 0.3,
                                                    'height': 0.3,
                                               },
                                           },
                                           # Class selectors
                                           {
                                               'selector': ".node",
                                               'style': {
                                                   'content': 'data(label)',
                                                   'font-size': 3, # 220, # 326,
                                                   'text-halign': 'right', # left, center, or right
                                                   'text-valign': 'center', # top, center, or bottom
                                                   'width': 0.3,
                                                   'height': 0.3,
                                                   'background-color': 'gray',
                                                   # 'line-color': 'red'
                                                   'opacity': 0.8,
                                               },
                                           },
                                           {
                                               'selector': ".original",
                                               'style': {
                                                   'line-color': 'gray',
                                                   'curve-style': 'bezier',
                                                   'opacity': 0 # 0.6
                                               },
                                           },
                                           {
                                               'selector': 'edge',
                                               'style': {
                                                   'width': 0.05,
                                                   'line-color': 'blue',
                                                   'curve-style': 'bezier',
                                                   },
                                           },
                                           {
                                               'selector': 'node[label = 0]',
                                               "style": {
                                                   'background-color': 'red',
                                                   'z-index': 10000,
                                               },
                                           },
                                       ],
                                       elements=elements,
                                       ),
                    ]
                             )
                ]
                         )
                ]
            )
                                                        
        @self.app.callback(            
            Output("live-graph", "figure"), [Input("graph-update", "n_intervals")]
        )
        def update_graph(n):
            data = [
                go.Scatter(
                    x=list(range(len(self.data_list[symbol]))),
                    y=self.data_list[symbol],
                    mode="lines+markers",
                    name=symbol,
                )
                for symbol in self.data_list.keys()
            ]
            fig = go.Figure(data=data)

            # Update x-axis range to show last 120 data points
            fig.update_xaxes(range=[max(0, n - 120), n])

            return fig


    def run(self):
        # self.app.run_server(debug=False, host='LMPH20258.local') # , host='LMPH20258.local', port=8050) # , port=8050, host='127.0.0.1')
        self.app.run_server(debug=True, host='LMPH20258.local', use_reloader=False)
        # self.app.run_server(debug=True, port=8050, host='127.0.0.1')

    
class App:
    """ CREATING THE MAIN APPLICATION CLASS - The App class will initialize the Tkinter window and generate
    random prices for each of the financial symbols
    """                
    def __init__(self, root, G):
        self.root = root
        self.G = G
        self.data_list = {"ETHUSDT": [], "BTCUSD": [], "BNBUSDT": []}

        frame = tk.Frame()
        greeting = ttk.Label(master=frame, text="Correlated network modules", width=30, foreground="green")
        greeting.pack()
        frame.pack()
        
        

        # Start the Dash application in a separate thread
        dash_thread = DashThread(self.data_list, self.G)
        dash_thread.start()

        # Open Dash app in web browser
        # webbrowser.open("http://localhost:8050")
        webbrowser.open("http://LMPH20258.local:8050")
        # webbrowser.open("http://x86_64-apple-darwin13.4.0:8050/")
        # webbrowser.open('http://127.0.0.1:8050/')

        # Start the price generation in tkinter after Dash app is launched
        self.root.after(1000, self.generate_prices)

    def generate_prices(self):
        for symbol in self.data_list.keys():
            new_price = random.randint(1, 100)  # Generate random price
            self.data_list[symbol].append(new_price)  # Store the price in list

        # Schedule the function to run again after 1 second
        self.root.after(1000, self.generate_prices)




"""
RUNNING THE APPLICATION - Finally, we create a Tkinter root window,
instantiate the App class, and start the Tkinter event loop
"""
if __name__ == "__main__":
    root = tk.Tk()
    G = nx.karate_club_graph()
    app = App(root, G)
    root.mainloop()
        
        
