
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate

import json
import networkx as nx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import scipy.stats as ss
from tqdm import tqdm
import webbrowser

from .classes import InfoNet
from .utils import stack_triu_where_

cyto.load_extra_layouts()


def correlation_network(R, condition, name='R'):
    """ Create graph from correlations `R` that satisfy given `condition`.

    Parameters
    ----------
    R : pandas DataFrame
        `n` by `n` symmetric correlations, where `n` is the number of nodes.
    condition : pandas DataFrame
        Boolean dataframe of the same size and order of rows and columns as `df` indicating values, where `True`, to include
        in the stacked dataframe.
    name : str
        Name of edge attribute that the correlations are saved as.

    Returns
    -------
    G_R : networkx Graph
        Graph with edges determined by correlations that satisfy the specified condition.
    """    
    E = stack_triu_where_(R, condition, name='R')
    E = E.reset_index().rename(columns={'level_0': 'source', 'level_1': 'target'})
    G_R = nx.from_pandas_edgelist(E, edge_attr='R')

    return G_R


def node_positions(G, bounding_box=(0., 0., 1400, 650), weight='R', **pos_kwargs):
    """ get x,y-coordinates of nodes in the graph.

    Parameters
    ----------
    G : networkx Graph
        The graph.
    bounding_box : 4-tuple
        Tuple of the form (x_min, y_min, x_max, y_max) with the dimensions of the bounding box where the nodes can be positioned.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node of the form {node: {'x': x-coordinate, 'y': y-coordinate}}
    """
    # pos = nx.layout.spring_layout(G, k=10)
    
    g = G.copy()
    nx.set_edge_attributes(g, {ee: 1000 if np.abs(g.edges[ee][weight]) < 1e-6 else np.abs(1/g.edges[ee][weight]) for ee in g.edges()}, name='weight_tmp')
    pos = nx.kamada_kawai_layout(g, weight='weight_tmp', scale=1, # center=((bounding_box[2] - bounding_box[0])/2, (bounding_box[3] - bounding_box[1])/2),                                 
                                 )

    X_MIN, Y_MIN, X_MAX, Y_MAX = bounding_box
    x = [k[0] for k in pos.values()]
    y = [k[1] for k in pos.values()]
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)

    pos = {node: {'x': (position[0] - x_min) / (x_max - x_min) * (X_MAX - X_MIN) + X_MIN,
                  'y': (position[1] - y_min) / (y_max - y_min) * (Y_MAX - Y_MIN) + Y_MIN,
                  } for node, position in pos.items()}
    return pos


def graph_elements(G, G2, show="combo", pos=None, **pos_kwargs):
    """ put graph elements in JSON format

    Parameters
    ----------
    G, G2: networkx Graph
        The graph and `functional` graph with correlated edges.
    show : {'combo', 'interaction', 'functional'}
        Select graph elements to show.

        Options
        -------
        'combo' : show interaction and functional network
        'interaction' : only show interaction network
        'functional' : only show functional network
    pos : dict
        Position of nodes.
    **pos_kwargs : dict
        Optional key-word arguments passed to `node_positions`.
        .. note:: this is ignored if `pos` is provided.

    Returns
    -------
    elements : list
        List of graph nodes and edges in JSON format.
    """

    if show not in ['combo', 'interaction', 'functional']:
        raise ValueError("Unrecognized value for show, must be one of ['combo', 'interaction', 'functional'].")

    # if pos is None:
    #     if show == 'functional':            
    #         pos = node_positions(G2, **pos_kwargs)
    #     else:
    #         pos = node_positions(G, **pos_kwargs)

    if show == 'functional':
        if pos is None:
            nodes = [{'group': 'nodes',
                      'data': {'id': g, 'label': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
                      'position': {'x': 0, 'y': 0},
                      # 'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
                      # 'classes' : [],
                      } for g in G2]
        else:
            nodes = [{'group': 'nodes',
                      'data': {'id': g, 'label': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
                      'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
                      # 'classes' : [],
                      } for g in G2]
        el = list(G2.edges())
    else:
        if pos is None:
            nodes = [{'group': 'nodes',
                      'data': {'id': g, 'label': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
                      'position': {'x': 0, 'y': 0},
                      # 'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
                      # 'classes' : [],
                      } for g in G]
        else:
            nodes = [{'group': 'nodes',
                      'data': {'id': g, 'label': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
                      'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
                      # 'classes' : [],
                      } for g in G]
        el = list(G.edges())
        if show == "combo":
            el = el + [ee for ee in G2.edges() if ee not in G.edges()] 
        
    # el = list(G.edges()) + [ee for ee in G2.edges() if ee not in G.edges()]  # set(G.edges()) | set(G2.edges())

    # interaction, functional, combo
    edges = [{'group': 'edges',
              'data': {'source': ee[0], 'target': ee[1],
                       # 'weight': G.edges[ee]['weight'],
                       },
              # 'classes': ["interaction", "functional", "combo"] if (ee in G.edges() and ee in G2.edges()) else ["interaction"] if ee in G.edges() else ["functional"],
              'classes': 'combo' if (ee in G.edges() and ee in G2.edges()) else 'interaction' if ee in G.edges() else 'functional',
              } for ee in el]
    return nodes + edges
    
styles = {
    'container': {
        'position': 'fixed',
        'display': 'flex',
        'flex-direction': 'column',
        'height': '100%',
        'width': '100%'
    },
    'in-line-container': {
        'display': 'inline-block',
        # 'border': 'thin lightgrey solid',
    },
    'border-container': {
        'border': 'thin lightgrey solid',
        'width': '99%',
    },
    'cy-container': {
        # 'flex': '1',
        'position': 'relative'
    },
    'cytoscape': {
        'position': 'absolute',
        'width': '90%', # '100%',
        'height': '90%', # '100%',
        'z-index': 999
    },
    'tab': {'height': 'calc(98vh - 80px)'}
}

radio_items_selected_graph_rendering = dcc.RadioItems(options=[{'label': 'Interaction network', 'value': 'interaction'},
                                                               {'label': 'Correlated neighborhood network', 'value': 'functional'},
                                                               {'label': 'Combo', 'value': 'combo'}], value='combo', # self.selected_graph, # 'combo',
                                                      id="radio_graph_selection",
                                                      labelStyle={"display": "flex", "align-items": "center"})

# switch_fix_node_positions = html.Button('Fix positions', id='btn_fix_pos', n_clicks=0, disabled=False)
switch_fix_node_positions = dbc.Switch(id="btn_fix_pos", # style=styles['container'], # class_name=, input_class_name=, input_style=, style=,
                                       disabled=False, label='Fix layout', # label_class_name=, label_style=, 
                                       label_id='btn_fix_pos_label', value=False, style=styles['in-line-container'])

class App:
    """ Creating Dash app

    Parameters
    ----------
    G : networkx Graph
        The interaction network.
    R : pandas DataFrame
        Symmetric pairwise correlations between nodes in `G`.
    R_thresh : float (R_thresh > 0.)
        Initial correlation threshold.
    selected_graph : {'combo', 'interaction', 'functional'}
        Initial graph to render.
    node_type : {int, str}
        Data type of node ids.
    pos_kwargs : dict
        Optional key-word arguments passed to initialize node positions.
    """
    def __init__(self, G, R, R_thresh=0.9, # selected_graph='combo',
                 node_type=int, **pos_kwargs):
        self.node_type = node_type
        self.G = G.copy()
        nx.set_edge_attributes(self.G, {ee: R.loc[ee[0], ee[1]] for ee in self.G.edges()}, name='R')
        self.R = R
        self.R_thresh = R_thresh

        self.G_R = correlation_network(R, R>R_thresh, name='R')

        self.selected_graph = 'combo' # selected_graph
        self.app = Dash(__name__,
                        external_stylesheets=[dbc.themes.FLATLY]) # ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

        self.stylesheet = [
            # group selector for all nodes
            {'selector': 'node',
             'style': {'content': 'data(label)', 'width': 15, 'height': 15, 'font-size': 24, 'background-color': 'gray', 'opacity': 0.9}},            
            {'selector': ".supernode",
             'style': {'background-color': 'green', 'opacity': 0.9, 'z-index': 10000}},
            {'selector': '[cluster = 0]',
             'style': {'background-color': 'brown', 'width': 15, 'height': 15, 'opacity': 0.9, 'z-index': 10000}},
            {'selector': '[cluster = 1]',
             'style': {'background-color': 'violet', 'opacity': 0.9, 'z-index': 10000}},
            {'selector': 'edge',
             'style': {'line-color': 'k', 'curve-style': 'bezier', 'opacity': 0.8, 'width': 2}},
            {'selector': ".interaction",
             'style': {'line-color': 'gray', 'curve-style': 'bezier', 'opacity': 0.8, 'width': 2}}, # , 'z-index': 11000}},
            {'selector': ".functional",
             'style': {'line-color': 'green', 'curve-style': 'bezier', 'opacity': 1., 'width': 2}}, # , 'z-index': 11000}},
            {'selector': ".combo",
             'style': {'line-color': 'orange', 'curve-style': 'bezier', 'opacity': 1., 'width': 2}}, # , 'z-index': 11000}},
            ]

        self.pos = None # node_positions(self.G, bounding_box=(0., 0., 1400, 650), weight='R', **pos_kwargs)
        # print(f"---- initial pos of node 0: {self.pos[0]} ----")
        self.elements = graph_elements(self.G, self.G_R, show=self.selected_graph, # "combo",
                                       pos=self.pos) # **pos_kwargs)
        # print(self.elements)
        # print(self.G, self.G_R, len([k for k in self.G_R.edges() if k in self.G.edges()]))

        self.app.layout = html.Div(            
            [
                html.H1(children="Functional network modules",
                        style={'textAlign': 'left'},
                        ),
                # dbc.Row(
                html.Div(className='row',
                         children=[dbc.Col(radio_items_selected_graph_rendering, width=2),
                                   dbc.Col(html.Div([
                             switch_fix_node_positions, # html.Button('Fix positions', id='btn_fix_pos', n_clicks=0, disabled=False),
                             # html.Button('Reset positions', id='btn_reset_pos', n_clicks=0, disabled=True),
                         ]),
                                 width=4),
                                   dbc.Col(html.Div([dcc.Slider(min=0.001, max=1.0, step=0.001, value=0.8, id='R-thresh-slider')]), width=4),
                                   # dbc.Col(html.P("Info", id="cyto_info"), width=2),
                                   ],
                         ), # className="nine columns"),
                dbc.Row([dbc.Col(html.Div([cyto.Cytoscape(id="cytoscape_visualization",
                                                          layout={'name': 'cose'},  # if self.pos is None else {'name': 'preset'},
                                                          # style={'width': "{}px".format(700), 'height': "{}px".format(500)},
                                                          style=styles['cytoscape'],
                                                          stylesheet=self.stylesheet,
                                                          elements=self.elements),
                                           ],
                                          className="nine columns",                                         # className='cy-container', style=styles['cy-container'],
                                          ),
                                 width={"size": 6}),
                         # dbc.Col(html.Div("One of three columns"), width={"size": 3, "order": 'last'}),
                         ]),
                # dbc.Row(dcc.Download(id="download-graph")),
            ], style=styles['border-container'])                


        # self.app.callback(Output('radio_graph_selection', 'value'),
        #                   Input('R-thresh-slider', 'value'), prevent_initial_call=True, allow_duplicate=True)(self.update_functional_network)

        
        # self.app.callback(Output('cytoscape_visualization', 'layout'),
        #                   Output("cyto_info", "children"),
        #                   Input('btn_fix_pos', 'n_clicks'), 
        #                   State('cytoscape_visualization', 'elements'))(self.fix_node_positions)
        self.app.callback(Output('cytoscape_visualization', 'layout'),
                          Output('radio_graph_selection', 'value'),
                          # Output('cytoscape_visualization', 'elements'),
                          # Input('btn_fix_pos', 'n_clicks'),
                          Input('btn_fix_pos', 'value'),                          
                          # Input('radio_graph_selection', 'value'),                          
                          State('cytoscape_visualization', 'elements'), prevent_initial_call=True)(self.fix_node_positions)
                
        self.app.callback(Output('cytoscape_visualization', 'elements'),
                          Input('radio_graph_selection', 'value'), Input('R-thresh-slider', 'value'))(self.update_elements)

        # @self.app.callback([Output('cytoscape_visualization', 'layout'), Output("cyto_info", "children")],
        #                    Input('btn_fix_pos', 'n_clicks'),
        #                    State('cytoscape_visualization', 'elements'))
        # def fix_node_positions(self, n_clicks, elements):
        #     if n_clicks:
        #         # new_pos = json.dumps(elements)
        #         s1 = json.dumps(self.pos)
        #         self.pos = {k['data']['id']: k['position'] for k in elements if k['group'] == 'nodes'}
        #         return {'name': 'preset'}, "Fixed node positions."


        # @self.app.callback(Output('cytoscape_visualization', 'elements'),
        #                    Input('radio_graph_selection', 'value'))
        # def update_elements(selected_graph):
        #     return graph_elements(self.G, self.G_R, show=selected_graph, pos=self.pos)
            


        # Open Dash app in web browser
        webbrowser.open("http://LMPH20258.local:8050")
        self.app.run_server(debug=True, host='LMPH20258.local', use_reloader=False)

    # def update_functional_network(self, value):
    #     self.G_R = correlation_network(self.R, R>value, name='R')
    #     self.elements = graph_elements(self.G, self.G_R, pos=self.pos) # **pos_kwargs)
    #     return self.selected_graph
        
    

    def fix_node_positions(self, value, elements):
        if value: # if n_clicks:
            # new_pos = json.dumps(elements)
            self.pos = {k['data']['id'] if self.node_type is str else int(k['data']['id']): k['position'] for k in elements if k['group'] == 'nodes'}

            print(f" ********** fix node positions for node 0: ({len(self.pos)}), {self.pos} **************")
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos) # **pos_kwargs)
            # print('******* fix node pos *******')
            # print(self.pos)
            # updated_elements = self.update_elements(btn_value)                
            
            return {'name': 'preset'}, self.selected_graph # , updated_elements # , "Fixed node positions."
        else:
            self.pos = None
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos)
            return {'name': 'cose'}, self.selected_graph
            
            
                

        


    # @self.app.callback(Output('cytoscape_visualization', 'elements'),
    #                    Input('radio_graph_selection', 'value'))
    def update_elements(self, selected_graph, R_thresh):
        self.selected_graph = selected_graph

        if R_thresh != self.R_thresh:
            self.R_thresh = R_thresh
            self.G_R = correlation_network(self.R, R>self.R_thresh, name='R')
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos) # **pos_kwargs)
        

        print(f"======== updated elements pos for node 0: {'None' if self.pos is None else self.pos[0]}")
        return graph_elements(self.G, self.G_R, show=selected_graph, pos=self.pos)
        

        

if __name__ == "__main__":
    n_obs = 120
    R_thresh = 0.2 # 0.9
    outdir = '/Users/renae12/Desktop/testing_app'
    
    G = nx.karate_club_graph()
    nx.set_node_attributes(G, {k: str(k) for k in G}, name="name")
    nx.set_node_attributes(G, {k: 0 if G.nodes[k]['club']=='Officer' else 1 for k in G}, name="cluster")

    data = pd.DataFrame(data=np.random.randn(n_obs, len(G)), index=range(n_obs), columns=list(G))

    inet = InfoNet(G, data, outdir)
    
    R, _ = ss.spearmanr(data.values)
    R = pd.DataFrame(data=R, index=data.columns, columns=data.columns)

    app = App(G, R, R_thresh=R_thresh)
    
    
    
    
