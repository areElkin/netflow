import base64
import io
import os

from dash import Dash, dcc, html, Input, Output, callback, State, dash_table
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

    .. note:: assumes nodes in `G` have ids in consecutive order from 0, 1, ..., n-1 where n is the number of nodes and that
        `G` has node attribute "name" with node label of type str
    .. note:: assumes nodes in `G2` are subset of nodes in `G` with the same ids.

    Parameters
    ----------
    G, G2: networkx Graph
        The graph and `functional` graph with correlated edges.
    show : {'combo', 'interaction', 'functional'}
        Select graph elements to show.

        Options : 

        - 'combo' : show interaction and functional network
        - 'interaction' : only show interaction network
        - 'functional' : only show functional network
    pos : dict
        Position of nodes.
    **pos_kwargs : dict
        Optional key-word arguments passed to `node_positions`.
        .. note:: this is ignored if `pos` is provided.

    Returns
    -------
    elements : dict
        dict of graph nodes and edges in JSON format {'nodes': [], 'edges': []}.
    """

    if show not in ['combo', 'interaction', 'functional']:
        raise ValueError("Unrecognized value for show, must be one of ['combo', 'interaction', 'functional'].")

    # if pos is None:
    #     if show == 'functional':            
    #         pos = node_positions(G2, **pos_kwargs)
    #     else:
    #         pos = node_positions(G, **pos_kwargs)

    # if pos is None:
    #     nx.set_node_attributes(G, {k: {'x': 0., 'y': 0.} for k in G}, name='position')
    # else:
    #     nx.set_node_attributes(G, pos, name='position')

    if pos is None:
        pos = {k: {'x': 0., 'y': 0.} for k in G}


    if show == 'functional':
        elements = {'nodes': nx.cytoscape_data(G.subgraph(G2.nodes()))['elements']['nodes'], 'edges': nx.cytoscape_data(G2)['elements']['edges']}
        # elements = nx.cytoscape_data(G2)
    else:
        G_json = nx.cytoscape_data(G)
        elements = {'nodes': G_json['elements']['nodes'], 'edges': G_json['elements']['edges']}
        if show == 'combo':
            G2_json_edges = nx.cytoscape_data(G2)['elements']['edges']
            elements['edges'] = elements['edges'] + [ee for ee in G2_json_edges if (ee['data']['source'], ee['data']['target']) not in G.edges()]
        # elements = nx.cytoscape_data(G)
    # for ee in elements['elements']['edges']:
    for vv in elements['nodes']:
        vv['position'] = pos[int(vv['data']['id'])]
        vv['group'] = 'nodes'

    for ee in elements['edges']:
        edg = (ee['data']['source'], ee['data']['target'])
        ee['group'] = 'edges'
        ee['classes'] = 'combo' if (edg in G.edges() and edg in G2.edges()) else 'interaction' if edg in G.edges() else 'functional'

    #########

    # if show == 'functional':
    #     if pos is None:
    #         nodes = [{'group': 'nodes',
    #                   'data': {'id': g, 'name': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
    #                   'position': {'x': 0, 'y': 0},
    #                   # 'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
    #                   # 'classes' : [],
    #                   } for g in G2]
    #     else:
    #         nodes = [{'group': 'nodes',
    #                   'data': {'id': g, 'name': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
    #                   'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
    #                   # 'classes' : [],
    #                   } for g in G2]
    #     el = list(G2.edges())
    # else:
    #     if pos is None:
    #         nodes = [{'group': 'nodes',
    #                   'data': {'id': g, 'name': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
    #                   'position': {'x': 0, 'y': 0},
    #                   # 'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
    #                   # 'classes' : [],
    #                   } for g in G]
    #     else:
    #         nodes = [{'group': 'nodes',
    #                   'data': {'id': g, 'name': G.nodes[g]['name'], 'cluster': G.nodes[g]['cluster']},
    #                   'position': {'x': pos[g]['x'], 'y': pos[g]['y']},
    #                   # 'classes' : [],
    #                   } for g in G]
    #     el = list(G.edges())
    #     if show == "combo":
    #         el = el + [ee for ee in G2.edges() if ee not in G.edges()] 
        
    # # el = list(G.edges()) + [ee for ee in G2.edges() if ee not in G.edges()]  # set(G.edges()) | set(G2.edges())

    # # interaction, functional, combo
    # edges = [{'group': 'edges',
    #           'data': {'source': ee[0], 'target': ee[1],
    #                    # 'weight': G.edges[ee]['weight'],
    #                    },
    #           # 'classes': ["interaction", "functional", "combo"] if (ee in G.edges() and ee in G2.edges()) else ["interaction"] if ee in G.edges() else ["functional"],
    #           'classes': 'combo' if (ee in G.edges() and ee in G2.edges()) else 'interaction' if ee in G.edges() else 'functional',
    #           } for ee in el]
    # return nodes + edges
    return elements # ['nodes'] + elements['edges']


def upload_topology(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

cose_layout = {
    'name': 'cose',
    'idealEdgeLength': 100,
    'nodeOverlap': 20,
    'refresh': 20,
    'fit': True,
    'padding': 30,
    'randomize': False,
    'componentSpacing': 100,
    'nodeRepulsion': 400000,
    'edgeElasticity': 100,
    'nestingFactor': 5,
    'gravity': 80,
    'numIter': 1000,
    'initialTemp': 200,
    'coolingFactor': 0.95,
    'minTemp': 1.0,
}

styles = {
    'container': {
        'position': 'fixed',
        'display': 'flex',
        'flex-direction': 'column',
        'height': '100%',
        'width': '100%',        
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
        'flex': '1',
        'position': 'relative',
    },
    'cytoscape': {
        'position': 'absolute',
        # 'position': 'relative',
        'width': '100%', # '90%',
        'height': '100%', # '90%',
        'z-index': 999,
        # 'border': 'thin lightgrey solid',
        # 'overflow-y': 'scroll',
    },
    'tab': {'height': 'calc(98vh - 80px)'}
}

# stylesheet with the .dbc class from dash-bootstrap-templates library
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

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
    clustering: {None, str}
        Node attribute with cluster ID used to color the nodes
    selected_graph : {'combo', 'interaction', 'functional'}
        Initial graph to render.
    node_type : {int, str}
        Data type of node ids.
    pos_kwargs : dict
        Optional key-word arguments passed to initialize node positions.
    """
    def __init__(self, G, R, R_thresh=0.9, # selected_graph='combo',
                 clustering=None,
                 node_type=int, **pos_kwargs):
        self.node_type = node_type
        self.G = G.copy()
        nx.set_edge_attributes(self.G, {ee: R.loc[ee[0], ee[1]] for ee in self.G.edges()}, name='R')
        self.R = R
        self.R_thresh = R_thresh
        self.clustering = clustering
        self.G_R = correlation_network(R, R>R_thresh, name='R')

        self.selected_graph = 'combo' # selected_graph
        self.app = Dash(__name__,
                        external_stylesheets=[dbc.themes.FLATLY, dbc_css]) # ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

        self.stylesheet = [
            # group selector for all nodes
            {'selector': 'node',
             'style': {'content': 'data(name)', # 'width': 15, 'height': 15, 'font-size': 24,
                       'background-color': 'gray', 'opacity': 0.9}},            
            {'selector': ".supernode",
             'style': {'background-color': 'green', 'opacity': 0.9, 'z-index': 998, # 10000,
                       }},
            {'selector': '[cluster = 0]',
             'style': {'background-color': 'brown', # 'width': 15, 'height': 15,
                       'opacity': 0.9, 'z-index': 998, # 10000
                       }},
            {'selector': '[cluster = 1]',
             'style': {'background-color': 'violet', 'opacity': 0.9, 'z-index': 998, # 10000,
                       }},
            {'selector': 'edge',
             'style': {'line-color': 'k', 'curve-style': 'bezier', 'opacity': 0.8, # 'width': 2,
                       }},
            {'selector': ".interaction",
             'style': {'line-color': 'gray', 'curve-style': 'bezier', 'opacity': 0.8, # 'width': 2,
                       }}, # , 'z-index': 11000}},
            {'selector': ".functional",
             'style': {'line-color': 'green', 'curve-style': 'bezier', 'opacity': 1., # 'width': 2,
                       }}, # , 'z-index': 11000}},
            {'selector': ".combo",
             'style': {'line-color': 'orange', 'curve-style': 'bezier', 'opacity': 1., # 'width': 2,
                       }}, # , 'z-index': 11000}},
            ]

        self.pos = None # node_positions(self.G, bounding_box=(0., 0., 1400, 650), weight='R', **pos_kwargs)
        # print(f"---- initial pos of node 0: {self.pos[0]} ----")
        self.elements = graph_elements(self.G, self.G_R, show=self.selected_graph, # "combo",
                                       pos=self.pos) # **pos_kwargs)
        # print(self.elements)
        # print(self.G, self.G_R, len([k for k in self.G_R.edges() if k in self.G.edges()]))


        radio_items_selected_graph_rendering = dbc.RadioItems(options=[{'label': 'Interaction network', 'value': 'interaction'},
                                                                       {'label': 'Correlated neighborhood network', 'value': 'functional'},
                                                                       {'label': 'Combo', 'value': 'combo'}], value=self.selected_graph, # 'combo',
                                                              id="radio_graph_selection",
                                                              labelStyle={"display": "flex", "align-items": "center"},
                                                              )

        # switch_fix_node_positions = html.Button('Fix positions', id='btn_fix_pos', n_clicks=0, disabled=False)
        switch_fix_node_positions = html.Div([dbc.Label("Layout", html_for="btn_fix_pos"),
                                              dbc.Switch(id="btn_fix_pos", # style=styles['container'], # class_name=, input_class_name=, input_style=, style=,
                                               disabled=False, label='Fix layout', # label_class_name=, label_style=,
                                               label_id='btn_fix_pos_label', value=False, # style=styles['in-line-container'],
                                               # class_name="ms-1",                                               
                                                         )], className="mb-3 ml-3")

        slider_R_thresh = html.Div([dbc.Label(u"\u03F1(W)", #  dcc.Markdown('$\rho_W$', mathjax=True),
                                              html_for="R-thresh-slider"),
                                    dcc.Slider(min=0.001, max=1.0, step=0.001, value=0.8, id='R-thresh-slider', # className="dbc",
                                     tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-1")], className="dbc")

        # cyto_rendering = dbc.Container([# html.Div([
        #     cyto.Cytoscape(id="cytoscape_visualization",
        #                    layout=cose_layout, # {'name': 'cose'},  # if self.pos is None else {'name': 'preset'},
        #                    # style=styles['cytoscape'], # {'height': '100vh', 'width': '100vw'}, # styles['cytoscape'],  # style={'width': "{}px".format(700), 'height': "{}px".format(500)}
        #                    stylesheet=self.stylesheet,
        #                    elements=self.elements)
        # ], # style={'border': 'thin lightgrey solid'}, # , 'width': '30vw'},
        #                                # className='pretty_container', # style=styles['cy-container'],
        #                                # className="dbc", # style=styles['cy-container'], # className="nine columns", # className='cy-container', style=styles['cy-container'],
        #                                # className="mb-3", color="primary", inverse=True,
        #                                style={'width': '100hv'})# ], # class_name='cy-container', #style=styles['cy-container'], # className="p-4") # "mx-3 mp-3")
        # )
        cyto_rendering = dbc.Card([
            dbc.CardHeader([html.H4("Network", className="card-title")], style={'color': 'primary'}),
            dbc.CardBody([
                html.Div([
                    cyto.Cytoscape(id="cytoscape_visualization",
                                   layout=cose_layout, # {'name': 'cose'},
                                   # style=styles['cytoscape'], # {'height': '100vh', 'width': '100vw'}, # styles['cytoscape'],  # style={'width': "{}px".format(700), 'height': "{}px".format(500)}
                                   style={
                                       'width': '100%',
                                       'height': '100%',
                                       'position': 'absolute', 'left': 0, 'top': 0, 'z-index': 999, # 'border': "1px solid #888",
                                   },
                                   stylesheet=self.stylesheet,
                                   elements=self.elements)
                ], # style={'position': 'relative', 'width': '100vw', 'height': '100vh'},
                         ), 
            ], style={'position': 'relative', 'background-color': 'white', # 'border': '30rem', # 'width': '100vw', # HERE ******
                      'height': '75vh'},  # className='pretty_container', # style=styles['cy-container'],
                         # className="dbc", # style=styles['cy-container'], # className="nine columns", # className='cy-container', style=styles['cy-container'],
                         # ***
                         className="mb-1 ml-1")], # ], # class_name='cy-container', #style=styles['cy-container'], # className="p-4") # "mx-3 mp-3")
                                  color="primary", inverse=True, # outline=True,                                  
                                  # style={'width': '100wv', 'heigh': '100hv'},
                                  )

        graph_tab = dbc.Row(
            children=[# html.Div([dbc.Card([
                dbc.Col([dbc.Card([
                    dbc.CardHeader([html.H4("Visualization", className="card-title"),
                                    html.H6("Rendering options", className="card-subtitle")]),
                    dbc.ListGroup([dbc.ListGroupItem(radio_items_selected_graph_rendering, color='primary'),
                                   dbc.ListGroupItem(switch_fix_node_positions, color='primary'), # html.Button('Fix positions', id='btn_fix_pos', n_clicks=0, disabled=False),
                                   dbc.ListGroupItem(slider_R_thresh, color='primary')], flush=True),
                ], # style={"width": "18rem"},  # className="d-flex flex-column", # "flex-grow-1", # "mb-1",
                                  color="primary", inverse=True,
                                  )],# className="three columns", # style=styles['border-container'],
                        width=3, # style={"height": "100%"},
                        ), # , class_name="flex-grow-1"), # className="nine columns"),
                dbc.Col(cyto_rendering, width=9, # className="mr-1",
                        # class_name="border",                                     # style={"height": "100%"},
                        ),
            ],
            className="g-2 h-75", # "h-75 g-2",
        )
        
        # btn_download = dbc.Row(dcc.Download(id="download-graph"))

        upload_tab = html.Div([dcc.Upload(id='network-upload',
                                          children=html.Div([
                                              'Drag and Drop or click to ',
                                              html.A('Select Network Topology File')
                                          ]),
                                          style={
                                              'width': '100%',
                                              'height': '60px',
                                              'lineHeight': '60px',
                                              'borderWidth': '1px',
                                              'borderStyle': 'dashed',
                                              'borderRadius': '5px',
                                              'textAlign': 'center',
                                              'margin': '10px'
                                          },
                                          # Allow multiple files to be uploaded
                                          multiple=False,
                                          ),
                               html.Div(id='output-network-upload'),
                               ])
        
        self.app.layout = dbc.Container(  # html.Div(
            [
                html.Div(html.H2(children="Cooperative network modules",
                                 style={'textAlign': 'left'},
                                 ), className="dbc"),
                dbc.Tabs(
                    [dbc.Tab(["loading", upload_tab], label="Initialization"),
                     dbc.Tab(graph_tab, label="Network visualization"),                     
                     dbc.Tab("Heatmaps", label="Data visualization"),
                     ])
            ], className="dbc", fluid=True,     # style={"height": "100vh", 'width': '100vw'},
        ) # , style=styles['container']) # className="dash-bootstrap")


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
                
        self.app.callback(Output('cytoscape_visualization', 'elements'), Output('btn_fix_pos', 'disabled'),
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

        self.app.callback(Output('output-network-upload', 'children'),
                          Input('network-upload', 'contents'),
                          State('network-upload', 'filename'),
                          # State('network-upload', 'last_modified'),
                          )(self.upload_graph)
                                

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
            # self.pos = {k['data']['id'] if self.node_type is str else int(k['data']['id']): k['position'] for k in elements if k['group'] == 'nodes'}
            # HERE: self.pos = {int(k['data']['id']): k['position'] for k in elements if k['group'] == 'nodes'}
            self.pos = {int(k['data']['id']): k['position'] for k in elements['nodes']}

            # print(f" ********** fix node positions for node 0: ({len(self.pos)}), {self.pos} **************")
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos) # **pos_kwargs)
            # print('******* fix node pos *******')
            # print(self.pos)
            # updated_elements = self.update_elements(btn_value)                
            
            return {'name': 'preset'}, self.selected_graph # , updated_elements # , "Fixed node positions."
        else:
            self.pos = None
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos)
            return cose_layout, self.selected_graph
            


    # @self.app.callback(Output('cytoscape_visualization', 'elements'),
    #                    Input('radio_graph_selection', 'value'))
    def update_elements(self, selected_graph, R_thresh):
        self.selected_graph = selected_graph

        if R_thresh != self.R_thresh:
            self.R_thresh = R_thresh
            self.G_R = correlation_network(self.R, R>self.R_thresh, name='R')
            self.elements = graph_elements(self.G, self.G_R, pos=self.pos) # **pos_kwargs)        

        # print(f"======== updated elements pos for node 0: {'None' if self.pos is None else self.pos[0]}")
        disabled = True if selected_graph == 'functional' else False
        
        return graph_elements(self.G, self.G_R, show=selected_graph, pos=self.pos), disabled


    def upload_graph(self, contents, names):
        if contents is not None:
            print(f"CONTENTS = {contents} and NAMES = {names}.")
            df = upload_topology(contents, names)

            if isinstance(df, html.Div):
                return df
            else:
                
                G = nx.from_pandas_edgelist(df)
                children = [
                    html.Div([f"Uploaded topology from {names} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."]),
                    ]
            return children
                
            
        

        

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
    
    
    
    
