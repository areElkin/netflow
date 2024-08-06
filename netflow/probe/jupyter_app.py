import itertools

import dash_cytoscape as cyto
import dash.exceptions as dash_exceptions
# import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from dash import callback_context, Dash, html, dcc, Input, Output, State
from jupyter_dash import JupyterDash
from matplotlib import colormaps as mpl_cm

from .visualization import sin_layout, wavy_curve_layout

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'container': {
        'position': 'fixed',
        'display': 'flex',
        'flex-direction': 'row',
        'height': '98%',
        'width': '100%',
        'gap': '10px',
    },
    'control-panel-container': {
        'width': '20%', 
        'display': 'inline-block', 
        'verticalAlign': 'top', 
        'border': '1px solid black', 
        'border-radius': '20px', 
        'padding': '5px',
        'margin': '3px',
    },
    'control-panel' : {
        'display': 'inline-block',
        'overflow-y': 'scroll', 
        'height': '81%',
        'paddingRight': '15px',
        'paddingLeft': '8px',
        'paddingBottom': '3px',
        'border': 'thin solid lightgrey',
        'border-radius': '10px',
        'z-index': 999,
    },
    'cy-container': {
        'flex': '0.78',
        'position': 'relative',
        'border': '1px solid black',
        'border-radius': '20px',
    },
    'cytoscape': {
        'position': 'absolute',
        'width': '100%', # '98%',
        'height': '80%', # '83%', # '100%',
        'z-index': 999
    },
    'pre': {
        'border': 'thin lightgrey solid',
        'border-radius': '20px',
        'overflowX': 'scroll',
        'overflowY': 'scroll',
        'width': '280px', # '100%',
        # 'height': '15%',
        'padding': '10px',
        'margin': '5px',
    },
    'pre-container': {
        # 'border': 'thin lightgrey solid',
        # 'border-radius': '20px',
        'height': '18%',
        'position': 'relative',
        'display': 'flex',
        'flex-direction': 'row',
        'width': '100%',
    },
}
        
COLORMAP_OPTIONS = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'turbo',
                    'Blues', 'BrBG', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd',
                    'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples',
                    'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'YlGn',
                    'YlGnBu', 'YlOrBr', 'YlOrRd', 'gray', 'hot', 'hsv', 'jet', 'rainbow', 'icefire']
COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in COLORMAP_OPTIONS]))

def create_colorbar(values, title, color_scale='viridis'):
    """ Create Plotly colorbar figure

    Parameters
    ----------
    values : `list`
        Values used to create the colorbar.
    title : `str`
        Colorbar title.
    color_scale : `str`
        The color scale to map values to the colorbar. Must be one of plotly supported color scales.

    Return
    ------
    fig : `go.Figure`
        The Plotly colorbar figure.
    """
    values = np.array([[min(values), max(values)]])
    trace = [go.Heatmap(z=values, colorscale=color_scale, showscale=True,
                        zmin=values.min(), zmax=values.max(),
                        colorbar=dict(title=title, titleside='right', ticks='outside',
                                      # tickvals=[0, 0.5, 1], ticktext=['Low', 'Medium', 'High'],
                                      x=0),
                        ),
             go.Heatmap(z=np.zeros_like(values), # ensure there are enough points to cover the whole image
                        colorscale="Picnic_r",  # any colorscale that has white at 0
                        showscale=False,
                        zmid=0),
             ]
    fig = go.Figure(data=trace)
    fig.update_layout(width=100,
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      xaxis_zeroline=False,
                      yaxis_zeroline=False,
                      xaxis_visible=False,
                      yaxis_visible=False,
                      margin=dict(l=0, r=0, b=0, t=0),
                      )
    fig.update_traces(hoverinfo='none')
    return fig


def get_node_colors(G, node_color_attr, D=None, node_cmap='jet'):
    """ get dict with color for each node 

    Parameters
    ----------
    G : networkx.Graph
        The network.
    node_color_attr : {`str`, `int`, `numpy.array`}
        Attribute to prescribe node colors.

        - str : color nodes by node attribute in ``G``
        - int : treated as the id of an observation and the distance to that
                observation is used as the node colors, extracted from the corresponding
                row in ``D``.
        . numpy.array : expected to be the same length as the number of nodes in ``G``
                        with the values to be mapped to colors for each node,
                        ordered consecutively by node index.
    D : `numpy.ndarray` (n, n)
        Must be provided if ``node_color_attr`` is `int`, otherwise it is ignored.
        Expected to be a symmetric matrix with values to be used as node colors. 
        When ``node_color_attr`` is an `int`, each node ``i`` is colored according to
        ``D[node_color_attr, i]``.
    node_cmap : `str`
        Colormap applied to nodes, must be found in ``matplotlib.colormaps``.    

    Returns
    -------
    node_color_map : `dict`
        The color for each node, keyed by the node index.
    cbar : `go.Figure`
        The Plotly colorbar figure.
    """
    if isinstance(node_color_attr, int):
        node_colors = D[node_color_attr, list(G.nodes())]
    elif isinstance(node_color_attr, str):
        node_colors = [G.nodes[node][node_color_attr] for node in G.nodes()]        
    else:  # numpy.array
        node_colors = node_color_attr[list(G.nodes())]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in set(node_colors)):
        map = {k: ix for ix, k in enumerate(sorted(set(node_colors)))}
        node_colors = [map[k] for k in node_colors]

    norm = colors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
    cmap = mpl_cm.get_cmap(node_cmap)

    node_color_map = {node: colors.rgb2hex(cmap(norm(val))) for node, val in zip(G.nodes(), node_colors)}
    cbar = create_colorbar(node_colors, 'nodes', color_scale=node_cmap)
    return node_color_map, cbar


def get_edge_colors(G, edge_color_attr, edge_cmap='jet'):
    """ get dict with color for each edge from edge attribute in the graph

    Parameters
    ----------
    G : networkx.Graph
        The network.
    edge_color_attr : `str`
        Attribute to prescribe node colors.
    edge_cmap : `str`
        Colormap applied to edges, must be found in ``matplotlib.colormaps``.

    Returns
    -------
    edge_color_map : `dict`
        The color for each edge, keyed by the edge.
    cmap : `go.Figure`
        The Plotly colorbar figure.
    """
    edge_colors = [G.edges[edge][edge_color_attr] for edge in G.edges()]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in set(edge_colors)):
        map = {k: ix for ix, k in enumerate(sorted(set(edge_colors)))}
        edge_colors = [map[k] for k in edge_colors]    
    norm = colors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
    cmap = mpl_cm.get_cmap(edge_cmap)
    edge_color_map = {edge: colors.rgb2hex(cmap(norm(val))) for edge, val in zip(G.edges(), edge_colors)}
    # edge_color_map = {edge: colors.rgb2hex(cmap(norm(G.edges[edge][edge_color_attr]))) for edge in G.edges()}
    cbar = create_colorbar(edge_colors, 'edges', color_scale=edge_cmap)
    return edge_color_map, cbar


def nx_to_cytoscape(G, pos=None,
                    node_color_attr=None, D=None, node_cmap='jet', default_node_color='#888', 
                    edge_color_attr=None, edge_cmap='jet', positions_records=None, return_cbar=False):
    """ Convert networkx graph to cytoscape syntax

    Parameters
    ----------
    G : networkx.Graph
        The network.
    pos : {`None`, `dict`}
        Node positions as returned from ``networkx.layout`` in the form {node: np.array([x, y])}.
        If `None`, default computes layout from ``network.layout.kamada_kawai``.
    node_color_attr : {`str`, `int`, `numpy.array`}
        Attribute to prescribe node colors.

        - str : color nodes by node attribute in ``G``
        - int : treated as the id of an observation and the distance to that
                observation is used as the node colors, extracted from the corresponding
                row in ``D``.
        . numpy.array : expected to be the same length as the number of nodes in ``G``
                        with the values to be mapped to colors for each node,
                        ordered consecutively by node index.
    D : `numpy.ndarray` (n, n)
        Must be provided if ``node_color_attr`` is `int`, otherwise it is ignored.
        Expected to be a symmetric matrix with values to be used as node colors. 
        When ``node_color_attr`` is an `int`, each node ``i`` is colored according to
        ``D[node_color_attr, i]``.
    node_cmap : `str`
        Colormap applied to nodes, must be found in ``matplotlib.colormaps``.
    default_node_color : `str` 
        The default node color. 
    edge_color_attr : `str`
        Attribute to prescribe node colors.
    edge_cmap : `str`
        Colormap applied to edges, must be found in ``matplotlib.colormaps``.
    positions_records : {`None`, `dict`}
        Optional. Provide dictionary of pre-computed node positions keyed by
        the name of the layout. If provided, it will be updated if a new layout
        position is computed.
    return_cbar : `bool`
        If `True`, also return edge and node colorbars.

    Returns
    -------
    elements : `dict`
        The cytoscape network elements.
    node_cbar_vis : {`go.Figure`, `None`}
        Only returned if ``return_cbar=True``. If the node colors are mapped to a feature value,
        the corresponding colorbar figure is returned as a `go.Figure`. Otherwise, `None` is returned.
    edge_cbar_vis : {`go.Figure`, `None`}
        Only returned if ``return_cbar=True``. If the edge colors are mapped to a feature value,
        the corresponding colorbar figure is returned as a `go.Figure`. Otherwise, `None` is returned.
    """
    if positions_records is None:
        positions_records = {}
        
    elements = {'nodes': nx.cytoscape_data(G)['elements']['nodes'],
                'edges': nx.cytoscape_data(G)['elements']['edges']}

    # set up node colors:
    if node_color_attr is not None:
        node_color_map, node_cbar_vis = get_node_colors(G, node_color_attr, D=D, node_cmap=node_cmap)

        # Note: can use this when adding a colorbar
        # if isinstance(node_color_attr, str):
        #     label = node_color_attr
        # elif isinstance(node_color_attr, int):
        #     label = f"pseudo-distance from {G.nodes[node_color_attr]['name']}"
        # else:
        #     label = "feature"
    else:
        node_color_map = {node: default_node_color for node in G.nodes()}  # Default color
        node_cbar_vis = None

    # set up node positions:
    if pos is None:
        if 'kamada_kawai' in positions_records:
            pos = positions_records['kamada_kawai']
        else:                                    
            pos = nx.kamada_kawai_layout(G)

    bounding_box=(3., 3., 400, 430)
    X_MIN, Y_MIN, X_MAX, Y_MAX = bounding_box
    x = [k[0] for k in pos.values()]
    y = [k[1] for k in pos.values()]
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    pos = {node: np.array([(position[0] - x_min) / (x_max - x_min) * (X_MAX - X_MIN) + X_MIN,
                           (position[1] - y_min) / (y_max - y_min) * (Y_MAX - Y_MIN) + Y_MIN,
                           ]) for node, position in pos.items()}

    # set up nodes:
    for vv in elements['nodes']:
        vv['data']['color'] = node_color_map[int(vv['data']['id'])]
        vv['position'] = dict(zip(['x', 'y'], pos[int(vv['data']['id'])]))
        # vv['group'] = 'nodes'
        vv['classes'] = 'nodes'    
        
    # set up edge colors:
    if edge_color_attr:
        edge_color_map, edge_cbar_vis = get_edge_colors(G, edge_color_attr, edge_cmap=edge_cmap)
    else:
        edge_color_map = {edge: '#222' for edge in G.edges()}  # Default color
        edge_cbar_vis = None

    # set up edges:
    for ee in elements['edges']:
        edg = (ee['data']['source'], ee['data']['target'])
        # ee['group'] = 'edges'
        ee['classes'] = 'edges'
        ee['data']['color'] = edge_color_map[edg]

    if return_cbar:
        return elements, node_cbar_vis, edge_cbar_vis
    else:
        return elements


def renderer(keeper, G, distance_key):
    """ Construct the interactive POSE visualization for rendering in a JupyterLab notebook

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    G : `networkx.Graph`
        The network (intended to be the POSE network).
    distance_key : `str`
        The key to reference the distance stored in ``keeper`` used to
        identify node colors relative to a particular observation
        (intended to be the distance used to construct the POSE).

    Returns
    -------
    app : `JupyterDash`
        The app object.
    """
    # record computed positions
    positions_records = {}
    D = keeper.distances[distance_key].data

    # Initialize the app
    # app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
    app = JupyterDash(__name__)

    # Define the layout of the app
    app.layout = html.Div(style=styles['container'],
                          children=[
                              html.Div(className='control-panel-container',
                                       style=styles['control-panel-container'], children=[
                                           html.H2("NetFlow POSE"),
                                           html.H3("Settings"),
                                           html.Div([
                                               dcc.Store(id='container-dimensions'),
                                               html.H4("Graph Layout"),
                                               dcc.Dropdown(
                                                   id='layout-dropdown',
                                                   options=[{'label': layout,
                                                             'value': layout} for layout in ['kamada_kawai', 'spring',
                                                                                             'circular', # 'cose',
                                                                                             'shell', 'grid',
                                                                                             'breadthfirst', 'sin',
                                                                                             'wavy_spiral']],
                                                   value='kamada_kawai',
                                               ),
                                               html.H4("Highlight Node by Label"),
                                               dcc.Input(id='node-label', type='text', value=''),
                                               html.Button('Highlight Node', id='highlight-button', n_clicks=0),
                                               html.H4("Node Size"),
                                               dcc.Slider(id='node-size-slider', min=0.01, max=30, step=0.01, value=8,
                                                          marks={1: 'smaller', 30: 'larger'}),
                                               html.H4("Set node color"),
                                               dcc.Input(id='node-fixed-color', type='text', value=''),
                                               html.Button('Color nodes', id='fixed-color-button', n_clicks=0),
                                               html.H4("Node Attribute for Coloring"),
                                               dcc.Dropdown(
                                                   id='node-attribute-dropdown',
                                                   options=[{'label': 'None',
                                                             'value': 'None'}] + [{'label': attr,
                                                                                   'value': attr} for attr in G.nodes(data=True)[0].keys()],
                                                   value='None',
                                               ),
                                               html.H4("Data Feature for Coloring Nodes"),
                                               dcc.Dropdown(
                                                   id='keeper-data-dropdown',
                                                   options=[{'label': 'None',
                                                             'value': 'None'}] + [{'label': dataset.label,
                                                                                   'value': dataset.label} for dataset in keeper.data],
                                                   value='None', # 'branch'
                                               ),
                                               dcc.Dropdown(id='feature-label', options=[], value='None'),
                                               html.Button('Color nodes', id='node-color-button', n_clicks=0),
                                               html.H4("Colormap for Nodes"),
                                               dcc.Dropdown(
                                                   id='node-colormap-dropdown',
                                                   options=[{'label': cmap,
                                                             'value': cmap} for cmap in COLORMAP_OPTIONS], # mpl.colormaps()], # ['YlGnBu', 'viridis', 'cividis', 'jet', 'nipy_spectral', 'gist_ncar']],
                                                   value='YlGnBu',
                                               ),
                                               html.H4("Edge Width"),
                                               dcc.Slider(id='edge-width-slider', min=0.001, max=14, step=0.001, value=1.5,
                                                          marks={1: 'thinner', 14: 'wider'}),
                                               html.H4("Edge Attribute for Coloring"),
                                               dcc.Dropdown(
                                                   id='edge-attribute-dropdown',
                                                   options=[{'label': 'None',
                                                             'value': 'None'}] + [{'label': attr,
                                                                                   'value': attr} for attr in list(G.edges(data=True))[0][-1].keys()],
                                                   value='None', 
                                               ),
                                               html.H4("Colormap for Edges"),
                                               dcc.Dropdown(
                                                   id='edge-colormap-dropdown',
                                                   options=[{'label': cmap,
                                                             'value': cmap} for cmap in COLORMAP_OPTIONS], # mpl.colormaps()], # ['YlGnBu', 'viridis', 'cividis', 'jet', 'nipy_spectral', 'gist_ncar']],
                                                   value='hot_r',
                                               ),                                               
                                               dcc.Interval(id='interval', interval=1000, n_intervals=0),  # Interval to capture dimensions periodically
                                           ],
                                                    style=styles['control-panel'], # styles['control-panel-container'],
                                                    ),
                                       ]),
                              html.Div(className='cy-container', style=styles['cy-container'], children=[
                                  html.Div(id='top-panel', style=styles['pre-container'], children=[
                                      html.Pre(id='cytoscape-mouseoverNodeData', style=styles['pre']),
                                      html.Div(id='node-colorbars-div', style={'height': '100%',
                                                                               # 'overflow': 'auto',
                                                                               'display': 'flex', # 'width': '30px',
                                                                               },
                                               children=[],
                                               ),
                                      html.Div(id='edge-colorbars-div', style={'height': '100%', 'display': 'flex'},
                                               children=[],
                                               ),
                                  ],
                                           ),
                                  # html.Div(children=[]
                                  cyto.Cytoscape(
                                      id='network-graph',
                                      elements=nx_to_cytoscape(G),
                                      style=styles['cytoscape'],
                                      layout={'name': 'preset', "fit": False},  # {'name': 'cose'},
                                      selectable=True,
                                      boxSelectionEnabled=True,
                                      zoom=1.,  # Initial zoom level
                                      pan={'x': 0., 'y': 0.},  # Initial pan position
                                      stylesheet=[
                                          {
                                              'selector': 'node',
                                              'style': {
                                                  # 'label': '', # 'data(name)',
                                                  # 'content': '',
                                                  'font-size': '8px',
                                                  'width': 8, # 'data(size)',
                                                  'height': 8, # 'data(size)',
                                                  'background-color': '#888', # 'tan', # 'data(color)',
                                                  'text-opacity': 0.,  # Hide labels by default,
                                                  'border-width': '0.5px',  # Adds a thin border around the nodes
                                                  'border-color': '#999',  # Sets the border color
                                              },
                                          },
                                          {
                                              'selector': 'edge',
                                              'style': {
                                                  'width': 1, # 'data(width)',
                                                  'line-color': '#222', # 'darkgray', # 'data(color)'
                                              }
                                          },
                                          # Class selectors
                                          {
                                              'selector': '.highlight',
                                              'style': {
                                                  'border-width': '2px', # 20,
                                                  'border-color': 'black',
                                                  # 'outline-width': 20,
                                                  # 'outline-color': 'black',
                                              },
                                          },
                                      ],
                                      # responsive=True,
                                  ),
                              ]),
                          ])

    app.clientside_callback(
        '''
        function(n_intervals) {
            const container = document.getElementById('network-graph');
            const dimensions = {width: container.offsetWidth, height: container.offsetHeight};
            return dimensions;
        }
        ''',
        Output('container-dimensions', 'data'),
        Input('interval', 'n_intervals')
    )

    @app.callback(
        Output('output', 'children'),
        Input('network-graph', 'selectedNodeData')
    )
    def display_selected_nodes(data):
        if not data:
            return "No nodes selected."
        return f"Selected nodes: {', '.join([node['name'] for node in data])}"

                                               
    @app.callback(
        Output('feature-label', 'options'),
        Input('keeper-data-dropdown', 'value'),
    )
    def set_feature_options(data_label):
        """ return list of feature options for given data in the keeper """
        if data_label == 'None':
            options = [] # [{'label': 'None', 'value': 'None'}]
        else:
            # uncomment to ensure only int/float dtypes are included in feature list
            # is_numeric = lambda x: np.issubdtype(keeper.data[data_label].data[x, :].dtype,
            #                                      np.floating) or np.issubdtype(keeper.data[data_label].data[x, :].dtype,
            #                                                                    np.integer)
            # options = [{'label': ft,
            #             'value': ft} for ft in sorted([k for ix, k in enumerate(keeper.data[data_label].feature_labels) if is_numeric(ix)])]
            options = [{'label': ft,
                        'value': ft} for ft in sorted(keeper.data[data_label].feature_labels)]
        return options    

    
    @app.callback(
        Output('network-graph', 'elements'),
        Output('network-graph', 'stylesheet'),
        Output('node-attribute-dropdown', 'value'),
        Output('keeper-data-dropdown', 'value'),
        Output('feature-label', 'value'),
        Output('network-graph', 'zoom'),
        Output('network-graph', 'pan'),
        Output('node-colorbars-div', 'children'),
        Output('edge-colorbars-div', 'children'),
        Output('network-graph', 'tapNodeData'),
        Output('node-fixed-color', 'value'), 
        Input('layout-dropdown', 'value'), 
        Input('highlight-button', 'n_clicks'), 
        State('node-label', 'value'), 
        Input('node-size-slider', 'value'), 
        Input('edge-width-slider', 'value'),
        Input('node-attribute-dropdown', 'value'),
        Input('edge-attribute-dropdown', 'value'),
        Input('node-colormap-dropdown', 'value'), 
        Input('edge-colormap-dropdown', 'value'),
        Input('network-graph', 'tapNodeData'),
        State('network-graph', 'elements'),
        State('container-dimensions', 'data'),
        State('network-graph', 'zoom'),
        State('network-graph', 'pan'),
        Input('node-color-button', 'n_clicks'),    
        State('keeper-data-dropdown', 'value'),
        State('feature-label', 'value'),
        State('node-fixed-color', 'value'), 
        Input('fixed-color-button', 'n_clicks'), 
        # prevent_initial_call=True,
        # suppress_callback_exceptions=True,
    )
    def update_graph(layout, n_clicks, node_label, 
                     node_size, 
                     edge_width, 
                     node_attr, edge_attr, node_cmap, edge_cmap, 
                     tap_node_data,
                     elements_in, dimensions,
                     current_zoom, current_pan, feature_n_clicks, data_label, ft_label,
                     node_fixed_color, node_fixed_color_n_clicks): 
        # Determine which input triggered the callback
        triggered_input = callback_context.triggered[0]['prop_id'].split('.')[0]

        if not dimensions:
            raise dash_exceptions.PreventUpdate
        width, height = dimensions['width'], dimensions['height']

        if layout in positions_records.keys():
            pos = positions_records[layout]
        else:
            if layout == 'spring':
                pos = nx.spring_layout(G)
                positions_records[layout] = pos
            elif layout == 'circular':
                pos = nx.circular_layout(G)
                positions_records[layout] = pos
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
                positions_records[layout] = pos
            elif layout == 'shell':        
                pos = nx.shell_layout(G)  
                positions_records[layout] = pos
            elif layout == 'grid':
                pos = nx.planar_layout(G)
                positions_records[layout] = pos
            elif layout == 'sin':
                pos = sin_layout(G, keeper.distances[distance_key].to_frame())
                positions_records[layout] = pos
            elif layout == 'wavy_spiral':
                pos = wavy_curve_layout(G, keeper.distances[distance_key].to_frame(),
                                        b=0.7*np.pi, # 0.75*np.pi, # 0.5*np.pi,
                                        C=0,
                                        h=0.5, # 8,
                                        t_start=1.5,
                                        sep=1)                      
                positions_records[layout] = pos
            else:
                layout = 'kamada_kawai'
                pos = nx.kamada_kawai_layout(G)
                positions_records[layout] = pos


        if pos is not None:
            bounding_box=(3., 3., width-3., height-3.)
            X_MIN, Y_MIN, X_MAX, Y_MAX = bounding_box
            x = [k[0] for k in pos.values()]
            y = [k[1] for k in pos.values()]
            x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
            pos = {node: np.array([(position[0] - x_min) / (x_max - x_min) * (X_MAX - X_MIN) + X_MIN,
                                   (position[1] - y_min) / (y_max - y_min) * (Y_MAX - Y_MIN) + Y_MIN,
                                   ]) for node, position in pos.items()}

        # print(f"before - tap_node_data = {tap_node_data}")
        # node_fixed_color, node_fixed_color_n_clicks
        # 'fixed-color-button'
        if triggered_input == 'fixed-color-button': 
            node_attr_value = 'None'
            data_label = 'None'
            ft_label = ''
            tap_node_data = None            
        elif triggered_input == 'network-graph':
            node_attr = int(tap_node_data['id'])
            node_attr_value = 'None'
            data_label = 'None'
            ft_label = ''
            node_fixed_color = '' 
        elif triggered_input == 'node-attribute-dropdown':
            node_attr_value = node_attr
            data_label = 'None'
            ft_label = ''
            tap_node_data = None
            node_fixed_color = '' 
        elif triggered_input == 'node-color-button':
            node_attr_value = 'None'
            tap_node_data = None
            node_fixed_color = '' 
        else:
            if tap_node_data is None: 
                node_attr_value = node_attr 
            else: 
                node_attr = int(tap_node_data['id'])
                node_attr_value = 'None'
                data_label = 'None'
                ft_label = ''
                node_fixed_color = ''

        # print(f"after - tap_node_data = {tap_node_data}")

        if data_label != 'None':
            nc = keeper.data[data_label].subset(features=[ft_label]).loc[ft_label].values

        nca = None if (node_attr=='None' and data_label=='None') else node_attr if node_attr!='None' else nc
        eca = None if edge_attr=='None' else edge_attr
        nfc = node_fixed_color if node_fixed_color != '' else '#888' 
        elements, node_cbar_vis, edge_cbar_vis = nx_to_cytoscape(G, pos=pos, 
                                                                 node_color_attr=nca,
                                                                 default_node_color=nfc, 
                                                                 D=D,node_cmap=node_cmap,
                                                                 edge_color_attr=eca,
                                                                 edge_cmap=edge_cmap,
                                                                 positions_records=positions_records,
                                                                 return_cbar=True)

        if node_cbar_vis is None:
            node_cbar_vis = []
        else:
            node_cbar_vis = [dcc.Graph(figure=node_cbar_vis,
                                       config={'displayModeBar': False},  # Hides the Plotly mode bar
                                       )]

        if edge_cbar_vis is None:
            edge_cbar_vis = []
        else:
            edge_cbar_vis = [dcc.Graph(figure=edge_cbar_vis,
                                       config={'displayModeBar': False},
                                       )]

        if (pos is None) and (layout not in positions_records):
            positions_records[layout] = {int(k['data']['id']): np.array([k['position']['x'], k['position']['y']]) for k in elements['nodes']}

        # Highlight node by label
        if node_label:
            for element in elements['nodes']:
                if element['data'].get('name') == node_label:
                    element['classes'] += ' highlight'
                    break

        stylesheet=[

            {
                'selector': 'node',
                'style': {
                    # 'label': 'data(name)',
                    'width': node_size, # 'data(size)',
                    'height': node_size, # 'data(size)',
                    'background-color': 'data(color)',
                    'text-opacity': '0',  # Hide labels by default,   
                    'font-size': '8px',
                    'border-width': '0.5px',  # Adds a thin border around the nodes
                    'border-color': '#999',  # Sets the border color
                    # 'z-index': 1001,
                },
            },
            {
                'selector': 'edge',
                'style': {
                    'width': edge_width, # 'data(width)',
                    'line-color': 'data(color)'
                },
            },
            # Class selectors
            {
                'selector': '.highlight',
                'style': {
                    'border-width': '3px',
                    'border-color': 'black',
                    # 'outline-width': 20,
                    # 'outline-color': 'black',
                    # 'z-index': 1001,
                },
            },
        ]

        return elements, stylesheet, node_attr_value, data_label, ft_label, current_zoom, current_pan, node_cbar_vis, edge_cbar_vis, tap_node_data, node_fixed_color

    @app.callback(Output('cytoscape-mouseoverNodeData', 'children'),
              Input('network-graph', 'mouseoverNodeData'))
    def displayHoverNodeData(data):
        # return json.dumps(data, indent=2) # f"You selected node {data} # {data['id']} -- {data['name']}" #
        if data is None:
            return "Hover node data"
        return "\n".join([f"{k}: {data[k] if not k.startswith('pse') else np.round(data[k], 3)}" for k in ['name',
                                                                                                           'id', 'is_root',
                                                                                                           'undecided',
                                                                                                           'unidentified',
                                                                                                           'pseudo-distance from root',
                                                                                                           ]])

    
    # suppress_callback_exceptions=True

    # app.run_server(mode='jupyterlab', host="127.0.0.1", port=8090, dev_tools_ui=True, debug=False,
    #           dev_tools_hot_reload=True, threaded=True)
    return app

    
def render_pose(keeper, G, distance_key):    
    """ Render the interactive POSE visualization in a JupyterLab notebook

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    G : `networkx.Graph`
        The network (intended to be the POSE network).
    distance_key : `str`
        The key to reference the distance stored in ``keeper`` used to
        identify node colors relative to a particular observation
        (intended to be the distance used to construct the POSE).
    """
    app = renderer(keeper, G, distance_key)
    app.run_server(mode='jupyterlab', host="127.0.0.1", port=8090, dev_tools_ui=True, debug=True, # False,
              dev_tools_hot_reload=True, threaded=True)
