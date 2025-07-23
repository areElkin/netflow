import itertools
import io

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash.exceptions as dash_exceptions
import dash_table
# import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as colors
import networkx as nx
import numpy as np
from openpyxl import Workbook
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go


from dash import callback_context, Dash, html, dcc, Input, Output, State, no_update
from datetime import datetime
from jupyter_dash import JupyterDash
from lifelines.statistics import logrank_test
from matplotlib import colormaps as mpl_cm

from .visualization import sin_layout, wavy_curve_layout forceatlas2_layout
from ..methods.stats import stat_test

from ..methods import stats as ms

# from importlib import reload
# from . import visualization as nfv
# reload(nfv)
# forceatlas2_layout = nfv.forceatlas2_layout

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'container': {
        'position': 'fixed',
        'display': 'flex',
        'flex-direction': 'row',
        'height': '98%',
        'width': '100%',
        'gap': '3px',
        # 'paddingRight': '15px',
        'pad': '5px', 
    },
    'control-panel-container': {
        # 'width': '18%',
        # 'flex': '1 1 0', # '0.12',
        'flex': '0 1 18%',
        'display': 'block', # 'inline-block', 
        'verticalAlign': 'top', 
        'border': '1px solid black', 
        'border-radius': '20px', 
        'padding': '5px',
        'margin': '3px',
    },
    'control-panel' : {
        'display': 'block', # 'inline-block',
        'overflow-y': 'scroll', 
        'height': '81%',
        'paddingRight': '13px', # '15px', # changed from '15px' when switched to fieldset
        # 'paddingLeft': '8px', # commented out when switched to fieldset
        'paddingBottom': '3px',
        # 'border': 'thin solid lightgrey',
        # 'border-radius': '10px',
        'gap': '3px',
        'z-index': 999,
    },
    'fieldset-panel': {
        'border': 'thin solid lightgrey', # #614051',
        'border-radius': '10px',
        'margin-bottom': '6px',
    },
    'fieldset-panel-stat': {
        'border': 'thin solid lightgrey', # #614051',
        'border-radius': '10px',
        'margin-bottom': '6px',
        'overflow-y': 'scroll',
        'height': '37%', # '34%',
    },
    'stat-panel-container': {
        'width': '20%',
        # 'flex': '1.2 1.1 0', # '0.15',
        # 'flex': '0 1.1 20%',
        # 'flex': '0 0 20%',
        'display': 'block', # 'inline-block',
        'verticalAlign': 'top', 
        'border': '1px solid black', 
        'border-radius': '20px', 
        'padding': '3px',
        'margin': '3px',
        # 'overflow-y': 'scroll', 
    },
    'stat-panel' : {        
        'display': 'block', # 'inline-block',
        'overflow-y': 'scroll', 
        'height': '37%', # '34%',
        'paddingRight': '15px',
        'paddingLeft': '8px',
        'paddingBottom': '3px',
        'border': 'thin solid lightgrey',
        'border-radius': '10px',
        'width': '88%', # here
        # 'position': 'relative',
        'z-index': 999,
    },
    'cy-container': {
        # 'width': '63%',
        # 'flex': '2.7 2.5 0', # '0.63', # '0.84', # '0.78', # '0.58', # '0.78', # '0.58', # '0.78',
        'flex': '0 2.7 47%', # 53%',
        'position': 'relative',
        'border': '1px solid black',
        'border-radius': '20px',
    },
    'cytoscape': {
        'position': 'absolute',
        'width': '100%', # '98%',
        'height': '80%', # '83%', # '100%',
        'z-index': 999,
    },
    'pre': {
        'border': 'thin lightgrey solid',
        'border-radius': '20px',
        'overflowX': 'scroll',
        'overflowY': 'scroll',
        'width': '250px', # '280px', # '100%',
        # 'height': '15%',
        'padding': '10px',
        'margin': '5px',
        'color': 'gray',
    },
    'pre-container': {
        # 'border': 'thin lightgrey solid',
        # 'border-radius': '20px',
        'height': '18%',
        'position': 'relative',
        'display': 'flex',
        'flex-direction': 'row',
        'width': '100%',
        'padding': '3px',
    },
    'stat-table': {
        'padding': '3px',
        'height': '29%',
        # 'width': '100%', # MY NEW
        # 'dislay': 'block',
        'overflowY': 'auto',
        'margin-top': '10px',
        # 'border': 'thin lightgrey solid',
    },
    'pre_box_labels': {
        'border': 'thin lightgrey solid',
        'border-radius': '10px',
        # 'overflowX': 'scroll',
        'overflowY': 'scroll',
        'white-space': 'pre-wrap',       # Wrap text and preserve whitespace
        'overflow-wrap': 'break-word',   # Break long words if necessary
        # 'width': '250px', # '280px', # '100%',
        # 'height': '15%',
        'padding': '3px',
        'margin': '3px',
        'height': '10%',
        'paddingBottom': '3px',
        'color': 'gray',
        # 'width': '24%', # MY NEW
    },
    'label': {
        'display': 'block',
        # 'font-weight': 'bold',
        'font-size': '12',
        'color': '#696969',
    },
    'label_tspace': {
        'display': 'block',
        # 'font-weight': 'bold',
        'font-size': '12',
        'color': '#696969',
        'margin-top': '8px',
    },
    'label_b': {
        'display': 'block',
        'font-weight': 'bold',
        'font-size': '14',
        'color': '#787878',
        'margin-bottom': '3px',
        'margin-top': '13px',
    },
    'h4': {
        'margin-bottom': '3px',
    },
    'h3': {
        'margin-bottom': '3px',
        'margin-top': '10px',
    },
    'fieldset-legend': {
        'font-size': '14px',
        # 'color': '#614051',
        # 'font-weight': 'bold',
    },
}

# COLORMAP_OPTIONS = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'turbo',
#                     'Blues', 'BrBG', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'OrRd',
#                     'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples',
#                     'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'YlGn',
#                     'YlGnBu', 'YlOrBr', 'YlOrRd', 'gray', 'hot', 'hsv', 'jet', 'rainbow', 'icefire']
# COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in COLORMAP_OPTIONS]))

COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in pc.named_colorscales()]))
# DISCRETE_COLORMAP_OPTIONS = pc.qualitative.__dict__.keys()
# DISCRETE_COLORMAP_OPTIONS = [cs for cs in DISCRETE_COLORMAP_OPTIONS if not cs.startswith('_')]
# DISCRETE_COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in DISCRETE_COLORMAP_OPTIONS]))

SEQUENTIAL_COLORMAP_OPTIONS = [cs for cs in pc.sequential.__dict__.keys() if not (cs.startswith('_') or cs.startswith('swatch'))]
SEQUENTIAL_COLORMAP_OPTIONS.append('nipy_spectral') #  += ['nipy_spectral', 'nipy_spectral_r']
SEQUENTIAL_COLORMAP_OPTIONS = sorted(set([k.split('_r')[0] for k in SEQUENTIAL_COLORMAP_OPTIONS]), key=lambda x: x.lower())
SEQUENTIAL_COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in SEQUENTIAL_COLORMAP_OPTIONS]))


DIVERGING_COLORMAP_OPTIONS = [cs for cs in pc.diverging.__dict__.keys() if not (cs.startswith('_') or cs.startswith('swatch'))]
DIVERGING_COLORMAP_OPTIONS = sorted(set([k.split('_r')[0] for k in DIVERGING_COLORMAP_OPTIONS]))
DIVERGING_COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in DIVERGING_COLORMAP_OPTIONS]))

DISCRETE_COLORMAP_OPTIONS = [cs for cs in pc.qualitative.__dict__.keys() if not (cs.startswith('_') or cs.startswith('swatch'))]
DISCRETE_COLORMAP_OPTIONS = sorted(set([k.split('_r')[0] for k in DISCRETE_COLORMAP_OPTIONS]))
DISCRETE_COLORMAP_OPTIONS = list(itertools.chain(*[[k, k+'_r'] for k in DISCRETE_COLORMAP_OPTIONS]))


def matplotlib_to_plotly_cmap(cmap, n=11, precision=2):
    """ Convert matplotlib colormap to a plotly colorscale

    Parameters
    ----------
    cmap :
        The matplotlib colormap.
    n : `int`
        The number of entries considered for the Plotly colorscale.
    precision : `int`
        The number of digits considered for rounding the scale values.

    Returns
    -------
    plotly_colorscale :
        The plotly colorscale.
    """
    scale = np.linspace(0, 1, n)
    colors = (cmap(scale)[:, :3]*255).astype(np.uint8)
    plotly_colorscale = [[round(s, precision), f'rgb{tuple(color)}'] for s, color in zip(scale, colors)]
    return plotly_colorscale
nipy_spectral_cmap = mpl_cm.get_cmap('nipy_spectral')
nipy_spectral_pl = matplotlib_to_plotly_cmap(nipy_spectral_cmap, n=255, precision=4)
nipy_spectral_r_pl = [[a,b] for a,b in zip([k[0] for k in nipy_spectral_pl], [k[1] for k in nipy_spectral_pl][::-1])]

# add new for exit button
# import time
# import signal
# import flask
# import threading
# import os

# def shutdown_server():
#     # Function to shutdown the server gracefully
#     def shutdown():
#         time.sleep(1)
#         # Shutdown the server
#         # func = flask.request.environ.get('werkzeug.server.shutdown')
#         # if func is None:
#         #     raise RuntimeError('Not running with the Werkzeug Server')
#         # func()
#         os.kill(os.getpid(), signal.SIGINT)

#     thread = threading.Thread(target=shutdown)
#     thread.start()
    
# end add new for exit button

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
    # if color_scale in COLORMAP_OPTIONS:
    values = np.array([[min(values), max(values)]])
    
    if color_scale == 'nipy_spectral':
        cs = nipy_spectral_pl
    elif color_scale == 'nipy_spectral_r':
        cs = nipy_spectral_r_pl
    else:
        cs = color_scale
    trace = [go.Heatmap(z=values,
                        colorscale=cs, showscale=True,
                        zmin=values.min(), zmax=values.max(),
                        colorbar=dict(title=dict(text=title, font=dict(size=12, family='Arial')), # title,
                                      titleside='right', ticks='outside',
                                      tickfont=dict(size=12, family='Arial'),                                                 
                                      # family='Arial', size=12,
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



def create_legend(values, title, color_scale=None):
    """ Create Plotly legend figure

    Parameters
    ----------
    values : {`list`, 'dict'}    
        Values used to create the legend. May be provided as:

        - `list` : Values that should be mapped to the ``color_scale``.
                   If ``values`` is a `list`, ``color_scale`` must be provided.
        - `dict` : Provide the color for each value, keyed by the values.
                   If ``values`` is a `dict`, ``color_scale`` is ignored.
    
    title : `str`
        Legend title.
    color_scale : `str`
        The color scale to map values to the legend.
        Must be one of plotly supported color scales.
        Ignored if ``values`` is a `dict`.

    Return
    ------
    fig : `go.Figure`
        The Plotly legend figure.
    """
    if isinstance(values, list):
        if color_scale.endswith('_r'):
            color_cycle = pc.qualitative.__dict__[color_scale.split('_r')[0]][::-1]
        else:
            color_cycle = pc.qualitative.__dict__[color_scale.split('_r')[0]]
            
        legend_map = {val: c for val, c in zip(sorted(set(values)), itertools.cycle(color_cycle))}
    if isinstance(values, dict):
        legend_map = values
    else:
        raise ValueError("Unrecognized type, values must be a list or a dict.")

    traces = []
    for label, cc in legend_map.items():
        traces.append(go.Scatter(
            x=[None],  # Empty data
            y=[None],  # Empty data
            mode='markers',
            marker=dict(color=cc, symbol='square'),
            # visible='legendonly',
            name=label,
        ))
        traces.append(go.Heatmap(z=np.zeros_like(np.array([[0, 0.]])), # ensure there are enough points to cover the whole image
                                 colorscale="Picnic_r",  # any colorscale that has white at 0
                                 showscale=False,
                                 zmid=0))
    layout = go.Layout(
        showlegend=True,
        legend=dict(
            title=dict(text=title, font=dict(size=12, family='Arial')),
            x=0,
            y=1,
            traceorder='normal',
            font=dict(
                family='Arial', # 'sans-serif',
                size=12,
                color='#000'
            ),
            bgcolor='white', # '#E2E2E2',
            bordercolor='#FFFFFF',                
            borderwidth=2,
            # x=1,
            # orientation="h",
            # itemwidth=0,
            itemsizing='constant',
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(width=max(120, 10*len(max([str(k) for k in legend_map.keys()], key=len))),  # 200,# 100,
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
        - numpy.array : expected to be the same length as the number of nodes in ``G``
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

    if node_cmap in (set(SEQUENTIAL_COLORMAP_OPTIONS) | set(DIVERGING_COLORMAP_OPTIONS)):
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in set(node_colors)):
            n_map = {k: ix for ix, k in enumerate(sorted(set(node_colors)))}
            node_colors = [n_map[k] for k in node_colors]                    

        vmin, vmax = min(node_colors), max(node_colors)
        if (node_cmap in DIVERGING_COLORMAP_OPTIONS) and (vmin<0) and (vmax>0):
            vmag = max(np.abs(vmin), vmax)
            vmin, vmax = -vmag, vmag
            
        # norm = colors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)    
        # cmap = mpl_cm.get_cmap(node_cmap)

        if node_cmap == 'nipy_spectral':
            node_color_map = {node: pc.sample_colorscale(nipy_spectral_pl, norm(val))[0] for node, val in zip(G.nodes(), node_colors)}
        elif node_cmap == 'nipy_spectral_r':
            node_color_map = {node: pc.sample_colorscale(nipy_spectral_r_pl, norm(val))[0] for node, val in zip(G.nodes(), node_colors)}
        else:
            # node_color_map = {node: colors.rgb2hex(cmap(norm(val))) for node, val in zip(G.nodes(), node_colors)}
            node_color_map = {node: pc.sample_colorscale(node_cmap, norm(val))[0] for node, val in zip(G.nodes(), node_colors)}
        # cbar = create_colorbar(node_colors, 'nodes', color_scale=node_cmap)
        cbar = create_colorbar([vmin, vmax], 'Nodes', color_scale=node_cmap)

    else:  # node_cmap in DISCRETE_COLORMAP_OPTIONS
        if node_cmap.endswith('_r'):
            color_cycle = pc.qualitative.__dict__[node_cmap.split('_r')[0]][::-1]
        else:
            color_cycle = pc.qualitative.__dict__[node_cmap.split('_r')[0]]
            
        legend_map = {val: c for val, c in zip(sorted(set(node_colors)), itertools.cycle(color_cycle))}

        node_color_map = {node: legend_map[val] for node, val in zip(G.nodes(), node_colors)}
        cbar = create_legend(legend_map, 'Nodes')
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

    if edge_cmap in (set(SEQUENTIAL_COLORMAP_OPTIONS) | set(DIVERGING_COLORMAP_OPTIONS)):
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in set(edge_colors)):
            e_map = {k: ix for ix, k in enumerate(sorted(set(edge_colors)))}
            edge_colors = [e_map[k] for k in edge_colors]

        vmin, vmax = min(edge_colors), max(edge_colors)
        if (edge_cmap in DIVERGING_COLORMAP_OPTIONS) and (vmin<0) and (vmax>0):
            vmag = max(np.abs(vmin), vmax)
            vmin, vmax = -vmag, vmag
            
        # norm = colors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)    
        # cmap = mpl_cm.get_cmap(edge_cmap)

        if edge_cmap == 'nipy_spectral':
            edge_color_map = {edge: pc.sample_colorscale(nipy_spectral_pl, norm(val))[0] for edge, val in zip(G.edges(), edge_colors)}
        elif edge_cmap == 'nipy_spectral_r':
            edge_color_map = {edge: pc.sample_colorscale(nipy_spectral_r_pl, norm(val))[0] for edge, val in zip(G.edges(), edge_colors)}
        else:
            # edge_color_map = {edge: colors.rgb2hex(cmap(norm(val))) for edge, val in zip(G.edges(), edge_colors)}
            edge_color_map = {edge: pc.sample_colorscale(edge_cmap, norm(val))[0] for edge, val in zip(G.edges(), edge_colors)}
            # edge_color_map = {edge: colors.rgb2hex(cmap(norm(G.edges[edge][edge_color_attr]))) for edge in G.edges()}
        # cbar = create_colorbar(edge_colors, 'edges', color_scale=edge_cmap)
        cbar = create_colorbar([vmin, vmax], 'Edges', color_scale=edge_cmap)
    else:  # edge_cmap in DISCRETE_COLORMAP_OPTIONS
        if edge_cmap.endswith('_r'):
            color_cycle = pc.qualitative.__dict__[edge_cmap.split('_r')[0]][::-1]
        else:
            color_cycle = pc.qualitative.__dict__[edge_cmap.split('_r')[0]]

        legend_map = {val: c for val, c in zip(sorted(set(edge_colors)), itertools.cycle(color_cycle))}

        edge_color_map = {edge: legend_map[val] for edge, val in zip(G.edges(), edge_colors)}
        cbar = create_legend(legend_map, 'Edges')
        
    return edge_color_map, cbar


def nx_to_cytoscape(G, pos=None,
                    node_color_attr=None, D=None, node_cmap='hsv', default_node_color='#888', 
                    edge_color_attr=None, edge_cmap='hsv', positions_records=None, return_cbar=False):
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
        - numpy.array : expected to be the same length as the number of nodes in ``G``
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
        edge_color_map = {edge: '#999' for edge in G.edges()}  # Default color
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


# def renderer(keeper, G, distance_key):
def renderer(keeper, pose_key, distance_key):
    """ Construct the interactive POSE visualization for rendering in a JupyterLab notebook

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    G : `networkx.Graph` # here
        The network (intended to be the POSE network). # here
    pose_key : `str`
        The key to reference the POSE stored in ``keeper.graphs``.
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
    # global positions_records, D, G
    
    positions_records = {}
    D = keeper.distances[distance_key].data

    # POSE graphs
    graph_opts = [k for k in keeper.graphs.graphs.keys() if 'POSE' in k]
    
    if pose_key not in graph_opts:
        raise ValueError(f"Unrecognized value for pose_key, must be one of {graph_opts} out of {list(keeper.graphs.graphs.keys())}.")
    graph_val = pose_key

    G = keeper.graphs[pose_key]
    

    # stylesheet with the .dbc class from dash-bootstrap-templates library                                                                                    
    # dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    
    # Initialize the app
    # app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
    app = JupyterDash(__name__, suppress_callback_exceptions=True,
                      # external_stylesheets=[dbc.themes.FLATLY, dbc_css],
                      )

    # Define the layout of the app

    # add new for exit button
    # app.layout = html.Div(children=[# add new add Div
    #     html.Button('X', id='close-button', style={'float': 'right'}),
    #     dcc.ConfirmDialog(
    #         id='confirm',
    #         message='Are you sure you want to close the app?',
    #     ),
    #     html.Div(id='content', children=["shut down app?"]),
    #     # end new add
    app.layout = html.Div(style=styles['container'],                          
                          children=[
                              html.Div(className='control-panel-container',
                                       style=styles['control-panel-container'], children=[
                                           html.H2("NetFlow POSE", style=styles['h4']),
                                           html.H3("Settings", style=styles['h4']),
                                           html.Div([
                                               dcc.Store(id='container-dimensions'),
                                               # html.Div(children=[
                                               # html.H4("POSE", style=styles['h4']),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('POSE', style=styles['fieldset-legend']),
                                                   html.Label("POSE:", style=styles['label']),
                                                   dcc.Dropdown(
                                                       className='custom-dropdown',
                                                       id='g-pose-dropdown',
                                                       options=[{'label': ky,
                                                                 'value': ky} for ky in graph_opts],
                                                       value=pose_key,
                                                   ),
                                                   html.Label("Distance:", style=styles['label']),
                                                   dcc.Dropdown(
                                                       className='custom-dropdown',
                                                       id='pose-distance-dropdown',
                                                       options=[{'label': ky,
                                                                 'value': ky} for ky in [dd.label for dd in keeper.distances]],
                                                       value=distance_key,
                                                   ),
                                                   html.Button('Select POSE', id='pose-button', n_clicks=0),
                                                   dbc.Switch(id='pose-switch', label=' restrict to ordering',
                                                              value=False,
                                                              class_name='custom-switch-container', # # The class of the container (div)
                                                              input_class_name='custom-control-input', # The class of the <input> checkbox element.
                                                              input_style={}, #  The style of the <input> checkbox element.
                                                              label_class_name='custom-control-label', # CSS classes to apply to the <label> element for each item.
                                                              label_style={}, # Inline style arguments to apply to the <label> element for each item
                                                              ),
                                               ]),
                                               # html.Div(children=[
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Graph Layout', style=styles['fieldset-legend']),
                                                   # html.H4("Graph Layout", style=styles['h4']),
                                                   html.Label("Layout:", style=styles['label']),
                                                   dcc.Dropdown(
                                                       id='layout-dropdown',
                                                       options=[{'label': layout,
                                                                 'value': layout} for layout in ['kamada_kawai', 'forceatlas2', 'spring',
                                                                                                 'circular', # 'cose',
                                                                                                 'shell', 'grid',
                                                                                                 'breadthfirst', 'sin',
                                                                                                 'wavy_spiral']],
                                                       value='kamada_kawai',
                                                   ),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Highlight Node', style=styles['fieldset-legend']),
                                                   # html.H4("Highlight Node", style=styles['h4']),
                                                   html.Label("Node label:", style=styles['label']),
                                                   dcc.Input(id='node-label', type='text', value=''),
                                                   html.Button('Highlight Node', id='highlight-button', n_clicks=0),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Node Size', style=styles['fieldset-legend']),
                                                   # html.H4("Node Size", style=styles['h4']),
                                                   dcc.Slider(id='node-size-slider', min=0.01, max=16, step=0.01, value=4,
                                                              marks={0.01: 'smaller', 16: 'larger'},
                                                              tooltip={"placement": "top", "always_visible": False},
                                                              className='my-slider',
                                                              ),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Node Opacity', style=styles['fieldset-legend']),
                                                   # html.H4("Node Opacity", style=styles['h4']),
                                                   dcc.Slider(id='node-opacity-slider', min=0.0, max=1.0, step=0.01, value=1.,
                                                              marks={0: 'transparent', 1: 'opaque'},
                                                              tooltip={"placement": "top", "always_visible": False},
                                                              className='my-slider',
                                                              ),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Set Node Color', style=styles['fieldset-legend']),
                                                   # html.H4("Set Node Color", style=styles['h4']),
                                                   # # html.Label(html.B("Set Fixed Color")),
                                                   html.Div(children=[
                                                       html.Label("Set Fixed Color", style=styles['label_b']),
                                                       html.Label("Color:", style=styles['label']),
                                                       dcc.Input(id='node-fixed-color', type='text', value=''),
                                                       html.Button('Color nodes', id='fixed-color-button', n_clicks=0),
                                                   ]),
                                                   # html.Label(html.B("Color By Node Attribute")),
                                                   html.Div(children=[
                                                       html.Label("Color By Node Attribute", style=styles['label_b']),
                                                       html.Label("Node attribute:", style=styles['label']),
                                                       dcc.Dropdown(
                                                           id='node-attribute-dropdown',
                                                           # BBB options=[{'label': 'None',
                                                           # BBB          'value': 'None'}] + [{'label': attr,
                                                           # BBB                             'value': attr} for attr in G.nodes(data=True)[0].keys()],
                                                           options=[{'label': attr,
                                                                     'value': attr} for attr in G.nodes(data=True)[0].keys()],
                                                           value=None, # BBB 'None',
                                                       ),
                                                   ]),
                                                   # html.Label(html.B("Color by Data Feature")),
                                                   html.Label("Color by Data Feature", style=styles['label_b']),
                                                   html.Label("Dataset:", style=styles['label']),
                                                   dcc.Dropdown(
                                                       id='keeper-data-dropdown',
                                                       # BBB options=[{'label': 'None',
                                                       # BBB          'value': 'None'}] + [{'label': dataset.label,
                                                       # BBB                                'value': dataset.label} for dataset in keeper.data],
                                                       options=[{'label': dataset.label,
                                                                 'value': dataset.label} for dataset in keeper.data],
                                                       value=None, # BBB 'None', # 'branch'
                                                   ),
                                                   html.Label("Feature:", style=styles['label']),
                                                   dcc.Dropdown(id='feature-label', options=[], value=None), # 'None'), AAA 
                                                   # html.Button('Color nodes', id='node-color-button', n_clicks=0), # here
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Colormap for Nodes', style=styles['fieldset-legend']),
                                                   # html.H4("Colormap for Nodes", style=styles['h4']),
                                                   html.Div([
                                                       dcc.RadioItems(id='node-colormap-type',
                                                                      options=[
                                                                          {'label': 'Sequential', 'value': 'sequential'},
                                                                          {'label': 'Diverging', 'value': 'diverging'},
                                                                          {'label': 'Discrete', 'value': 'discrete'},
                                                                      ],
                                                                      value='sequential',
                                                                      labelStyle={'display': 'block', 'margin-right': '10px'}, # 'display': 'indline-block'
                                                                      inputClassName='custom-radioinput',
                                                                      className='custom-radioitems',
                                                                      labelClassName='custom-radiolabel',
                                                                      ),
                                                       dcc.Dropdown(
                                                           id='node-colormap-dropdown',
                                                           options=[{'label': cmap,
                                                                     'value': cmap} for cmap in SEQUENTIAL_COLORMAP_OPTIONS], # COLORMAP_OPTIONS], # mpl.colormaps()], # ['YlGnBu', 'viridis', 'cividis', 'jet', 'nipy_spectral', 'gist_ncar']],
                                                           value='nipy_spectral', # 'Turbo_r', # 'YlGnBu',
                                                       ),
                                                   ]),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Edge Width', style=styles['fieldset-legend']),
                                                   # html.H4("Edge Width", style=styles['h4']),
                                                   dcc.Slider(id='edge-width-slider', min=0.001, max=14, step=0.001, value=0.406, # 0.1,
                                                              marks={0.001: 'thinner', 14: 'wider'},
                                                              tooltip={"placement": "top", "always_visible": False},
                                                              className='my-slider',
                                                              ),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Set Edge Color', style=styles['fieldset-legend']),
                                                   # html.H4("Set Edge Color", style=styles['h4']),
                                                   # # html.Label(html.B("Color by Edge Attribute")),
                                                   html.Label("Color by Edge Attribute", style=styles['label_b']),
                                                   html.Label("Edge attribute:", style=styles['label']),
                                                   dcc.Dropdown(
                                                       id='edge-attribute-dropdown',
                                                       # BBB options=[{'label': 'None',
                                                       # BBB          'value': 'None'}] + [{'label': attr,
                                                       # BBB                                'value': attr} for attr in list(G.edges(data=True))[0][-1].keys()],
                                                       options=[{'label': attr,
                                                                 'value': attr} for attr in list(G.edges(data=True))[0][-1].keys()],
                                                       value=None, # BBB 'None', 
                                                   ),
                                               ]),
                                               html.Fieldset(style=styles['fieldset-panel'], children=[
                                                   html.Legend('Colormap for Edges', style=styles['fieldset-legend']),
                                                   # html.H4("Colormap for Edges", style=styles['h4']),
                                                   html.Div([
                                                       dcc.RadioItems(id='edge-colormap-type',
                                                                      options=[
                                                                          {'label': 'Sequential', 'value': 'sequential'},
                                                                          {'label': 'Diverging', 'value': 'diverging'},
                                                                          {'label': 'Discrete', 'value': 'discrete'},
                                                                      ],
                                                                      value='sequential',
                                                                      labelStyle={'display': 'block', 'margin-right': '10px'},
                                                                      inputClassName='custom-radioinput',
                                                                      className='custom-radioitems',
                                                                      labelClassName='custom-radiolabel',
                                                                      ),
                                                       dcc.Dropdown(
                                                           id='edge-colormap-dropdown',
                                                           options=[{'label': cmap,
                                                                     'value': cmap} for cmap in SEQUENTIAL_COLORMAP_OPTIONS], 
                                                           value='hot_r',
                                                       ),
                                                   ]),
                                               ]),
                                               # html.Fieldset(style=styles['fieldset-panel'], children=[
                                               #     html.Legend('Filter edges', style=styles['fieldset-legend']),
                                               #     html.Label("Filter by Edge Attribute", style=styles['label_b']),
                                               #     html.Label("Edge attribute:", style=styles['label']),
                                               #     dcc.Dropdown(
                                               #         id='edge-filter-attribute-dropdown',
                                               #         # BBB options=[{'label': 'None',
                                               #         # BBB          'value': 'None'}] + [{'label': attr,
                                               #         # BBB                                'value': attr} for attr in list(G.edges(data=True))[0][-1].keys()],
                                               #         options=[{'label': attr,
                                               #                   'value': attr} for attr in list(G.edges(data=True))[0][-1].keys() if \
                                               #                  all(isinstance(k, (int, float)) for k in nx.get_edge_attributes(G, attr).values())],
                                               #         value=None, # BBB 'None', 
                                               #     ),
                                               #     html.Label("Range:", style=styles['label']),
                                               #     dcc.RangeSlider(id='edge-filter-slider',
                                               #                     min=0, max=1, step=1e-4, value=0,
                                               #                     allowCross=False,
                                               #                     tooltip={"placement": "top", "always_visible": False},
                                               #                     className='my-slider',
                                               #                     ),
                                               # ]),
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
                                                                               'flex': 0.4,
                                                                               'overflow-y': 'scroll',
                                                                               'overflow-x': 'scroll', # 'hidden',
                                                                               'paddingBottom': '2px',
                                                                               # 'white-space': 'nowrap',
                                                                               # 'align-items': 'center', 'justify-content': 'center',
                                                                               },
                                               children=[],
                                               ),
                                      html.Div(id='edge-colorbars-div', style={'height': '100%', 'display': 'flex',
                                                                               'flex': 0.4,
                                                                               'overflow-y': 'scroll',
                                                                               'overflow-x': 'scroll', # 'hidden',
                                                                               'paddingBottom': '2px',
                                                                               },
                                               children=[],
                                               ),
                                  ],
                                           ),
                                  dcc.Store(id='store-selectedNodeData'),
                                  dcc.Store(id='store-tapNodeData'),
                                  cyto.Cytoscape(
                                      id='network-graph',
                                      elements=nx_to_cytoscape(G),
                                      style=styles['cytoscape'],
                                      layout={'name': 'preset', "fit": False},  # {'name': 'cose'},
                                      # selectable=True,
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
                                                  'width': 4, # 8, # 'data(size)',
                                                  'height': 4, # 8, # 'data(size)',
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
                                                  'line-color': '#999', # '#222', # 'darkgray', # 'data(color)'
                                              }
                                          },
                                          # Class selectors
                                          {
                                              'selector': '.highlight_by_label',
                                              'style': {
                                                  'border-width': '1.5px', # 20,
                                                  'border-color': 'red', # 'black',
                                                  # 'outline-width': 20,
                                                  # 'outline-color': 'black',
                                              },
                                          },                                          
                                          {
                                              'selector': '.highlight_box_select',
                                              'style': {
                                                  'border-width': '1.5px',
                                                  'border-color': 'gray', # 'black',
                                                  # 'z-index': 1999,
                                              },
                                          },
                                      ],
                                      # responsive=True,
                                  ),
                                  
                              ]),
                              # begin box over
                              html.Div(style=styles['stat-panel-container'], children=[
                                  html.H3("Statistical Testing", style=styles['h4']),
                                  # html.H4("Options", style=styles['h3']),
                                  # html.Div(style=styles['stat-panel'], # {'display': 'flex', 'flex': 0.3, 'height': '100%'},
                                  html.Fieldset(style=styles['fieldset-panel-stat'], # styles['stat-panel'],
                                                children=[
                                                    html.Legend('Options', style=styles['fieldset-legend']),
                                                    html.Div([
                                                        html.Label("Dataset:", style=styles['label']),
                                                        dcc.Dropdown(
                                                            id='stat-keeper-data-dropdown',
                                                            # BBB options=[{'label': 'None',
                                                            # BBB          'value': 'None'}] + [{'label': dataset.label,
                                                            # BBB                                'value': dataset.label} for dataset in keeper.data],
                                                            options=[{'label': dataset.label,
                                                                      'value': dataset.label} for dataset in keeper.data],
                                                            value=None), # BBB 'None'),
                                                    ]),
                                                    html.Div([
                                                        html.Label("Statistical test:", style=styles['label_tspace']),
                                                        dcc.Dropdown(id='stat-test-dropdown',
                                                                     options=[{'label': 'Mann Whitney U Test', 'value': 'MWU'},
                                                                              {'label': 'T-test', 'value': 't-test'}], value='MWU'),
                                                        html.Div(children=["alpha: ",
                                                                           dcc.Input(id='alpha-label', type='number', min=0., max=1., value=0.05, step=0.001)]),
                                                    ]),
                                                    html.Div([
                                                        html.Label("Multiple test correction:", style=styles['label_tspace']),
                                                        dcc.Dropdown(id='correction-drop-down',
                                                                     options=[{'label': k,
                                                                               'value': kk} for k,kk in zip(['Bonferroni', 'Sidak', 'Holm-Sidak', 'Holm',
                                                                                                             'Simes-Hochber', 'Hommel', 'FDR-BH',
                                                                                                             'FDR-BY', 'FDR-TSBH', 'FDR-TSBKY'],
                                                                                                            ['bonferroni', 'sidak', 'holm-sidak', 'holm',
                                                                                                             'simes-hochber', 'hommel', 'fdr_bh',
                                                                                                             'fdr_by', 'fdr_tsbh', 'fdr_tsbky'])],
                                                                     value='fdr_bh'),
                                                    ]),
                                                    # HERE SURV!
                                                    html.Div([
                                                        html.Label("Log-rank survival:", style=styles['label_tspace']),
                                                        html.Label("Dataset:", style=styles['label']),
                                                        dcc.Dropdown(
                                                            id='surv-keeper-data-dropdown', # 'keeper-data-dropdown'
                                                            options=[{'label': dataset.label,
                                                                      'value': dataset.label} for dataset in keeper.data],
                                                            value=None),
                                                        html.Label("Time:", style=styles['label']),
                                                        dcc.Dropdown(id='time-attr', options=[], value=None), # 'feature-label'
                                                        html.Label("Event status:", style=styles['label']),
                                                        dcc.Dropdown(id='event-attr', options=[], value=None), # 'feature-label'
                                                    ]),
                                                ]),
                                  # html.H4("Results", style=styles['h3']),
                                  # html.Fieldset(style={'width': '100%'}, # styles['stat-panel'],
                                  #          children=[
                                  #              html.Legend('Results', style=styles['h3']),
                                  html.Div(id="box-output", children=[],
                                           # style={'display': 'flex', 'flex': 0.7},
                                           style=styles['stat-table'],
                                           ),
                                  html.Button("Download Data", id="btn-download", n_clicks=0, style={'display': 'none'}),
                                  dcc.Download(id="download-data"),
                                  html.Pre(id="box-selected-labels", style=styles['pre_box_labels'],
                                           children=['Selected nodes'],
                                           ),
                                  html.Div(children=[
                                      dbc.Switch(id='box-selected-switch', label='Highlight selected nodes',
                                                 value=True,
                                                 class_name='custom-switch-container', # # The class of the container (div)
                                                 input_class_name='custom-control-input', # The class of the <input> checkbox element.
                                                 input_style={}, #  The style of the <input> checkbox element.
                                                 label_class_name='custom-control-label', # CSS classes to apply to the <label> element for each item.
                                                 label_style={}, # Inline style arguments to apply to the <label> element for each item
                                                 ),
                                  ]),
                                  # ]),
                              ]),
                              # end box here
                          ])
    # ]) # add new for exit button


    # # add new for exit button
    # # Show confirm dialog on button click
    # @app.callback(
    #     Output('confirm', 'displayed'),
    #     Input('close-button', 'n_clicks')
    # )
    # def display_confirm(n_clicks):
    #     print(f"close button clicked - n_clicks = {n_clicks}")
    #     if n_clicks:
    #         return True
    #     return False

    # # Handle confirm dialog response
    # @app.callback(
    #     Output('content', 'children'),
    #     Input('confirm', 'submit_n_clicks'),
    #     prevent_initial_call=True
    # )
    # def close_app(submit_n_clicks):
    #     if submit_n_clicks:
    #         shutdown_server()
    #         return ['Server shutting down...']

    #     return ['App content goes here']
    # # end add new for exit button


    
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

    # COMBINES NEXT TWO CALLBACKS
    # Define the callback to update the dropdown node colormap options based on the selected radio item
    @app.callback(
        Output('node-colormap-dropdown', 'options'),
        Output('node-colormap-dropdown', 'value', # allow_duplicate=True,  # add here
               ),
        Input('node-colormap-type', 'value'),
        # Input('node-colormap-dropdown', 'options'),
    )
    def set_node_colormap_options(cmap_type):
        if cmap_type == 'sequential':
            colormap_options = [{'label': name, 'value': name} for name in SEQUENTIAL_COLORMAP_OPTIONS]
            # cmap_value = SEQUENTIAL_COLORMAP_OPTIONS[0]
            cmap_value = 'nipy_spectral' # 'Turbo_r' # 'Blackbody'
        elif cmap_type == 'diverging':
            colormap_options = [{'label': name, 'value': name} for name in DIVERGING_COLORMAP_OPTIONS]
            # cmap_value = DIVERGING_COLORMAP_OPTIONS[0]
            cmap_value = 'balance_r'
        else:  # discrete
            colormap_options = [{'label': name, 'value': name} for name in DISCRETE_COLORMAP_OPTIONS]
            cmap_value = 'Bold' # DISCRETE_COLORMAP_OPTIONS[0]

        return colormap_options, cmap_value
    # Define the callback to update the dropdown node colormap options based on the selected radio item
    # @app.callback(
    #     Output('node-colormap-dropdown', 'options'),        
    #     Input('node-colormap-type', 'value'),
    # )
    # def set_node_colormap_options(cmap_type):
    #     if cmap_type == 'sequential':
    #         colormap_options = [{'label': name, 'value': name} for name in SEQUENTIAL_COLORMAP_OPTIONS]
    #     elif cmap_type == 'diverging':
    #         colormap_options = [{'label': name, 'value': name} for name in DIVERGING_COLORMAP_OPTIONS]
    #     else:  # discrete
    #         colormap_options = [{'label': name, 'value': name} for name in DISCRETE_COLORMAP_OPTIONS]
    #     return colormap_options


    # # Define the callback to update the default value of the node colormap option based on the available options
    # @app.callback(
    #     Output('node-colormap-dropdown', 'value', # allow_duplicate=True,  # add here
    #            ),
    #     Input('node-colormap-dropdown', 'options'),
    #     # prevent_initial_call=True # add here
    # )
    # def set_default_node_colormap_options(cmap_options):
    #     if cmap_options:
    #         return cmap_options[0]['value']


    # COMBINES NEXT TWO CALLBACKS
    # Define the callback to update the dropdown edge colormap options based on the selected radio item
    @app.callback(
        Output('edge-colormap-dropdown', 'options'),
        Output('edge-colormap-dropdown', 'value', # allow_duplicate=True,  # add here
               ),
        Input('edge-colormap-type', 'value'),
    )
    def set_edge_colormap_options(cmap_type):
        if cmap_type == 'sequential':
            colormap_options = [{'label': name, 'value': name} for name in SEQUENTIAL_COLORMAP_OPTIONS]
            # cmap_value = SEQUENTIAL_COLORMAP_OPTIONS[0]
            cmap_value = 'nipy_spectral' # 'Turbo_r'
        elif cmap_type == 'diverging':
            colormap_options = [{'label': name, 'value': name} for name in DIVERGING_COLORMAP_OPTIONS]
            # cmap_value = DIVERGING_COLORMAP_OPTIONS[0]
            cmap_value = 'balance_r'
        else:  # discrete
            colormap_options = [{'label': name, 'value': name} for name in DISCRETE_COLORMAP_OPTIONS]
            cmap_value = 'Bold' # DISCRETE_COLORMAP_OPTIONS[0]

        return colormap_options, cmap_value
    # # Define the callback to update the dropdown edge colormap options based on the selected radio item
    # @app.callback(
    #     Output('edge-colormap-dropdown', 'options'),
    #     Input('edge-colormap-type', 'value'),
    # )
    # def set_edge_colormap_options(cmap_type):
    #     if cmap_type == 'sequential':
    #         colormap_options = [{'label': name, 'value': name} for name in SEQUENTIAL_COLORMAP_OPTIONS]
    #     elif cmap_type == 'diverging':
    #         colormap_options = [{'label': name, 'value': name} for name in DIVERGING_COLORMAP_OPTIONS]
    #     else:  # discrete
    #         colormap_options = [{'label': name, 'value': name} for name in DISCRETE_COLORMAP_OPTIONS]
    #     return colormap_options


    # # Define the callback to update the default value of the edge colormap option based on the available options
    # @app.callback(
    #     Output('edge-colormap-dropdown', 'value'),
    #     Input('edge-colormap-dropdown', 'options'),
    # )
    # def set_default_edge_colormap_options(cmap_options):
    #     if cmap_options:
    #         return cmap_options[0]['value']


    # @app.callback(
    #     Output('store-tapNodeData', 'data'),
    #     Output('store-selectedNodeData', 'data'),
    #     Input('network-graph', 'tapNodeData'),
    #     Input('network-graph', 'selectedNodeData'),
    #     State('store-tapNodeData', 'data'),
    #     State('store-selectedNodeData', 'data')
    # )
    # def manage_interactive_node_data(tap_data, selected_data, stored_tap_data, stored_selected_data):
    #     triggered_input = callback_context.triggered[0]['prop_id'].split('.')[0]
    #     triggered_attr = callback_context.triggered[0]['prop_id'].split('.')[1]

    #     print(f"manager input 0: \n >>> tap_data = {tap_data} \n >>> selected_data = {selected_data} \n >>> stored_tap_data = {stored_tap_data} \n >>> stored_selected_data = {stored_selected_data}")
    #     if triggered_input == 'network-graph':
    #         if triggered_attr == 'tapNodeData':
    #             # update tapNodeData
    #             # print('updating tapNodeData')
    #             stored_tap_data = tap_data
    #         elif triggered_attr == 'selectedNodeData':
    #             # update selectedNodeData
    #             stored_selected_data = selected_data

    #     print(f"manager output 1: \n >>> stored_tap_data = {stored_tap_data} \n >>> stored_selected_data = {stored_selected_data}")
    #     return stored_tap_data, stored_selected_data
                
        

    # ATTEMPTING TO USE dcc.STORE HERE
    # @app.callback(
    #     Output('box-output', 'children'),
    #     Output('btn-download', 'style'),
    #     Output("box-selected-labels", 'children'),
    #     # Output('selected-node-data', 'data'),
    #     # Input('selected-node-data', 'data'), # Input('network-graph', 'boxSelectedData'), # 'selectedNodeData'),
    #     # Input('network-graph', 'selectedNodeData'),
    #     Input('store-selectedNodeData', 'data'),
    #     Input('stat-keeper-data-dropdown', 'value'),
    #     Input('stat-test-dropdown', 'value'),
    #     Input('alpha-label', 'value'),
    #     Input('correction-drop-down', 'value'),
    # )
    # ORIGINALLY HERE
    # @app.callback(
    #     Output('box-output', 'children'),
    #     Output('btn-download', 'style'),
    #     Output("box-selected-labels", 'children'),
    #     # Output('selected-node-data', 'data'),
    #     # Input('selected-node-data', 'data'), # Input('network-graph', 'boxSelectedData'), # 'selectedNodeData'),
    #     Input('network-graph', 'selectedNodeData'),
    #     Input('stat-keeper-data-dropdown', 'value'),
    #     Input('stat-test-dropdown', 'value'),
    #     Input('alpha-label', 'value'),
    #     Input('correction-drop-down', 'value'),
    # )    
    def display_selected_nodes(data, keeper_data_label, test, alpha, method,
                               surv_data_label, surv_t_label, surv_e_label):
                              # print(f"display_selected_nodes : TRIGGERED INPUT = {callback_context.triggered[0]['prop_id']}")
        # print(f"display selected nodes = is box data none: {data is None} - data = {data}")

        if not data:
            return "", {'display': 'none'}, 'Selected nodes' # [] # "No nodes selected."

        selected_obs = set([node['name'] for node in data])
        unselected_obs = list(set(keeper.observation_labels) - selected_obs)
        selected_obs = list(selected_obs)

        label_output = f"Selected nodes' labels (n={len(selected_obs)}): " + ", ".join(selected_obs)

        if (surv_t_label is not None) and (surv_e_label is not None):
            if (len(selected_obs) > 1) and (len(unselected_obs) > 1):
                surv_df = keeper.data[surv_data_label].to_frame()
                surv_df = surv_df.loc[[surv_t_label, surv_e_label]].T
                lr = logrank_test(surv_df.loc[selected_obs, surv_t_label], surv_df.loc[unselected_obs, surv_t_label],
                                  event_observed_A=surv_df.loc[selected_obs, surv_e_label],
                                  event_observed_B=surv_df.loc[unselected_obs, surv_e_label])
                label_output = "Selected nodes' labels " + f"(n={len(selected_obs)}; log-rank p = {lr.p_value:2.2e}): " + ", ".join(selected_obs)

        
        if (not data) or (keeper_data_label is None): # BBB == 'None') :
            # return "", {'display': 'none'}, 'Selected nodes' # [] # "No nodes selected."
            return "", {'display': 'none'}, label_output # mlti
        
        df = keeper.data[keeper_data_label].to_frame()
        # select columns that are floats or integers:
        # df = df[df.select_dtypes(include=['float', 'int']).columns]
        # df = df.loc[df.T.select_dtypes(include=['float', 'int']).columns]
        rows = [rw for rw, vv in df.iterrows() if all([isinstance(k, (float, int)) for k in vv])]
        if len(rows) == 0:
            return "", {'display': 'none'}, label_output 
        
        df = df.loc[rows]
        df1 = df[selected_obs]
        df2 = df[unselected_obs]
        df1_mean = df1.mean(axis=1)
        df2_mean = df2.mean(axis=1)
        HL = (df1_mean > df2_mean).replace({True: 'H', False: 'L'})
        HL[(df1_mean - df2_mean).abs() < 1e-6] = 'E' # set to approx equal
        HL.name = 'high (H) / low (L)'        
        record = stat_test(df1, df2, test=test, alpha=alpha, method=method)
        record.insert(0, HL.name, HL)
        record = record.loc[record['p-value'] <= alpha]
        if record.shape[0] == 0:
            return "", {'display': 'none'}, label_output
        
        record = record.sort_values(by='p-value')
        record.index.name = 'feature'
        record = record.reset_index()
        dash_record = dash_table.DataTable(
            id='stat-datatable',
            columns=[{"name": i, "id": i, "type": "numeric",
                      "format": dash_table.Format.Format(precision=3)} if \
                     pd.api.types.is_numeric_dtype(record[i]) else \
                     {"name": i, "id": i} for i in record.columns],
            data=record.to_dict('records'),
            style_table={'height': '100%', # '200px',
                         # 'width': '100%', # '24%', # MY NEW
                         'overflowY': 'scroll', # 'auto',
                         'overflowX': 'scroll'},
            style_cell={'textAlign': 'left',
                        'padding': '2px',
                        # 'border': '1px solid grey',
                        },
            style_header={
                # 'backgroundColor': 'white',
                'backgroundColor': 'rgb(210, 210, 210)',
                'fontWeight': 'bold',
                'color': 'black',
                'border': '1px solid black',
            },
            page_size=5,  # Adjust as needed for your display
            style_data={
                'color': 'black',
                'backgroundColor': 'white',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': col},
                    'type': 'numeric',
                    'format': {'specifier': '.3f'}
                } for col in record.select_dtypes(include=['float', 'int']).columns
            ] + [ # add striped rows
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                },
            ]
        )
        # return f"Selected nodes: {', '.join([node['name'] for node in data])}"

        # mlti = [dash_record,
        #         html.Button("Download Data", id="btn-download", n_clicks=0),
        #         dcc.Download(id="download-data")]
        # label_output = [html.Pre(", ".join(selected_obs),
        #                          style=styles['pre_box_labels'])]
        
        return dash_record, {'display': 'block'}, label_output # mlti
        

    @app.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),        
        State('stat-datatable', 'data'),
        State('stat-keeper-data-dropdown', 'value'),
        State('stat-test-dropdown', 'value'),
        State('correction-drop-down', 'value'),
        State("box-selected-labels", 'children'),
        prevent_initial_call=True,
    )
    def download_data(n_clicks, data, data_label, stat_test, correction, obs_labels):
        df = pd.DataFrame(data)        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = "_".join(["selected_nodes", str(data_label), stat_test.replace('-', ''),
                          correction.replace('-', ''), timestamp])
        # fname = fname + ".csv"
        # return dcc.send_data_frame(df.to_csv, fname)
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='stats')

                obs_df = pd.DataFrame({'Selected observations': obs_labels.split(', ')})
                obs_df.to_excel(writer, index=False, sheet_name='observations')

            output.seek(0)
            return dcc.send_bytes(output.getvalue(), fname+".xlsx")
                                               

    # @app.callback(
    #     Output('feature-label', 'options'),
    #     Input('keeper-data-dropdown', 'value'),
    # )
    def set_feature_options(data_label):
        """ return list of feature options for given data in the keeper """
        if (data_label is None) or (data_label == 'None'):
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
        Output('network-graph', 'stylesheet'),
        Input('node-size-slider', 'value'), 
        Input('edge-width-slider', 'value'),
        Input('node-opacity-slider', 'value'),
        Input('box-selected-switch', 'value'),
    )
    def update_stylesheet(node_size, edge_width, node_opacity, box_switch):
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
                    # 'border-width': '0.5px',  # Adds a thin border around the nodes
                    # 'border-color': '#999',  # Sets the border color
                    'opacity': node_opacity,
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
                'selector': '.highlight_by_label',
                'style': {
                    'border-width': '1.5px',
                    'border-color': 'red',
                    # 'outline-width': 20,
                    # 'outline-color': 'black',
                    # 'z-index': 1001,
                },
            },
        ]
        if box_switch:
            stylesheet.append(
                {
                    'selector': '.highlight_box_select',
                    'style': {
                        'border-width': '1.5px',
                        'border-color': 'black',
                        # 'z-index': 1999,
                    },
                })

        return stylesheet
    
    
    @app.callback(        
        Output('network-graph', 'elements'), # (elements)
        # Output('network-graph', 'stylesheet'), # moved to its own callback
        # Output('network-graph', 'selectedNodeData'), # Output('network-graph', 'boxSelectedData'),
        # Output('selected-node-data', 'data'), ####################        
        Output('node-attribute-dropdown', 'value'), # (node_attr)
        Output('keeper-data-dropdown', 'value'), # (data_label)
        Output('feature-label', 'value'), # (ft_label)
        Output('feature-label', 'options'), # (fl_label_opts) 
        Output('network-graph', 'zoom'), # (current_zoom)
        Output('network-graph', 'pan'), # (current_pan)
        Output('node-colorbars-div', 'children'), # (node_cbar_vis_in)
        Output('edge-colorbars-div', 'children'), # edge_cbar_vis_in)
        # Output('network-graph', 'tapNodeData'), # (tap_node_data)
        Output('node-fixed-color', 'value'), # (node_fixed_color)
        Output('network-graph', 'mouseoverNodeData'), # (mouseover_node_data)
        # Output('store-tapNodeData', 'data'), # new (stored_tap_node_data)
        Output('store-selectedNodeData', 'data'), # new (stored_selected_node_data)
        Output('box-output', 'children'), # new (box_selected_table)
        Output('btn-download', 'style'), # new (stat_download_button_style)
        Output("box-selected-labels", 'children'), # new (box_selected_output)
        Output('time-attr', 'value'),                  # *** surv_t_label
        Output('time-attr', 'options'),                  # *** surv_t_opts
        Output('event-attr', 'value'),                 # *** surv_e_label
        Output('event-attr', 'options'),                 # *** surv_e_opts
        Output('network-graph', 'tapNodeData'),             # (tap_node_data) # ABC added
        Output('network-graph', 'selectedNodeData'),        # new (selected_node_data) xxx
        # Output('node-colormap-type', 'value'),  # 
        # Output('node-colormap-dropdown', 'value'), # add here
        Input('layout-dropdown', 'value'),                 # (layout) x
        Input('highlight-button', 'n_clicks'),             # (n_clicks)
        State('node-label', 'value'),  # (node_label)
        # Input('node-size-slider', 'value'), # moved to its own callback
        # Input('edge-width-slider', 'value'), # moved to its own callback
        Input('node-attribute-dropdown', 'value'),         # (node_attr) xxx
        Input('edge-attribute-dropdown', 'value'),         # (edge_attr)
        Input('node-colormap-dropdown', 'value'),          # (node_cmap)
        Input('edge-colormap-dropdown', 'value'),          # (edge_cmap)
        Input('network-graph', 'tapNodeData'),             # (tap_node_data) xxx
        Input('network-graph', 'selectedNodeData'),        # new (selected_node_data) xxx
        # State('store-tapNodeData', 'data'), # new (stored_tap_node_data)
        State('store-selectedNodeData', 'data'), # new (stored_selected_node_data)
        Input('stat-keeper-data-dropdown', 'value'),       # new (stat_keeper_data_label) xxx
        Input('stat-test-dropdown', 'value'),              # new (stat_test) xxx
        Input('alpha-label', 'value'),                     # new (alpha) xxx
        Input('correction-drop-down', 'value'),            # new (stat_correction) xxx
        Input('surv-keeper-data-dropdown', 'value'),  # *** surv_data_label
        Input('time-attr', 'value'),                  # *** surv_t_label
        State('time-attr', 'options'),                  # *** surv_t_opts
        Input('event-attr', 'value'),                 # *** surv_e_label
        State('event-attr', 'options'),                 # *** surv_e_opts
        State('network-graph', 'elements'), # (elements_in)
        State('container-dimensions', 'data'), # (dimensions)
        State('network-graph', 'zoom'), # (current_zoom)
        State('network-graph', 'pan'), # (current_pan)
        # Input('node-color-button', 'n_clicks'),    # here
        Input('keeper-data-dropdown', 'value'), # (data_label)
        Input('feature-label', 'value'),                   # here : State -> Input   (ft_label) xxx
        State('feature-label', 'options'), # (fl_label_opts) 
        State('node-fixed-color', 'value'),  # (node_fixed_color)
        Input('fixed-color-button', 'n_clicks'),           # (node_fixed_color_n_clicks) xxx
        Input('pose-button', 'n_clicks'),                  # (pose_n_clicks) xxx        
        State('g-pose-dropdown', 'value'), # (pose_key)
        State('pose-distance-dropdown', 'value'), # (pose_dist_key)
        Input('pose-switch', 'value'), # (pose_ordering)
        # Input('network-graph', 'selectedNodeData'), # 'boxSelectedData'), 
        # State('network-graph', 'selectedNodeData'), # add select
        State('network-graph', 'mouseoverNodeData'), # (mouseover_node_data)
        # State('network-graph', 'stylesheet'), # 2  # moved to its own callback
        State('node-colorbars-div', 'children'), # 9 (node_cbar_vis_in)
        State('edge-colorbars-div', 'children'), # 10 # (edge_cbar_vis_in)
        # State('node-colormap-type', 'value'),  #
        State('box-output', 'children'), # new (box_selected_table)
        State('btn-download', 'style'), # new (stat_download_button_style)
        State("box-selected-labels", 'children'), # new (box_selected_output)        
        # prevent_initial_call=True,
        # suppress_callback_exceptions=True,        
    )
    def update_graph(layout, n_clicks, node_label, 
                     # node_size, edge_width, # moved to its own callback
                     node_attr, edge_attr, node_cmap, edge_cmap, 
                     tap_node_data,
                     selected_node_data, # stored_tap_node_data, # new
                     stored_selected_node_data, stat_keeper_data_label, # new
                     stat_test, alpha, stat_correction, # new
                     surv_data_label,
                     surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,  # ***
                     elements_in, dimensions,
                     current_zoom, current_pan, # feature_n_clicks, # here
                     data_label, ft_label, fl_label_opts,
                     node_fixed_color, node_fixed_color_n_clicks, # node_cmap_type,
                     pose_n_clicks, pose_key, pose_dist_key, pose_ordering,
                     # box_select_node_data,
                     # select_node_data,  # add select
                     mouseover_node_data,
                     # stylesheet_in, # moved to its own callback
                     node_cbar_vis_in, edge_cbar_vis_in,
                     box_selected_table, stat_download_button_style, box_selected_output,
                     ): 
        # Determine which input triggered the callback
        triggered_input = callback_context.triggered[0]['prop_id'].split('.')[0]
        triggered_attr = callback_context.triggered[0]['prop_id'].split('.')[1]        

        
        # only update the highlighting of the box selected nodes
        # if (triggered_input == 'network-graph'):
        #     print(f"selected node data = {selected_node_data}")
        #     aa = "\n".join([f"\n****\nTriggered_input = {triggered_input}; triggered_attr = {triggered_attr}",
        #                     f" > selected_node_data type = {type(selected_node_data)}",
        #                     f" > selected_node_data len if not None = {'None' if selected_node_data is None else len(selected_node_data)}",
        #                     f" > selected_labels = {'None' if selected_node_data is None else set([node['name'] for node in selected_node_data])}",
        #                     f" > stored_selected_labels = {'None' if stored_selected_node_data is None else set([node['name'] for node in stored_selected_node_data])}",
        #                     f" > tap_node_data = {'None' if tap_node_data is None else tap_node_data['name']}",
        #                     # f" > stored_tap_node_data = {'None' if stored_tap_node_data is None else stored_tap_node_data['name']}"])
        #                     # f" > stored_tap_node_data = {'None' if stored_tap_node_data is None else stored_tap_node_data}",
        #                     ])
        #     print(aa)
        #     bb = "\n".join([f"\n input selected node data: {selected_node_data}",
        #                     f" input tap node data: {tap_node_data}",
        #                     f" input stored selected node data: {stored_selected_node_data}"])
        #     print(bb)

        if triggered_input == 'highlight-button':
            elements = elements_in
            if node_label:
                labels = set(node_label.split(', '))
            
                for element in elements['nodes']:
                    if element['data'].get('name') in labels:
                        if 'classes' in element:
                            if 'highlight_by_label' in element['classes']:
                                continue
                            else:
                                element['classes'] += ' highlight_by_label'
                        else:
                            element['classes'] = 'nodes highlight_by_label'
                    elif 'classes' in element:
                        element['classes'] = element['classes'].replace(' highlight_by_label', '').strip()
                    else:
                        continue
                    
            return (elements,
                    node_attr, data_label, ft_label, fl_label_opts,
                    current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, 
                    node_fixed_color, mouseover_node_data,
                    stored_selected_node_data, box_selected_table, stat_download_button_style,
                    box_selected_output, surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                    tap_node_data, None) # selected_node_data)
            
        if (triggered_input == 'network-graph') and (triggered_attr == 'selectedNodeData'):  # 'boxSelectedData'): #
            tap_label = '' if tap_node_data is None else tap_node_data['name']            # ABC commented line
            # stored_tap_label = '' if stored_tap_node_data is None else stored_tap_node_data['name']            # ABC commented line

            # if (len(selected_labels - stored_selected_labels) == 1) and not tap_node_data:
            #     print("*** Expected tapNodeData....")
            
            # selected_labels = set() if selected_node_data is None else set([node['name'] for node in selected_node_data])
            selected_labels = [] if selected_node_data is None else list(set([node['name'] for node in selected_node_data]))
            
            # if len(selected_labels) == 0: # ABC added line
            #     stored_selected_node_data = None # ABC added line
            #     return (elements_in, # ABC added line
            #             node_attr, data_label, ft_label, fl_label_opts, # ABC added line
            #             current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, # ABC added line
            #             node_fixed_color, mouseover_node_data, # ABC added line
            #             stored_tap_node_data, stored_selected_node_data, box_selected_table, stat_download_button_style, # ABC added line
            #             box_selected_output, tap_node_data) # ABC added line            
                        
            # ABC begin commenting
            # if tap_node_data and not stored_tap_node_data:
            #     triggered_attr = 'tapNodeData'
            # elif tap_node_data and (len(selected_labels - stored_selected_labels) == 1) and (tap_label == list(selected_labels - stored_selected_labels)[0]):
            #     triggered_attr = 'tapNodeData'
            # elif tap_node_data and stored_tap_node_data and (tap_label != stored_tap_label):
            #     triggered_attr = 'tapNodeData'
            # else:
            # ABC end commenting

            if (len(selected_labels) == 1) and (tap_label == selected_labels[0]):
                triggered_attr = 'tapNodeData'
                selected_node_data = None
                # print(f">>> {selected_node_data} and \n\n {stored_selected_node_data}")
                # HERE elif (len(selected_labels) == 1) and (tap_label == selected_labels[0]):
            else:
                if (selected_node_data is not None) and (len(selected_node_data) == 0):
                    selected_node_data = None
                    
                if (stored_selected_node_data is None) and (selected_node_data is None):
                    stored_selected_node_data = None # if selected_node_data is None else selected_node_data[:] # ABC added line
                elif (selected_node_data is None): #  or (len(selected_node_data == 0)):
                    stored_selected_node_data = None
                elif stored_selected_node_data is None:
                    stored_selected_node_data = selected_node_data[:] 
                else: # (stored_selected_node_data is not None) and (selected_node_data is not None)
                    stored_selected_labels = set([node['name'] for node in stored_selected_node_data])
                    for node in selected_node_data: # ABC added line
                        if (node['name'] not in stored_selected_labels) and (node['name'] != tap_label): # ABC added line
                            stored_selected_node_data.append(node) # ABC added line
                            stored_selected_labels.add(node['name']) # ABC added line
                    selected_node_data = None

                # ABC begin indented to the left
                # a = 0 if stored_selected_node_data is None else len(stored_selected_node_data)
                # if len(selected_node_data) - a > 1:
                # stored_selected_node_data = selected_node_data # ABC commented out

                box_selected_table, stat_download_button_style, box_selected_output = display_selected_nodes(stored_selected_node_data,
                                                                                                             stat_keeper_data_label,
                                                                                                             stat_test, alpha,
                                                                                                             stat_correction,
                                                                                                             surv_data_label, surv_t_label, surv_e_label)

                # update elements
                if stored_selected_node_data is None:
                    selected_obs = set()
                else:
                    selected_obs = set([node['name'] for node in stored_selected_node_data])

                elements = elements_in
                for element in elements['nodes']:
                    if element['data'].get('name') in selected_obs:
                        if 'classes' in element:
                            if 'highlight_box_select' in element['classes']:
                                continue
                            else:
                                element['classes'] += ' highlight_box_select'
                        else:
                            element['classes'] = 'nodes highlight_box_select'
                    elif 'classes' in element:
                        # element['classes'] = ''.join(element['classes'].split(' highlight_box_select'))
                        element['classes'] = element['classes'].replace(' highlight_box_select', '').strip()
                    else:
                        continue

                # # first remove previously highlighted nodes            
                # elements = elements_in
                # for element in elements['nodes']:
                #     if 'classes' in element:                    
                #         if ' highlight_box_select' in element['classes']:
                #             # print(f"unupdated selected box element : {element['classes']}")
                #             # element['classes'] = ''.join(element['classes'].split(' highlight_box_select'))
                #             element['classes'] = element['classes'].replace(' highlight_box_select', '').strip()

                #             # print(f"updated selected box element : {element['classes']}")

                # # update new selected nodes
                # # selected_obs = [node['name'] for node in select_node_data]
                # selected_obs = [node['name'] for node in selected_node_data]


                # for element in elements['nodes']:
                #     if element['data'].get('name') in selected_obs:
                #         if 'classes' in element:
                #             element['classes'] += ' highlight_box_select'
                #         else:
                #             element['classes'] = 'nodes highlight_box_select'
                #         selected_obs.remove(element['data'].get('name'))
                #         if len(selected_obs) == 0:
                #             break

                # stylesheet = stylesheet_in
                # stylesheet.append({
                #     'selector': '.highlight_box_select',
                #     'style': {
                #         'border-width': '2px',
                #         'border-color': 'gray',
                #         # 'z-index': 1999,
                #     },
                # })

                # no_update
                # bb = "\n".join([f"\n\n output selected node data: {selected_node_data}",
                #             f" output tap node data: {tap_node_data}",
                #             f" output stored selected node data: {stored_selected_node_data}"])
                # print(bb)
                return (elements, # stylesheet, # select_node_data,
                        node_attr, data_label, ft_label, fl_label_opts,
                        current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, 
                        node_fixed_color, mouseover_node_data,
                        # stored_tap_node_data,
                        stored_selected_node_data, box_selected_table, stat_download_button_style,                        
                        box_selected_output,
                        surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                        tap_node_data, None) # selected_node_data)
                              # ABC END INDENTATION


                              
        if triggered_input in ['stat-keeper-data-dropdown', 'stat-test-dropdown', 'alpha-label', 'correction-drop-down',
                               'time-attr', 'event-attr']:
            box_selected_table, stat_download_button_style, box_selected_output = display_selected_nodes(stored_selected_node_data,
                                                                                                         stat_keeper_data_label,
                                                                                                         stat_test, alpha,
                                                                                                         stat_correction,
                                                                                                         surv_data_label, surv_t_label, surv_e_label)
            # bb = "\n".join([f"\n\n output selected node data: {selected_node_data}",
            #                 f" output tap node data: {tap_node_data}",
            #                 f" output stored selected node data: {stored_selected_node_data}"])
            # print(bb)
            return (elements_in, # stylesheet, # select_node_data,
                    node_attr, data_label, ft_label, fl_label_opts,
                    current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, 
                    node_fixed_color, mouseover_node_data,
                    # stored_tap_node_data,
                    stored_selected_node_data, box_selected_table, stat_download_button_style,
                    box_selected_output,
                    surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                    tap_node_data, None) # selected_node_data)
        

        if triggered_input == 'keeper-data-dropdown':
            fl_label_opts = set_feature_options(data_label)
            ft_label = None
            # bb = "\n".join([f"\n\n output selected node data: {selected_node_data}",
            #                 f" output tap node data: {tap_node_data}",
            #                 f" output stored selected node data: {stored_selected_node_data}"])
            # print(bb)
            return (elements_in, # stylesheet, # select_node_data,
                    node_attr, data_label, ft_label, fl_label_opts,
                    current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, 
                    node_fixed_color, mouseover_node_data,
                    # stored_tap_node_data,
                    stored_selected_node_data, box_selected_table, stat_download_button_style,
                    box_selected_output,
                    surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                    tap_node_data, None) # selected_node_data)

        if triggered_input == 'surv-keeper-data-dropdown':
            surv_t_opts = surv_e_opts = set_feature_options(surv_data_label)
            surv_e_label = surv_t_label = None
            
            # box_selected_table, stat_download_button_style, box_selected_output = display_selected_nodes(stored_selected_node_data,
            #                                                                                              stat_keeper_data_label,
            #                                                                                              stat_test, alpha,
            #                                                                                              stat_correction,
            #                                                                                              surv_data_label, surv_t_label, surv_e_label)
            if '(log-rank p' in box_selected_output:
                box_selected_output = ''.join([box_selected_output.split(' (log-rank p')[0], ':', box_selected_output.split('):')[-1]])

            return (elements_in, # stylesheet, # select_node_data,
                    node_attr, data_label, ft_label, fl_label_opts,
                    current_zoom, current_pan, node_cbar_vis_in, edge_cbar_vis_in, 
                    node_fixed_color, mouseover_node_data,
                    # stored_tap_node_data,
                    stored_selected_node_data, box_selected_table, stat_download_button_style,
                    box_selected_output,
                    surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                    tap_node_data, None) # selected_node_data)
            
        D = keeper.distances[pose_dist_key].data
        G = keeper.graphs[pose_key]

        if pose_ordering:
            G = G.edge_subgraph([ee for ee in G.edges() if G.edges[ee]['edge_origin'].startswith('POSE')]) #  in ['POSE', 'POSE + NN']]

        # if change between POSE or ordering, reset positions_records
        if triggered_input == 'pose-switch':
            positions_records.clear()

        # if new pose, reset layouts and tapNodeData and selectedNodeData
        if triggered_input == 'pose-button':            
            positions_records.clear()            
            mouseover_node_data = None
            # tap_node_data = None # ABC added
            # stored_tap_node_data = None
            # stored_selected_node_data = None 
            # box_selected_table, stat_download_button_style, box_selected_output = display_selected_nodes(stored_selected_node_data,
            #                                                                                              stat_keeper_data_label,
            #                                                                                              stat_test, alpha,
            #                                                                                              stat_correction)

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
            elif layout == 'forceatlas2':
                pos = nfv.forceatlas2_layout(G)
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
            node_attr_value = None # BBB 'None'
            data_label = None, # BBB 'None'
            ft_label = None # '' AAA
            tap_node_data = None # stored_tap_node_data = None # ABC updated commented line
            # stored_tap_node_data = False # ABC added line
        elif (triggered_input == 'network-graph') and (triggered_attr == 'tapNodeData'):
            # stored_tap_node_data = tap_node_data # ABC commented out
            # print(f"ENTERED TAPNODEDATA \n-- select_node_data = {select_node_data}\n--tap_node_data = {tap_node_data}\n\n")
            node_attr = int(tap_node_data['id'])
            node_attr_value = None # BBB 'None'
            data_label = None, # BBB 'None'
            ft_label = None # '' AAA
            node_fixed_color = ''
            # if node_cmap in DISCRETE_COLORMAP_OPTIONS:
            #     node_cmap_type = 'sequential'
            #     node_cmap = SEQUENTIAL_COLORMAP_OPTIONS[0]
            # stored_tap_node_data = True # ABC added line
            # print(f" > stored_tap_node_data from tapNodeData trigger: = {'None' if stored_tap_node_data is None else stored_tap_node_data}") # ABC added line
        elif triggered_input == 'node-attribute-dropdown':
            # print("ENTERED NODE ATTRIBUTE DROPDOWN")
            node_attr_value = node_attr
            data_label = None, # BBB 'None'
            ft_label = None # '' AAA
            tap_node_data = None # stored_tap_node_data = None # ABC updated comented line
            # stored_tap_node_data = False # ABC added line
            node_fixed_color = '' 
        elif triggered_input == 'feature-label': # 'node-color-button':  # here
            node_attr_value = None # BBB 'None'
            tap_node_data = None # stored_tap_node_data = None # ABC updated comented line
            node_fixed_color = ''
            # stored_tap_node_data = False # ABC added line
        else:
            if tap_node_data is None: # stored_tap_node_data is None: # ABC updated commented line
                node_attr_value = node_attr 
            else: 
                node_attr = int(tap_node_data['id']) # int(stored_tap_node_data['id']) # ABC updated commented line
                node_attr_value = None # BBB  'None'
                data_label = None # BBB 'None'
                ft_label = None # ''  AAA
                node_fixed_color = ''

        # print(f"after - tap_node_data = {tap_node_data}")

        # BBB if (data_label != 'None') and (ft_label is not None):
        if (data_label is not None) and (ft_label is not None):
            nc = keeper.data[data_label].subset(features=[ft_label]).loc[ft_label].values
        else:
            nc = None

        # BBB nca = None if (node_attr=='None' and data_label=='None') else node_attr if node_attr!='None' else nc
        nca = None if (node_attr is None and data_label is None) else node_attr if (node_attr is not None) else nc
        # BBB eca = None if edge_attr=='None' else edge_attr
        eca = None if edge_attr is None else edge_attr
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
                                       # style={'width': '100%'},
                                       )]

        if edge_cbar_vis is None:
            edge_cbar_vis = []
        else:
            edge_cbar_vis = [dcc.Graph(figure=edge_cbar_vis,
                                       config={'displayModeBar': False},
                                       )]

        if (pos is None) and (layout not in positions_records):
            positions_records[layout] = {int(k['data']['id']): np.array([k['position']['x'], k['position']['y']]) for k in elements['nodes']}

        # Highlight specified node by label
        if node_label:
            labels = set(node_label.split(', '))
            for element in elements['nodes']:
                if element['data'].get('name') in labels: # == node_label:
                    if 'classes' in element:
                        element['classes'] += ' highlight_by_label'
                    else:
                        element['classes'] = 'nodes highlight_by_label'
                    labels.remove(element['data'].get('name'))
                    if len(labels) == 0:
                        break

        # print(f"BEFORE HIGHLIGHTING \n-- select_node_data = {select_node_data}\n--tap_node_data = {tap_node_data}\n\n")
        # Highlight selected box nodes
        # add select uncomment below:
        # if select_node_data:
        tap_label = '' if tap_node_data is None else tap_node_data['name']            # ABC commented line
        if stored_selected_node_data is not None:
            selected_obs = [node['name'] for node in stored_selected_node_data]

            for element in elements['nodes']:
                if (element['data'].get('name') in selected_obs) and (element['data'].get('name') != tap_label):
                    if 'classes' in element:
                        if 'highlight_box_select' in element['classes']:
                            continue
                        else:
                            element['classes'] += ' highlight_box_select'
                    else:
                        element['classes'] = 'nodes highlight_box_select'
                elif 'classes' in element:
                    # element['classes'] = ''.join(element['classes'].split(' highlight_box_select'))
                    element['classes'] = element['classes'].replace(' highlight_box_select', '').strip()
                else:
                    continue
            # for element in elements['nodes']:
            #     if element['data'].get('name') in selected_obs:
            #         if 'classes' in element:
            #             element['classes'] += ' highlight_box_select'
            #         else:
            #             element['classes'] = 'nodes highlight_box_select'                    
            #         selected_obs.remove(element['data'].get('name'))
            #         if len(selected_obs) == 0:
            #             break                
                    
                

        # MOVED TO ITS OWN CALLBACK
        # stylesheet=[

        #     {
        #         'selector': 'node',
        #         'style': {
        #             # 'label': 'data(name)',
        #             'width': node_size, # 'data(size)',
        #             'height': node_size, # 'data(size)',
        #             'background-color': 'data(color)',
        #             'text-opacity': '0',  # Hide labels by default,   
        #             'font-size': '8px',
        #             'border-width': '0.5px',  # Adds a thin border around the nodes
        #             'border-color': '#999',  # Sets the border color
        #             # 'z-index': 1001,
        #         },
        #     },
        #     {
        #         'selector': 'edge',
        #         'style': {
        #             'width': edge_width, # 'data(width)',
        #             'line-color': 'data(color)'
        #         },
        #     },
        #     # Class selectors
        #     {
        #         'selector': '.highlight',
        #         'style': {
        #             'border-width': '2px',
        #             'border-color': 'red',
        #             # 'outline-width': 20,
        #             # 'outline-color': 'black',
        #             # 'z-index': 1001,
        #         },
        #     },
        #     {
        #         'selector': '.highlight_box_select',
        #         'style': {
        #             'border-width': '2px',
        #             'border-color': 'black',
        #             # 'z-index': 1999,
        #         },
        #     },
        # ]
        
        # print(f"select node data out = {select_node_data}")
        # print(f"BEFORE RETURNING \n-- select_node_data = {select_node_data}\n--tap_node_data = {tap_node_data}\n\n")
        # return elements, stylesheet, select_node_data, 
            
        # return (elements, # stylesheet, # select_node_data, 
        #         node_attr_value, data_label, ft_label,
        #         current_zoom, current_pan, node_cbar_vis, edge_cbar_vis, tap_node_data,
        #         node_fixed_color, mouseover_node_data, # , node_cmap_type # , node_cmap
        #         )
        # bb = "\n".join([f"\n\n output selected node data: {selected_node_data}",
        #                     f" output tap node data: {tap_node_data}",
        #                     f" output stored selected node data: {stored_selected_node_data}"])
        # print(bb)
        return (elements, # stylesheet, # select_node_data,
                node_attr_value, data_label, ft_label, fl_label_opts,
                current_zoom, current_pan, node_cbar_vis, edge_cbar_vis, # tap_node_data, # NOTE: should this be here?
                node_fixed_color, mouseover_node_data,
                # stored_tap_node_data,
                stored_selected_node_data, box_selected_table, stat_download_button_style,
                box_selected_output,
                surv_t_label, surv_t_opts, surv_e_label, surv_e_opts,
                tap_node_data, None) # selected_node_data)

    
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

    
def render_pose(keeper, G, distance_key, port=8090):    
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
    app.run_server(mode='jupyterlab', host="127.0.0.1", port=port, dev_tools_ui=True, debug=True, # False,
              dev_tools_hot_reload=True, threaded=True)
