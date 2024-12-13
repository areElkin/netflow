import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re

from collections.abc import Iterable
from cycler import cycler
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.colors import BoundaryNorm, Colormap, to_rgba_array
from matplotlib.lines import Line2D

from .clustering import high_res_branch_graph
from .._logging import _gen_logger, set_verbose

logger = _gen_logger(__name__)


def plot_topology(G, 
                  pos=None, # 1, 2, 3, 4, 5
                  with_node_labels=False, # 1 was with_labels
                  with_edge_labels=False,
                  ax=None, # 1, 2, 3, 4, 5
                  nodelist=None,  # 1, 2, 3
                  edgelist=None, # 1, 3
                  node_size=300, # 1, 2, 3
                  node_color='#1f78b4', # 1, 2
                  edge_color='k', # 1, 3
                  node_cbar=False,
                  edge_cbar=False,
                  node_cbar_kws={'location':'bottom', 'orientation':'horizontal', 'pad':0.01},
                  edge_cbar_kws={'location':'bottom', 'orientation':'horizontal', 'pad':0.1},
                  node_cbar_label=None,
                  edge_cbar_label=None,
                  node_cbar_labelpad=None, #  (default = -10),
                  edge_cbar_labelpad=None,
                  node_cbar_label_y=None,
                  edge_cbar_label_y=None,
                  node_shape='o', # 1, 2, 3
                  node_shape_mapper=None,
                  node_alpha=None, # 1, 2 was alpha (same for nodes and edges)
                  edge_alpha=None, # 3, 5 was alpha
                  node_label_alpha=None, # 4 was alpha
                  edge_label_alpha=None, # 5 was alpha
                  node_cmap=None, # 1, 2 was cmap
                  edge_cmap=None, # 1, 3
                  node_vmin=None, # 1, 2 was vmin
                  node_vmax=None, # 1, 2 was vmax
                  edge_vmin=None, # 1, 3
                  edge_vmax=None, # 1, 3
                  node_cmap_drawedges=None,
                  edge_cmap_drawedges=None,
                  border_linewidths=None, # 1, 2 was linewidths
                  bordercolors=None, # 2 was edgecolors
                  edge_width=1.0, # 1, 3 was width                  
                  edge_style='solid', # 1, 3 was style
                  edge_style_mapper=None,
                  node_labels=None, # 1, 4 was labels
                  edge_labels=None,  # 5
                  node_show_legend=None,
                  edge_show_legend=None,
                  node_font_size=12, # 1, 4 was font_size (same for nodes and edges)
                  edge_font_size=10, # 5 was font_size
                  node_font_color='k', # 1, 4 was font_color (same for nodes and edges)
                  edge_font_color='k', # 5 was font_color
                  node_font_weight='normal', # 1, 4 was font_weight (same for nodes and edges)
                  edge_font_weight='normal', # 5 was font_weight
                  node_font_family='sans-serif', # 1, 4 was font_family (same for nodes and edges)
                  edge_font_family='sans-serif', # 5 was font_family
                  node_ticklabels_mapper=None,
                  node_bbox=None, # 4 was bbox
                  edge_bbox=None, # 5 was bbox
                  node_horizontalalignment='center', # 4 was horizontalalignment
                  edge_horizontalalignment='center', # 5 was horizontalalignment
                  node_verticalalignment='center', # 4 was verticalalignment
                  edge_verticalalignment='center', # 5 was verticalalignment
                  node_clip_on=True, # 4 was clip_on
                  edge_clip_on=True, # 5 was clip_on                  
                  margins=None, # 2
                  min_source_margin=0, # 3
                  min_target_margin=0, # 3
                  edge_label_pos=0.5, # 5 was label_pos                      
                  rotate=True, # 5
                 ):
    """ Draw the graph topology.

    Draw the graph topology with NetworkX with options for 
    multiple node shapes. 
    
    Parameters
    ----------
    G : graph
        A networkx graph    
    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a kamada kawai layout positioning will be computed.
        See :py:mod:`networkx.drawing.layout` for functions that
        compute node positions.    
    with_{node,edge}_labels :  bool (default=False)
        Set to True to draw labels on the nodes and edges.    
    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.
    nodelist : list (default=list(G))
        Draw only specified nodes.
    edgelist : list (default=list(G.edges()))
        Draw only specified edges.
    node_size : scalar or array (default=300)
        Size of nodes.  If an array is specified it must be the
        same length as nodelist.
    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.
    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    {node,edge}_cbar : bool (default = False)
        If `True`, show color bar for node and edge colors.
        This is ignored if {node,edge}_color is a str.
    {node,edge}_cbar_kws : dict
        Optional keyword arguments passed to `matplotlib.pyplot.colorbar` for
        the node and edge colorbars.
    {node,edge}_cbar_label : {`None`, `str`}
        Optional label for node and edge colorbar, if displayed.
    {node,edge}_cbar_labelpad : {`None`, `int`} (default = -10)
        Optional padding for node and edge colorbar label, if displayed.
    {node,edge}_cbar_label_y : {`None`, `float`}
        Optional y position for node and edge colorbar label, if displayed.
    node_shape :  string or array of shapes  (default='o')
        The shape of the node.  Can be a single shape or a sequence of shapes
        with the same length as nodelist. Shape specification is as 
        matplotlib.scatter marker, one of 'os^>v<dph8'.
        Alternatively, to add a description of the node class represented by
        the shape to the legend, provide a single node class label or a
        sequence of node class labels with the same length as nodelist. If 
        shape speficiations are not one of 'os^>v<dph8', they are treated as 
        node class labels and are added to the legend. Use ``node_shape_mapper`` 
        to manually specify the shape for each class.
    node_shape_mapper : None or dict (default=None)
        A dictionary to map node class labels to shapes, keyed by node class labels
        that is expected to have a key for all unique node class labels in ``node_shape``. 
        The default is to cycle through the possible shapes 'os^>v<dph8', which may 
        not be unique if there are more classes than unique markers. This is ignored
        if ``node_shape`` contains the shapes directly.
    node_alpha : float, array of floats or None (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).
    edge_alpha : float or None (default=None)
        The edge transparency.
    node_label_alpha : float or None (default=None)
        The node label text transparency
    edge_label_alpha : float or None (default=None)
        The text transparency
    {node,edge}_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of nodes and edges
    {node,edge}_vmin,{node,edge}_vmax : float, optional
        Minimum and maximum for node and edge colormap scaling
    {node,edge}_cmap_drawedges : `bool`
        Whether to draw lines at color boundries on node,edge colorbar.
        This is ignored if no colorbar is being shown.
        Default behavior is to draw edges for a discrete colormap and
        not for continuous colormaps.
    border_linewidths : scalar or sequence (default=1.0) 
        Line width of symbol border.
    bordercolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders.    
    edge_width : float or array of floats (default=1.0) 
        Line width of edges                          
    edge_style : string or array of strings (default=solid line)
        Edge line style. Can be a single style or a sequence of styles
        with the same length as edgelist. Styles can be any of
        '-', '--', '-.', ':' or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)
        Alternatively, to add a description of the edge class represented by
        the style to the legend, provide a single edge class label or a
        sequence of edge class labels with the same length as edgelist. If 
        style speficiations are not one of
        ['-', '--', '-.', ':', 'solid', 'dashed', 'dotted', 'dashdot'], they are treated 
        as edge  class labels and are added to the legend. Use ``edge_style_mapper`` 
        to manually specify the style for each class.
    edge_style_mapper : None or dict (default=None)
        A dictionary to map  edge class labels to styles, keyed by edge class 
        and is expected to have a key for all include all unique edge class labels in
        ``edge_style``. 
        The default is to cycle through the possible edge styles ['-', '--', '-.', ':'],    
        which may not be unique if there are more classes than unique styles. 
        This is ignored if ``edge_style`` contains the styles directly.
    node_labels : dictionary (default=None)
        Node labels in a dictionary of text labels keyed by node
    edge_labels : dictionary (default=None)
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.
    {node,edge}_show_legend : `bool`
        If `True`, add legend with node shape or edge style.
        Default is `True`, but this is ignored if ``node_shape`` or
        ``edge_style`` is not provided.
    {node,edge}_font_size : int (default=12 for nodes, 10 for edges)
        Font size for text labels    
    {node,edge}_font_color : string (default='k' black)
        Font color string
    {node,edge}_font_weight : string (default='normal')
        Font weight    
    {node,edge}_font_family : string (default='sans-serif')
        Font family
    node_ticklabels_mapper : {`None`, `dict`} (default=None)
        If provided, used to assign labels to numeric values in color-bar when a
        discrete colormap is used.
    node_bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.
    edge_bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.
    {node,edge}_horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}    
    {node,edge}_verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
    {node,edge}_clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries
    margins : float or 2-tuple, optional
        Sets the padding for axis autoscaling. Increase margin to prevent
        clipping for nodes that are near the edges of an image. Values should
        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
        for details. The default is `None`, which uses the Matplotlib default.
    min_source_margin : int (default=0)
        The minimum margin (gap) at the begining of the edge at the source.    
    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.
    edge_label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)
    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    See Also
    --------
    networkx.draw
    networkx.draw_networkx
    networkx.draw_networkx_nodes
    networkx.draw_networkx_edges
    networkx.draw_networkx_labels
    networkx.draw_networkx_edge_labels
    """
    node_shape_options = 'os^>v<dph8'
    edge_style_options = ['-', '--', '-.', ':', 'solid', 'dashed', 'dotted', 'dashdot']

    qualitative_cmaps = {'Pastel1': 9, 
                         'Pastel2': 8, 
                         'Paired': 12, 
                         'Accent': 8,                         
                         'Dark2': 8, 
                         'Set1': 9, 
                         'Set2': 8, 
                         'Set3': 12,
                         'tab10': 10, 
                         'tab20': 20, 
                         'tab20b': 20, 
                         'tab20c': 20} 
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if pos is None:
        pos = nx.layout.kamada_kawai_layout(G)

    if nodelist is None:
        nodelist = list(G)

    if edgelist is None:
        edgelist = list(G.edges())

    if node_cmap_drawedges is None:
        if node_cbar and (not isinstance(node_color, str)) and (node_cmap is not None):        
            node_cmap_drawedges = True if node_cmap in qualitative_cmaps.keys() else False
        else:
            node_cmap_drawedges = False
    if edge_cmap_drawedges is None:
        if edge_cbar and (not isinstance(edge_color, str)) and (edge_cmap is not None):        
            edge_cmap_drawedges = True if edge_cmap in qualitative_cmaps.keys() else False
        else:
            edge_cmap_drawedges = False
    
    legend_handles = []

    # draw nodes for each shape
    if isinstance(node_shape, str):
        node_shape = [node_shape]*len(nodelist)
    if set(node_shape).issubset(set(node_shape_options)): # plot for each shape without legend
        node_shape_mapper = None
    else: # plot for each shape with legend
        if node_shape_mapper is None:
            node_shape_mapper = dict(zip(sorted(set(node_shape)), itertools.cycle(node_shape_options)))
        node_shape = [node_shape_mapper[k] for k in node_shape]
        
    if isinstance(node_color, Iterable) and not isinstance(node_color, str):
        # check if sequence is float or strings
        # if strings, check if colors or categories
        
        try:            
            _ = to_rgba_array(node_color)
        except:
            if node_vmin is None:
                node_vmin = min(node_color)
            if node_vmax is None:
                node_vmax = max(node_color)       

    # for node colorbar:
    if node_cbar and (not isinstance(node_color, str)) and (node_cmap is not None):
        nc_set = sorted(set([k for k in node_color if node_vmin <= k <= node_vmax]))
        cur_vmax = max(nc_set)
        cur_vmin = min(nc_set)        
        if node_cmap in qualitative_cmaps.keys():
            node_vmax = qualitative_cmaps[node_cmap]            
            if len(nc_set) <= qualitative_cmaps[node_cmap]:
                # node_boundaries = [k/qualitative_cmaps[node_cmap] for k in range(max(node_color)+2)]
                node_boundaries = [k/qualitative_cmaps[node_cmap] for k in range(cur_vmin, cur_vmax+2)]
                # ticks = [k/qualitative_cmaps[node_cmap] + 0.5*(1/qualitative_cmaps[node_cmap]) for k in range(max(node_color)+1)]
                node_ticks = [k/qualitative_cmaps[node_cmap] + 0.5*(1/qualitative_cmaps[node_cmap]) for k in range(cur_vmin, cur_vmax+1)]
                if node_ticklabels_mapper is None:
                    node_ticklabels = [str(k) for k in nc_set]
                else:
                    node_ticklabels = [node_ticklabels_mapper[k] for k in nc_set]
                node_norm = BoundaryNorm(node_boundaries, len(nc_set))
            else:
                if node_cmap_drawedges:
                    # node_boundaries = np.append(np.append(nc_set[0], np.array(nc_set[:-1]) + (np.diff(nc_set) / 2)), nc_set[-1]) # nc_set                    
                    node_boundaries = np.append(np.append(nc_set[0]-(np.diff(nc_set[:2])/2), np.array(nc_set[:-1]) + (np.diff(nc_set) / 2)),
                                                nc_set[-1] + (np.diff(nc_set[-2:]) / 2)) 
                else:
                    node_boundaries = None
                node_ticks = [cur_vmin, cur_vmax]
                node_ticklabels = [str(np.round(nc_set[0], 2)), str(np.round(nc_set[-1], 2))]            
                node_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)            
        else:            
            # if cur_vmax - cur_vmin < 1e-6:
            #     cur_vmax += 1e-3
            #     cur_vmin -+ 1e-3 
            #     if isinstance(cur_vmax, float):
            #         cur_vmax += 1e-3
            #         cur_vmin -+ 1e-3
            #     else:
            #         cur_vmax += 1
            #         cur_vmin -+ 1
            # node_vmax=None
            # nc_set = sorted(set(node_color))            
            if node_cmap_drawedges:
                nc_set = sorted(set([k for k in node_color if node_vmin <= k <= node_vmax]))
                # node_boundaries = np.append(np.append(nc_set[0], np.array(nc_set[:-1]) + (np.diff(nc_set) / 2)), nc_set[-1]) # nc_set
                # node_boundaries = np.append(np.append(nc_set[0]-(np.diff(nc_set[:2])/2), np.array(nc_set[:-1]) + (np.diff(nc_set) / 2)),
                #                                 nc_set[-1] + (np.diff(nc_set[-2:]) / 2)) 
                if len(nc_set) == 1:
                    node_boundaries = [nc_set[0] - 1e-3, nc_set[0] + 1e-3]
                    node_ticks = nc_set
                    node_ticklabels = [str(np.round(nc_set[0], 2))]
                    node_norm = plt.Normalize(vmin=nc_set[0], vmax=nc_set[0]+1e-3)
                else:
                    node_boundaries = np.append(np.append(nc_set[0]-(np.diff(nc_set[:2])/2), np.array(nc_set[:-1]) + (np.diff(nc_set) / 2)),
                                                nc_set[-1] + (np.diff(nc_set[-2:]) / 2))
                    node_ticks = [cur_vmin, cur_vmax]
                    node_ticklabels = [str(np.round(nc_set[0], 2)), str(np.round(nc_set[-1], 2))]
                    node_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)
                
            else:                
                node_boundaries = None
                if cur_vmax > cur_vmin:
                    node_ticks = [cur_vmin, cur_vmax]
                    node_ticklabels = [str(np.round(nc_set[0], 2)), str(np.round(nc_set[-1], 2))]
                    node_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)
                else:
                    node_boundaries = [cur_vmin- 1e-3, cur_vmin+1e-3]
                    node_ticks = [cur_vmin]
                    node_ticklabels = [str(np.round(nc_set[0], 2))]
                    # node_norm = plt.Normalize(vmin=nc_set[0], vmax=nc_set[0]+1e-3)
                    node_norm = plt.Normalize(vmin=nc_set[0], vmax=nc_set[0]+1e-3)
            # node_ticks = [cur_vmin, cur_vmax]
            # node_ticklabels = [str(np.round(nc_set[0], 2)), str(np.round(nc_set[-1], 2))]
            # node_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)

    elif (not isinstance(node_color, str)) and (node_cmap is not None):
        if node_cmap in qualitative_cmaps.keys():
            node_vmax = max(node_vmax, qualitative_cmaps[node_cmap])

    # for edge colorbar:
    if (isinstance(edge_color, Iterable)) and (not isinstance(edge_color, str)):
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
            
    if edge_cbar and (not isinstance(edge_color, str)) and (edge_cmap is not None):
        
        ec_set = sorted(set([k for k in edge_color if edge_vmin <= k <= edge_vmax]))
        cur_vmax = max(ec_set)
        cur_vmin = min(ec_set)
        # if edge_cmap in qualitative_cmaps.keys():
        #     edge_vmax = qualitative_cmaps[edge_cmap]            
        #     if len(ec_set) <= qualitative_cmaps[edge_cmap]:
        #         # edge_boundaries = [k/qualitative_cmaps[edge_cmap] for k in range(max(edge_color)+2)]
        #         edge_boundaries = [k/qualitative_cmaps[edge_cmap] for k in range(cur_vmin, cur_vmax+2)]
        #         # ticks = [k/qualitative_cmaps[edge_cmap] + 0.5*(1/qualitative_cmaps[edge_cmap]) for k in range(max(edge_color)+1)]
        #         edge_ticks = [k/qualitative_cmaps[edge_cmap] + 0.5*(1/qualitative_cmaps[edge_cmap]) for k in range(cur_vmin, cur_vmax+1)]
        #         edge_ticklabels = [str(k) for k in ec_set]
        #         edge_norm = BoundaryNorm(edge_boundaries, len(ec_set))
        #     else:
        #         edge_boundaries = None
        #         edge_ticks = [cur_vmin, cur_vmax]
        #         edge_ticklabels = [str(np.round(ec_set[0], 2)), str(np.round(ec_set[-1], 2))]            
        #         edge_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)
        # else:
        #     # edge_vmax=None
        #     # ec_set = sorted(set(edge_color))
        #     edge_boundaries = None
        #     edge_ticks = [cur_vmin, cur_vmax]
        #     edge_ticklabels = [str(np.round(ec_set[0], 3)), str(np.round(ec_set[-1], 3))]
        #     edge_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)
        if edge_cmap_drawedges:
            # edge_boundaries = np.append(np.append(ec_set[0], np.array(ec_set[:-1]) + (np.diff(ec_set) / 2)), ec_set[-1]) # ec_set
            edge_boundaries = np.append(np.append(ec_set[0]-(np.diff(ec_set[:2])/2), np.array(ec_set[:-1]) + (np.diff(ec_set) / 2)),
                                        ec_set[-1] + (np.diff(ec_set[-2:]) / 2)) 
        else:
            edge_boundaries = None
        edge_ticks = [cur_vmin, cur_vmax]
        edge_ticklabels = [str(np.round(ec_set[0], 3)), str(np.round(ec_set[-1], 3))]
        edge_norm = plt.Normalize(vmin=cur_vmin, vmax=cur_vmax)
        
    for shape_class in set(node_shape):
        indices = [i for i, j in enumerate(node_shape) if j == shape_class]
        nodelist_tmp = [nodelist[i] for i in indices]
        if isinstance(node_size, Iterable) and not isinstance(node_size, str):
            node_size_tmp = [node_size[i] for i in indices]
        else:
            node_size_tmp = node_size
        if isinstance(node_color, Iterable) and not isinstance(node_color, str):
            node_color_tmp = [node_color[i] for i in indices]
        else:
            node_color_tmp = node_color
        if isinstance(border_linewidths, Iterable):
            border_linewidths_tmp = [border_linewidths[i] for i in indices]
        else:
            border_linewidths_tmp = border_linewidths
        if bordercolors is None:
            bordercolors_tmp = bordercolors
        elif isinstance(bordercolors, Iterable) and not isinstance(bordercolors, str):
            bordercolors_tmp = [bordercolors[i] for i in indices]
        else:
            bordercolors_tmp = bordercolors
            
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist_tmp, node_size=node_size_tmp,                    
                               node_color=node_color_tmp, node_shape=shape_class,
                               alpha=node_alpha, # MAYBE CHANGE TO ALPHA BY NODE?
                               cmap=node_cmap, vmin=node_vmin, vmax=node_vmax,
                               linewidths=border_linewidths_tmp, edgecolors=bordercolors_tmp,
                               margins=margins,
                               )

        
    if node_shape_mapper is not None:
        node_show_legend =  True if node_show_legend is None else node_show_legend
        if node_show_legend:
            for class_label, class_shape in node_shape_mapper.items():
                legend_handles.append(Line2D([0], [0], marker=class_shape, color='k',
                                             label=class_label, lw=0,
                                             markerfacecolor='gray', markersize=10))
                    

    # draw edges for each style
    if isinstance(edge_style, str):
        edge_style = [edge_style]*len(edgelist)
    if set(edge_style).issubset(set(edge_style_options)): # plot for each style without legend
        edge_style_mapper = None
    else: # plot for each style with legend
        if edge_style_mapper is None:
            edge_style_mapper = dict(zip(sorted(set(edge_style)), itertools.cycle(edge_style_options))) 
        edge_style = [edge_style_mapper[k] for k in edge_style]
    
    for edge_class in set(edge_style):
        indices = [i for i, j in enumerate(edge_style) if j == edge_class]
        edgelist_tmp = [edgelist[i] for i in indices]

        if isinstance(edge_width, Iterable):
            edge_width_tmp = [edge_width[i] for i in indices]
        else:
            edge_width_tmp = edge_width
        if isinstance(edge_color, Iterable) and not isinstance(edge_color, str):
            edge_color_tmp = [edge_color[i] for i in indices]
        else:
            edge_color_tmp = edge_color
        
        nx.draw_networkx_edges(G, pos, edgelist=edgelist_tmp, # None, 
                               width=edge_width_tmp, edge_color=edge_color_tmp, style=edge_class,
                               alpha=edge_alpha,
                               edge_cmap=edge_cmap,
                               edge_vmin=edge_vmin, edge_vmax=edge_vmax, ax=ax,
                               node_size=node_size, 
                               nodelist=nodelist, 
                               node_shape='8', # node_shape, # 'o',  
                               min_source_margin=min_source_margin, min_target_margin=min_target_margin,
                               )

    if edge_style_mapper is not None:
        edge_show_legend =  True if edge_show_legend is None else edge_show_legend
        if edge_show_legend:
            for class_label, class_style in edge_style_mapper.items():
                legend_handles.append(Line2D([0], [0], marker=None, color='k',
                                             label=class_label, lw=2, ls=class_style))
    

    if with_node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=node_font_size, font_color=node_font_color, 
                                font_family=node_font_family, font_weight=node_font_weight, alpha=node_label_alpha, 
                                bbox=node_bbox, horizontalalignment=node_horizontalalignment, 
                                verticalalignment=node_verticalalignment, ax=ax, clip_on=node_clip_on, 
                                )

    if with_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=edge_label_pos, 
                          font_size=edge_font_size, font_color=edge_font_color, 
                          font_family=edge_font_family, font_weight=edge_font_weight, # 'normal', 
                          alpha=edge_label_alpha, bbox=edge_bbox, horizontalalignment=edge_horizontalalignment,
                          verticalalignment=edge_verticalalignment, ax=ax, rotate=rotate, clip_on=edge_clip_on,
                         )

    if len(legend_handles) > 0:
        legend = ax.legend(handles=legend_handles)

    # add colorbars
    # for node colorbar:    
    if node_cbar and (not isinstance(node_color, str)) and (node_cmap is not None):
        node_sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
        node_sm._A = []        

        cur_node_cbar = plt.colorbar(node_sm, ax=ax, ticks=node_ticks, boundaries=node_boundaries,                                     
                                     drawedges=node_cmap_drawedges, **node_cbar_kws)

        pad_node_ticklabels = False
        if node_ticklabels is not None:
            cur_node_cbar.set_ticklabels(node_ticklabels);
            if (len(node_ticklabels) > 2) or (len(node_ticklabels) == 1):
                pad_node_ticklabels = True
                
        node_cbar_label = "nodes" if node_cbar_label is None else "nodes: " + node_cbar_label
        if node_cbar_label is not None:
            node_cbar_labelpad = (-10 if (((node_cmap is None) or (node_cmap not in qualitative_cmaps)) and (not pad_node_ticklabels)) else 1) # -30

        cur_node_cbar.set_label(node_cbar_label, labelpad=node_cbar_labelpad, y=node_cbar_label_y)
    
    # for edge colorbar:
    if edge_cbar and (not isinstance(edge_color, str)) and (edge_cmap is not None):
        edge_sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        edge_sm._A = []
        
        cur_edge_cbar = plt.colorbar(edge_sm, ax=ax, ticks=edge_ticks, boundaries=edge_boundaries,
                                     drawedges=edge_cmap_drawedges, **edge_cbar_kws)

        edge_cbar_label = "edges" if edge_cbar_label is None else "edges: " + edge_cbar_label
        if edge_cbar_label is not None:
            edge_cbar_labelpad = -10 # -30            
        cur_edge_cbar.set_label(edge_cbar_label, labelpad=edge_cbar_labelpad, y=edge_cbar_label_y)

        if edge_ticklabels is not None:
            cur_edge_cbar.set_ticklabels(edge_ticklabels);
    
    return ax


def KM_between_groups(T, E, groups, min_group_size=10,
                      figsize=(9,6), show_censors=False, ci_show=False,
                      show_at_risk_counts=True, precision=4,
                      ttl='', xlabel=None, colors=None, ax=None, **kwargs):
    """ KM analysis for each group, with log-rank p-values.

    Note: Does not check if any observation appears in more than one group
    before plotting the KM curve. However, common observations are removed
    before computing the log-rank p-value.

    If ax is provided, any artists are removed from the legend and set
    to not be visible.

    Parameters
    ----------
    T : `pandas.Series`
        Time to event.
    E : `pandas.Series`
        Indicate if event occured (1) or not (0).
    groups 
        ?
    min_group_size : `int`
        Minimum group size to be included in analysis.
    figsize 
        Figure size.
    precision : `int`
        Precision shown for log-rank p-value.
    ttl : `str`, optional
        Figure title.
    colors : `dict`
        (Optional) Colors to plot lines, keyed by group label.
    ax : `matplotlib.axes._axes.Axes`
        (Optional) the axis to use for plotting the survival curves.
    **kwargs : `dict`
        Key-word arguments passed to `lifelines.KaplanMeierFitter.plot`
    """
    T = T.loc[groups.index]
    E = E.loc[groups.index]
    
    c_record = []

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        # to remove any old legend for making a gif:
        for _artist  in ax.lines + ax.collections + ax.patches + ax.images: # put this before you call the 'mean' plot function.
            _artist.set_label(s=None)
            _artist.set_visible(False)
    kmfs = []

    # first remove observations with NaN survival data:
    obs_nan = T.index[T.isna()].tolist()
    obs_nan = list(set(obs_nan) | set(E.index[E.isna()].tolist()))
    if len(obs_nan)>0:
        T = T.drop(index=obs_nan)
        E = E.drop(index=obs_nan)
        groups = groups.drop(index=obs_nan)

    for group_label in sorted(groups.unique()):
        group_members = groups.index[groups == group_label].tolist()
        if len(group_members) >= min_group_size:
            c_record.append(group_label)
            
            kmf = KaplanMeierFitter()
            kmf.fit(T.loc[group_members], E.loc[group_members], 
                    label=f"{group_label} (n = {len(group_members)})")
            kmfp = kmf.plot(ax=ax, show_censors=show_censors, ci_show=ci_show,
                            c=None if colors is None else colors[group_label],
                            **kwargs)
            kmfs.append(kmf)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ttl2 = ''

    for clus_a, clus_b in itertools.combinations(groups.unique(), 2):
        group_a = groups.index[groups==clus_a].tolist()
        group_b = groups.index[groups==clus_b].tolist()
        intersect_ab = set(group_a) & set(group_b)
        if len(intersect_ab) > 0:
            print(f"Removing {len(intersect_ab)} observations found in both {clus_a} and {clus_b}")
            group_a = list(set(group_a) - intersect_ab)
            group_b = list(set(group_b) - intersect_ab)
        if (len(group_a) >= min_group_size) and (len(group_b) >= min_group_size):
            lrp = logrank_test(T.loc[group_a], T.loc[group_b], 
                               event_observed_A=E.loc[group_a], 
                               event_observed_B=E.loc[group_b])            
            lrp = lrp.p_value
            if lrp <=0.05:
                ttl2 += f"\n{clus_a} vs {clus_b} :  p = {np.round(lrp, precision)}"

    if show_at_risk_counts:
        add_at_risk_counts(*kmfs, ax=ax)
    plt.tight_layout()

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_ylabel('survival')

    total_ttl = ttl
    if ttl2:
        if ttl:
            total_ttl = ttl + '\n' + 'logrank ' + ttl2
    ax.set_title(total_ttl);


def alpha_numeric_sorted(a): 
    """ Sort the given iterable alpha-numerically.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(a, key = alphanum_key)

def dotplot(x, y, c=None, s=None, size_exponent=None, marker=None, cmap=None, 
            vmin=None, vmax=None, norm=None, 
            cbar_label=None, cbar_labelpad=None, cbar_label_y=None,
            cbar_kwargs=None, ax=None, figsize=(5, 4), 
            size_legend_kwargs=None, size_legend_label=None, **kwargs):
    """ plot dotplots with prescribed color and shape

    Dots are plotted via `matplotlib.pyplot.scatter`.
    
    Parameters
    ----------
    x
    y
    c
    s
    size_exponent
    marker
    cmap
    vmin
    vmax
    norm
    cbar_label
    cbar_labelpad
    cbar_label_y
    cbar_kwargs : `dict`
    ax
    figsize
    size_legend_kwargs
    size_legend_label
    kwargs
    """
    if cmap is None:
        cmap = 'jet'
    if marker is None:
        marker = 'o'

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    if c is not None:
        if vmin is None:
            vmin = min(c)
        if vmax is None:
            vmax = max(c)

    if 'linewidths' in kwargs:
        lw = kwargs['linewidths']
    else:
        lw = 1
        
    if (s is not None) and (not isinstance(s, (float, int))):
        if size_legend_kwargs is None:
            size_legend_kwargs =  {'loc' : 'center left',
                                   'bbox_to_anchor' : (1, 0.5)}
        # if size_legend_label is None:
        #     size_legend_label = None        
        size_min = min(s)
        size_max = max(s)

        size_range = np.linspace(size_min, size_max, num=4, endpoint=True)[::-1] # markersizes
        if size_min != 0 or size_max != 1:
            dot_range = size_max - size_min # size_range
            size_values = (size_range - size_min) / dot_range
        else:
            size_values = size_range

        if size_exponent is None:
            size = size_values
        else:
            # size = size_values**size_exponent
            # size = size * (size_max - size_min) + size_min
            size = size_values*10 + size_min

        legend_markers = [Line2D([], [], color="white", marker='o',
                                 markerfacecolor="lightgray",
                                 markersize=ms, markeredgewidth=lw,
                                 markeredgecolor='k') for ms in size]
        lgd = plt.legend(legend_markers,                          
                         [str(int(np.round(ms, 2))) for ms in size_range],
                         numpoints=1, title=size_legend_label, **size_legend_kwargs)
        if size_exponent is not None:
            s = [k**size_exponent for k in s]

    if all([isinstance(k, (float, int)) for k in x]):
        xx = x
        x_ref = None
    else:
        x_ref = dict(zip(alpha_numeric_sorted(x), range(len(x))))        
        xx = [x_ref[k] for k in x]
        x_ref = {vv: kk for kk, vv in x_ref.items()}

    if all([isinstance(k, (float, int)) for k in y]):
        yy = y
        y_ref = None
    else:
        y_ref = dict(zip(alpha_numeric_sorted(y), range(len(y))))        
        yy = [y_ref[k] for k in y]
        y_ref = {vv: kk for kk, vv in y_ref.items()}

    # ax.scatter(x, y, s=s,
    ax.scatter(xx, yy, s=s,
               marker=marker, c=c, cmap=cmap, vmin=vmin, vmax=vmax, 
               **kwargs)


    x_ticks = sorted(set(xx))    
    x_labels = None if x_ref is None else [str(x_ref[k]) for k in x_ticks]

    y_ticks = sorted(set(yy))    
    y_labels = None if y_ref is None else [str(y_ref[k]) for k in y_ticks]
    
    _ = ax.set_xticks(x_ticks, labels=x_labels)
    _ = ax.set_yticks(y_ticks, labels=y_labels)
    # _ = ax.set_xticks(sorted(set(x)))
    # _ = ax.set_yticks(sorted(set(y)))

    if c is not None:
        if cbar_labelpad is None:
            cbar_labelpad = -10 # -30
        # cbar_label_y = cbar_label_y or 0.5
        if cbar_label is None:
            cbar_label = 'mean value'
        if cbar_kwargs:
            cbar_kwargs = {'location':'bottom', 'orientation':'horizontal', 'pad':0.01} 
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, # ticks=[vmin, vmax],
                          **cbar_kwargs,
                         )
        cb.set_label(cbar_label, labelpad=cbar_labelpad, y=cbar_label_y)
        cb.set_ticklabels([str(np.round(vmin, 2)), str(np.round(vmax, 2))])



def _branch_order(G_tda, Ghr):
    """ Get ordered branches starting with longest branch. """
    nl = [_largest_branch(G_tda)]
    nl_update = nl[:] # [_largest_branch(G_tda)]
    ix = 0

    while len(nl) < len(Ghr):
        for k in nl:
            nbhd = list(Ghr.neighbors(k)) # _neighboring_branches(G_tda, k) #  # changed
            new_nbhd = list(set(nbhd) - set(nl_update))
            if len(new_nbhd) == 0:
                continue
            aa = dict(Ghr.subgraph(new_nbhd).nodes.data())
            nl_update = nl_update + sorted(aa, key=lambda x: aa[x]['n_members'], reverse=True)
        nl = nl_update[:] 
        ix+=1
        if ix > len(Ghr):
            raise AssertionError("Failed to converge")

    return nl_update



def _branch_member_labels(G_tda, branch):
    members = [G_tda.nodes[k]['name'] for k in G_tda if G_tda.nodes[k]['branch']==branch]
    return members


def _branch_member_indices(G_tda, branch):
    members = [k for k in G_tda if G_tda.nodes[k]['branch']==branch]
    return members


def _sorted_members(G_tda, d, branch, root=None):
    if root is None:
        if 'root' in G_tda.nodes[0]:
            root = [G_tda.nodes[k]['name'] for k in G_tda if G_tda.nodes[k]['root']==1][0]
        else: # is_root'
            root = [G_tda.nodes[k]['name'] for k in G_tda if G_tda.nodes[k]['is_root']=='Yes'][0]
    if isinstance(root, int):
        root = G_tda.nodes[root]['name']
    branch_labels = _branch_member_labels(G_tda, branch)
    members = d.loc[root, branch_labels].sort_values(ascending=False).index.tolist()
    return members


def _largest_branch(G_tda):
    branches = list(dict(G_tda.nodes.data('branch')).values())
    count = {k: branches.count(k) for k in set(branches) - set([-1])}
    ix = max(count, key=count.get)
    return ix


def _trace_sin(n_points, start=0, resolution=0.35, h=1., period=np.pi/8):
    if period <= 0:
        raise AssertionError("period must be greater than 0.")
    if h <= 0:
        raise AssertionError("h must be greater than 0.")
    if resolution <= 0:
        raise AssertionError("resolution must be greater than 0.")

    x = np.linspace(start, start + n_points*resolution, n_points)
    y = h*np.sin(period*x)
    pos = np.asarray([x, y]).T
    return pos


def _curve_sin(n_points, a=1., b=np.pi, C=0., h=0.5,  start=1.5):
    """
    Parameters
    ----------
    h 
        height
    a  
        period
    """
    t = np.linspace(start, start + n_points*b*2, n_points)
    # t = np.linspace(start, start + n_points*2, n_points)
    x = t # r(t) * np.cos(t)
    y = h*np.sin(a*t) # r(t) * np.sin(t)
    pos = np.asarray([x, y]).T
    return pos


def _curve_trace(n_points, a=0.5, b=np.pi, C=0., h=0.5,  start=1.5):
    """ Trace out the wavy spiral curve.

    Parameters
    ----------
    a : `float`
        Affects radius of the wave. Larger values result in a tighter wave.
    h : `float`
        Affects how tight and long the spiral is. Smaller values result in
        a tighter, longer  spiral
    """
    f = lambda t, a, C: a/2 * (1/np.sinh(t)  + t * np.sqrt(t**2)) + C
    radius = lambda t, a, b, C: a * t + np.sin(b*f(t, a, C));    
    r = lambda t: radius(t,a,b, C)
    t = np.linspace(start, start + h*b, n_points)
    x = r(t) * np.cos(t)
    y = r(t) * np.sin(t)
    pos = np.asarray([x, y]).T
    return pos

# testing



def _interbranch_edges(G_tda, branch_a, branch_b):
    edges = [ee for ee in G_tda.edges() if ((G_tda.nodes[ee[0]]['branch'] in [branch_a,
                                                                              branch_b]) and \
                                            (G_tda.nodes[ee[1]]['branch'] in [branch_a,
                                                                              branch_b]) and \
                                            (G_tda.nodes[ee[0]]['branch'] != G_tda.nodes[ee[1]]['branch']))]

    edges = [ee if G_tda.nodes[ee[0]]['branch'] == branch_a else (ee[1], ee[0]) for ee in edges]    
    return edges


def _neighboring_branches(G_tda, branch):
    members = _branch_member_indices(G_tda, branch)
    branches = set()
    for v in members:
        for k in set(G_tda.neighbors(v)) - set(members):
            branches.add(G_tda.nodes[k]['branch'])

    if branch in branches:
        logger.msg(f"Branch shouldn't be included...")
        branches = branches = set([branch])
    return branches


def sin_layout(G_tda, d, res=0.35, h=1, period=np.pi/8, sep=1, root=None):
    """ Node positions for sinusoidal layout of branches """
    Ghr = high_res_branch_graph(G_tda)
    ix_ref = {G_tda.nodes[k]['name'] : k for k in G_tda}

    ordered_branches = _branch_order(G_tda, Ghr)
    wavy_spiral_record = {}

    # first branch - longest branch
    branch_0 = ordered_branches[0] # _largest_branch(G_tda)    
    cur_branch_labels = _sorted_members(G_tda, d, branch_0, root=root)
    cur_branch_labels = [ix_ref[k] for k in cur_branch_labels]
    # n_0 = len(cur_branch_labels)

    POS = _trace_sin(len(cur_branch_labels), start=0,
                     resolution=res, h=h, period=period)

    wavy_spiral_record[branch_0] = {'nodelist': cur_branch_labels,
                                    'pos': dict(zip(cur_branch_labels, POS))}

    for branch_1 in ordered_branches[1:]:
        cur_branch_labels = _sorted_members(G_tda, d, branch_1, root=root)
        cur_branch_labels = [ix_ref[k] for k in cur_branch_labels]

        POS = _trace_sin(len(cur_branch_labels), start=0,
                         resolution=res, h=h, period=period)

        wavy_spiral_record[branch_1] = {'nodelist': cur_branch_labels,
                                        'pos': dict(zip(cur_branch_labels, POS))}

        branch_x = list(set(Ghr.neighbors(branch_1)) & set(wavy_spiral_record.keys()))[0]
        connecting_edges = _interbranch_edges(G_tda, branch_x, branch_1)[0]

        local_pos_source = wavy_spiral_record[branch_x]['pos'][connecting_edges[0]]
        local_pos_target = wavy_spiral_record[branch_1]['pos'][connecting_edges[1]]

        xy_min_source = np.asarray(list(wavy_spiral_record[branch_x]['pos'].values())).min(axis=0)
        xy_max_source = np.asarray(list(wavy_spiral_record[branch_x]['pos'].values())).max(axis=0)
        xy_min_target = np.asarray(list(wavy_spiral_record[branch_1]['pos'].values())).min(axis=0)
        xy_max_target = np.asarray(list(wavy_spiral_record[branch_1]['pos'].values())).max(axis=0)

        dx = local_pos_source[0] - local_pos_target[0]
        dy = np.abs(xy_max_target[1] - (xy_min_source[1]-sep))

        wavy_spiral_record[branch_1]['pos'] = {k: (v+[dx, -dy]) for k,v in wavy_spiral_record[branch_1]['pos'].items()}

    # nodelist = []
    POS = {}
    for k,v in wavy_spiral_record.items():
        # nodelist += v['nodelist']
        POS = {**POS, **v['pos']}

    return POS


def wavy_curve_layout(G_tda, d, a=0.5, b=0.5*np.pi, C=0, h=8, t_start=1.5,
                      sep=1, root=None):
    """ Node positions for wavy spiral layout of branches """
    Ghr = high_res_branch_graph(G_tda)
    ix_ref = {G_tda.nodes[k]['name'] : k for k in G_tda}

    ordered_branches = _branch_order(G_tda, Ghr)
    wavy_spiral_record = {}

    # first branch - longest branch
    branch_0 = ordered_branches[0] # _largest_branch(G_tda)    
    cur_branch_labels = _sorted_members(G_tda, d, branch_0, root=root)
    cur_branch_labels = [ix_ref[k] for k in cur_branch_labels]
    n_0 = len(cur_branch_labels)

    POS = _curve_trace(len(cur_branch_labels), a=a, b=b, C=C,
                       h=len(cur_branch_labels)/h,
                       start=t_start)
    wavy_spiral_record[branch_0] = {'nodelist': cur_branch_labels,
                                    'pos': dict(zip(cur_branch_labels, POS))}

    for branch_1 in ordered_branches[1:]:
        cur_branch_labels = _sorted_members(G_tda, d, branch_1, root=root)
        cur_branch_labels = [ix_ref[k] for k in cur_branch_labels]

        POS = _curve_trace(len(cur_branch_labels), a=a, b=b, C=C,
                           h=len(cur_branch_labels)/h, start=t_start)
        
        wavy_spiral_record[branch_1] = {'nodelist': cur_branch_labels,
                                        'pos': dict(zip(cur_branch_labels, POS))}

        branch_x = list(set(Ghr.neighbors(branch_1)) & set(wavy_spiral_record.keys()))[0]
        connecting_edges = _interbranch_edges(G_tda, branch_x, branch_1)[0]
        
        local_pos_source = wavy_spiral_record[branch_x]['pos'][connecting_edges[0]]        
        local_pos_target = wavy_spiral_record[branch_1]['pos'][connecting_edges[1]]

        xy_min_source = np.asarray(list(wavy_spiral_record[branch_x]['pos'].values())).min(axis=0)
        xy_max_source = np.asarray(list(wavy_spiral_record[branch_x]['pos'].values())).max(axis=0)
        xy_min_target = np.asarray(list(wavy_spiral_record[branch_1]['pos'].values())).min(axis=0)
        xy_max_target = np.asarray(list(wavy_spiral_record[branch_1]['pos'].values())).max(axis=0)

        dx = local_pos_source[0] - local_pos_target[0]
        dy = np.abs(xy_max_target[1] - (xy_min_source[1]-sep)) 
        
        wavy_spiral_record[branch_1]['pos'] = {k: (v+[dx, -dy]) for k,v in wavy_spiral_record[branch_1]['pos'].items()}

        # # determine which corner to add plot to
        # if np.abs(local_pos_source[0] - xy_min_source[0]) < np.abs(local_pos_source[0] - xy_max_source[0]):
        #     # sign_x = -1
        #     delta_x = xy_min_source[0] - xy_max_target[0]
        #     # print('left')
        # else:
        #     # sign_x = 1
        #     delta_x = (xy_max_source[0] - xy_min_target[0])
        #     # print('right')
        # if np.abs(local_pos_source[1] - xy_min_source[1]) < np.abs(local_pos_source[1] - xy_max_source[1]):
        #     # sign_y = -1
        #     delta_y = xy_min_source[1] - xy_max_target[1]
        #     # print('bottom')
        # else:
        #     delta_y = xy_max_source[1] - xy_min_target[1]
        #     # print('top')
        # # update positions
        # wavy_spiral_record[branch_1]['pos'] = {k: (v-[delta_x, delta_y]) for k,v in wavy_spiral_record[branch_1]['pos'].items()}

    # nodelist = []
    POS = {}
    for k,v in wavy_spiral_record.items():
        # nodelist += v['nodelist']
        POS = {**POS, **v['pos']}

    return POS

