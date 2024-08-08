import os

import pandas as pd
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path
from textwrap import dedent

def _docstring_parameter(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """
    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj
    return dec


_desc_distance = """\
Add a symmeric distance matrix to the keeper.
"""

_desc_data_distance = """\
data : {`numpy.ndarray`, `pandas.DataFrame`}
    The distance matrix of size (num_observations, num_observations).
"""

def load_from_file(file_name, file_path=None, file_format=None,
                  delimiter=',', **kwargs):
    
    """ Load data from file.

    .. note::
       Currently loads data using ``pandas.read_csv``.
       Additional formats will be added in the future.

    Parameters
    ----------
    file_name: {`str`, `pathlib.Path`, 
        Input data file name.
    file_path: {`str` `pathlib.Path`}, optional (default: None)
        File path. Empty string by default
    file_format: `str`, optional (default: None)
        File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
        If `None`, ``file_format`` will be inferred from the file extension
        in ``file_name``.
        Currently, this is ignored.
    delimiter: `str`, optional (default: ',')
        Delimiter to use.
    **kwargs
        Additional key-word arguments passed to ``pandas.read_csv``.

    """
    if file_path is None:
        file_path = ''

    if not isinstance(file_name, str):
        file_name = str(file_name)
        
    if file_format is None:
        file_format = file_name.split('.')[-1]
    else:
        file_name = '.'.join([file_name, file_format])
        
    _fp = Path(os.path.join(file_path, file_name))

    # if file_format is None:
    #     file_format = _fp.suffix.split('.')[-1]

    if file_format in ['txt', 'csv', 'tsv']:
        f = pd.read_csv(_fp, sep=delimiter, **kwargs)
    else:
        raise ValueError("Unrecognized file_format.")

    return f


def _local_join_strings(strings, delim=None):
    """ Return delimiter-joined unique strings.

    The resulting string is formed by delimiter-joining the unique
    strings, ordered by the first time they appear.

    Parameters
    ----------
    strings : Iterable(`str`)
        Strings to be joined.
    delim : `str`
        The delimiter used to join the unique strings, default = '_'.
        

    Returns
    -------
    joined_string : `str`
        The delimiter-joined string of unique strings.
    """
    if delim is None:
        delim = "_"
    strings = list(OrderedDict.fromkeys(strings))
    joined_string = delim.join([k for k in strings if k is not None])
    return joined_string


def fuse_labels(labels, delim=None):
    """ Fuse labels

    Each label is expected to be a string consisting of multiple substrings
    concatenated by a delimiter. Each label is split using the delimiter. The fused
    label is constructed by joining the substrings in the same order so that common
    substrings appear once and all label-specific substrings are added ordered by the
    original labels, and is prepended with "fused".

    See below for an example.

    .. note::
       Currently, it is not possible to specify a substring of  underscore-joined
       strings to be treated as a single substring (e.g., "my_feature_description" in
       the label "X_my_feature_description_euc_distance").

       The labels do not need to have the same number of delimited substrings, but
       if they don't, the fused label may not be as expected.

    Parameters
    ----------
    labels : list(str)
        Labels that should be fused.
    delim : str
        The delimiter, default = '_'.

    Returns
    ------
    fused_label : str
        The fused label
    
    Examples
    --------
    >>> fuse_labels(["X_euc_distance", "X_wass_distance"])
    >>> "fused_X_euc_wass_distance"

    """
    # ensure unique labels:
    if len(labels) > len(set(labels)):
        raise ValueError("Duplicate labels detected, all labels must be unique.")

    if delim is None:
        delim = '_'

    fused_label = [_local_join_strings(strings, delim=delim) \
                   for strings in zip_longest(*[k.split(delim) for k in labels])]

    fused_label = delim.join(["fused"] + fused_label)
    return fused_label
    
    
