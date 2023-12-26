import os

import pandas as pd
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

                .. Note::

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

        _fp = Path(os.path.join(file_path, file_name))

        if file_format is None:
            file_format = _fp.suffix.split('.')[1]

        if file_format in ['txt', 'csv', 'tsv']:
            f = pd.read_csv(_fp, sep=delimiter, **kwargs)
        else:
            raise ValueError("Unrecognized file_format.")

        return f
