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
