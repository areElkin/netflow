# NetFlow

<PUT DESCRIPTION HERE>

## Documentation


NetFlow documentation can be found in `docs/build/html/index.html`
- To open the documentation from the terminal in the `netflow/docs` directory, run ```$ open build/html/index.html```.
- The documentation includes installation instructions and Jupyter notebook tutorials


## Installation


Installation instructions are provided in the documentation and may be
accessed here: [Installation](docs/source/building/index.rst)


## Tutorials

Jupyter notebooks with tutorials on NetFlow usage are provided in the documentation
and may be accessed here: [Tutorials](docs/source/tutorial/notebooks)

An example of the NetFlow pipepline, as performed in the original manuscript
may be accessed here: [MM_tutorial](docs/source/tutorial/notebooks/MM_paper_example.ipynb)


## Acknowledgments


There are many great works that have inspired the development of this package. We would like to acknowledge a few such sources that have contributed  more than just inspiration, either in methodology or code, for their contribution:

- TDA
- OMT
- DPT

This package includes code from [Scanpy](https://github.com/scverse/scanpy), which is licensed under the BSD 3-Clause License. The original license text can be found in the `LICENSE` file included in this package. In particular, the code pertainng to the computation of the diffusion pseudo-time distance (DPT) and branching algorithm proposed in [Haghverdi16]_ incorporated in this package were primiarily sourced or adapted from the Scanpy implementation. We would like to give a special thanks to the contriubtors of that project for providing the underlying functionality for the DPT branching procedure. 


## License Information

This project is licensed under the "BSD 3-Clause License" - see the LICENSE file for details about both licenses used by our project.