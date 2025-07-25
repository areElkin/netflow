# NetFlow

<PUT DESCRIPTION HERE>

## Documentation

Full documentation, including installation instructions, tutorials, and API reference, is available at:
[https://areElkin.github.io/netflow](https://areElkin.github.io/netflow)

## Installation

Installation instructions can be found in:

- Documentation: [Installation](https://areElkin.github.io/netflow/building/index.html),
- Local source file: [`docs/source/building/index.rst`](docs/source/building/index.rst)

## Tutorials

Jupyter notebooks demonstrating NetFlow usage are provided in:

- Documentation: [Tutorials](https://areElkin.github.io/netflow/tutorial/index.html)
- Local source file: [`docs/source/tutorial/notebooks`](docs/source/tutorial/notebooks)

The full workflow reproducing the NetFlow pipeline performed on multiple myeloma (MM) from the original manuscript may be accessed from:

- Documentation: [MM_workflow](https://areElkin.github.io/netflow/tutorial/notebooks/MM_paper_example.html)
- Local source file: [`docs/source/tutorial/notebooks/MM_paper_example.ipynb`](docs/source/tutorial/notebooks/MM_paper_example.ipynb)

## Acknowledgments

We would like to acknowledge several foundational works that contributed to the NetFlow methodology, either in inspiration or code, used in this library:

- Topological data analysis (TDA)
- Optimal mass transport (OMT)
- Diffusion pseudotime (DPT)

This package includes code from [Scanpy](https://github.com/scverse/scanpy), licensed under the BSD 3-Clause License. The original license text can be found in the `LICENSE` file included in this package. In particular, the implementation of the DPT distance and branching algorithm, originally described by Haghverdi et al. (2016), was primarily sourced or adapted from the Scanpy codebase. We are grateful to the Scanpy contributors for providing the basis for this functionality.

## License

This project is licensed under the "BSD 3-Clause License" - see the [`LICENSE`](LICENSE.txt) file for details.

## Cite

Please cite our paper if you use this code in your work.

```
@article{elkin2025netflow,
  title={NetFlow: a framework to explore topological representations of high-dimensional biomedical data},
  author={Elkin, Rena and Oh, Jung Hun and Simhal, Anish K and Deasy, Joseph O},
  year={2025},
}
```