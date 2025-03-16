# mmcontext

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

Align embeddings across multiple modalities using context-aware embeddings at the sample level.

**mmcontext** is built upon the excellent [sentence-transformers](https://www.sbert.net/) framework maintained by [Hugging Face](https://huggingface.co/). By leveraging their comprehensive documentation and extensive capabilities for text embeddings, **mmcontext** enables you to efficiently generate multi-modal embeddings without reinventing the wheel.

![Conceptual Diagram](figs/concept.png)

## Getting Started

Please refer to our [documentation](https://mmcontext.readthedocs.io), especially the [API documentation](https://mmcontext.readthedocs.io/en/latest/api.html). This project was built using the amazing [scverse cookie cutter template](https://github.com/scverse/cookiecutter-scverse)—check it out!

## Installation

Ensure you have **Python 3.10** or newer installed.

Currently, `mmcontext` is available for installation directly from the GitHub source:

```bash
pip install git+https://github.com/mengerj/mmcontext.git@main

# If sentence-transformers dependency is outdated, use:
pip install git+https://github.com/mengerj/sentence-transformers.git@master
```

_(PyPI release coming soon!)_

## Contributing

This package is under active development. Contributions and suggestions are very welcome—please open an [issue](https://github.com/mengerj/mmcontext/issues) to propose enhancements, report bugs, or discuss potential improvements.

## Release Notes

Review the latest changes in the [Changelog](https://mmcontext.readthedocs.io/en/latest/changelog.html).

## Contact

Encountered a bug or need help? Please use the [issue tracker](https://github.com/mengerj/mmcontext/issues).

## Citation

If you find **mmcontext** useful for your research, please consider citing it as follows until an official publication is available:

```bibtex
@misc{mmcontext,
  author = {Your Name},
  title = {mmcontext: Multi-modal Contextual Embeddings},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  url = {https://github.com/mengerj/mmcontext}
}
```

---

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mengerj/mmcontext/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mmcontext
[tests]: https://github.com/mengerj/mmcontext/actions/workflows/test.yml
[documentation]: https://mmcontext.readthedocs.io
