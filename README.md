# FIM

This project contains all the models developed in the "Foundation Models for Inference" series of papers.

## Table of Contents
- [FIM](#fim)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

In order to set up the necessary environment:

1. create a virtual environment using your conda or python virtualenv

2. install the project in the virtual environment:
   ```
   pip install -e .
   ```


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.


## Usage

To start training, follow these steps:

1. Make sure you have activated the virtual environment (see [Installation](#installation)).

2. Create a configuration file in YAML format, e.g., `config.yaml`, with the necessary parameters for training.

3. Run the training script, providing the path to the configuration file:
   ```bash
   python scripts/train_model.py --config configs/train/example.yaml
   ```

   This will start the training process using the specified configuration and save the trained model to the specified location.

4. Monitor the training progress and adjust the parameters in the configuration file as needed.

## Contributing

Contributions are welcome! Here's how you can contribute to this project:

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes and commit them:
   ```bash
   git commit -m "Add your commit message here"
   ```

3. Push your changes to your forked repository:
   ```bash
   git push origin feature/your-feature
   ```

4. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/reasoningschema.svg?branch=main)](https://cirrus-ci.com/github/<USER>/reasoningschema)
[![ReadTheDocs](https://readthedocs.org/projects/reasoningschema/badge/?version=latest)](https://reasoningschema.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/reasoningschema/main.svg)](https://coveralls.io/r/<USER>/reasoningschema)
[![PyPI-Server](https://img.shields.io/pypi/v/reasoningschema.svg)](https://pypi.org/project/reasoningschema/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/reasoningschema.svg)](https://anaconda.org/conda-forge/reasoningschema)
[![Monthly Downloads](https://pepy.tech/badge/reasoningschema/month)](https://pepy.tech/project/reasoningschema)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/reasoningschema)
-->
