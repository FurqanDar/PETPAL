# Positron Emission Tomography Processing and Analysis Library (PETPAL)

<figure>
<img src="docs/PETPAL_logo.png" alt="PETPAL Logo" width="50%">
<figcaption>A comprehensive 4D-PET/MR analysis software suite.</figcaption>
</figure>



## Installation

Currently, we only support building the package directly from source. Clone the repository using your preferred method. After navigating to the top-level directory (where `pyproject.toml` exists), we run the following command in the terminal:

```shell
pip install .  # Installs the package
```

If you are going to be actively developing and making changes to the package source code, it is recommended to instead do:

```shell
pip install -e .  # Installs the package as symlinks to the source code
```

## Documentation
 The official docs are hosted on [read the docs](https://petpal.readthedocs.io/en/latest/), which contain helpful tutorials to get started with using PETPAL, and the API reference. 


### Building Documentation Locally

To generate the documentation in HTML using sphinx, we first navigate to the `$src/docs/` directory. Then, we run the following commands:

```shell
make clean
make html 
```

Then, open `$src/docs/build/html/index.html` using any browser or your IDE.
