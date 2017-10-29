# Keras Documentation

The source for Keras documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- follow instructions of installing keras from github in the root `README.md`
- install at least one backend, e.g. `pip install tensorflow`
- install MkDocs: `pip install mkdocs`
- `cd` to the `docs/` folder and run:
    - `python autogen.py`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
    - `mkdocs build`    # Builds a static site in "site" directory
