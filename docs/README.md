# Keras Documentation

The source for Keras documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- install MkDocs: `pip install mkdocs`
- `pip install -e .` to make sure that Python will import your modified version of Keras.
- `cd` to the `docs/` folder and run:
    - `KERAS_BACKEND=tensorflow python autogen.py`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
    - `mkdocs build`    # Builds a static site in "site" directory
