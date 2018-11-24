# Keras Documentation

The source for Keras documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- install MkDocs: `pip install mkdocs`
- `cd` to the `docs/` folder and run:
    - `KERAS_BACKEND=tensorflow python -c 'import keras; keras.docs.generate()'`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
    - `mkdocs build`    # Builds a static site in "site" directory
