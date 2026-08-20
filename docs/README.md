# Build docs

Simply install with the `docs` dependency group:

```sh
uv sync --all-extras --group docs 
```

And then run:

```sh
# cd ./docs
uv run make html
```
