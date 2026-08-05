# `torch-log-wmse-audio-quality` (alias distribution)

This directory builds the legacy-named distribution as a **metadata-only** package: it ships no code and
simply depends on `torch-log-wmse` at the same version.

## Why

Both names used to be built from the repo's single `setup.cfg`, with the publish workflow rewriting only
the `name` field. That meant **both wheels shipped both package directories and both `RECORD` files
claimed the same 11 payload paths**. pip has no file-ownership arbitration between distributions, so:

* installing one name over the other silently overwrote files the first one owned, defeating hash pinning;
* uninstalling either name deleted the survivor's files while its `.dist-info` and `pip list` entry
  remained, so `pip list` reported the package as installed while `import torch_log_wmse` raised
  `ModuleNotFoundError`.

Reproduced on pip 24.0 and 26.2.1, in both uninstall orders, with no warning from pip and nothing from
`pip check`.

Shipping zero files here removes the overlap entirely. `pip install torch-log-wmse-audio-quality` still
works and still provides both import names, because the real distribution ships both packages.

## Version

The version is injected at build time from `torch_log_wmse/__init__.py`, which stays the single source of
truth. See `.github/workflows/pypi.yml`.
