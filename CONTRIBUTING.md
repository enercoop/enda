# Contributing

The project can be built using [Poetry](https://python-poetry.org/). You can install it following the instructions here: <https://python-poetry.org/docs/#installation>.

To develop `enda`, you can clone the repository and install the dependencies with poetry:

```sh
git clone https://github.com/enercoop/enda.git
cd enda
poetry install --with dev
```

If you are not using `Poetry` you can install the dependencies with `pip`:

```sh
pip install enda[dev]
```

To run the tests, you can use the following command:

```sh
poetry run pytest # or just pytest if you have activated the virtual environment
```

To run tests using tox, you can use the following command:

```sh
poetry run tox # or just tox if you have activated the virtual environment
```

## Building and publishing

After you have built a new feature, upgraded a dependency or fixed a bug, you need to update the version number in `pyproject.toml` and `enda/__init__.py` and publish the new version to PyPI.

If you are using `Poetry`, you can add the poetry plugin `poetry-bumpversion` to help you with this:

```sh
poetry self add poetry-bumpversion
```

and then you can use the following commands to update the version number:

```sh
poetry version patch # or minor or major
```

See the [`poetry-bumpversion`](https://pypi.org/project/poetry-bumpversion/) documentation for more details:

After that you will need to build the package with:

```sh
poetry build
```

and then you can publish the new version to PyPI:

```sh
poetry publish
```

You will need to setup your PyPI credentials for this to work. See the [Poetry documentation](https://python-poetry.org/docs/repositories/#configuring-credentials) for more details.

