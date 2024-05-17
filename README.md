# enda

![PyPI](https://img.shields.io/pypi/v/enda) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## What is it?

**enda** is a Python package that provides tools to manipulate **timeseries** data in conjunction with **contracts** data for analysis and **forecasting**.

Initially, it has been developed to help [Rescoop.eu](https://www.rescoop.eu/) members build various applications, such as short-term electricity load and production forecasts, specifically for the [RescoopVPP](https://www.rescoopvpp.eu/) project. Hence some tools in this package perform TSO (transmission network operator) and DNO (distribution network operator) data wrangling as well as weather data management. enda is mainly developed by [Enercoop](https://www.enercoop.fr/).

## Main Features

Here are some things **enda** does well:

- Provide robust machine learning algorithms for **short-term electricity load and production forecasts**. enda provides a convenient wrapper around the popular multipurpose machine-learning backends [Scikit](https://scikit-learn.org/stable/) and [H2O](https://h2o.ai/platform/ai-cloud/make/h2o/). The load forecast was originally based on Komi Nagbe's thesis (<http://www.theses.fr/s148364>).
- Manipulate **timeseries** data, such as load curves. enda handles timeseries-specific detection of missing data, like time gaps, frequency changes, extra values, as well as various resampling methods. 
- Provide several **backtesting** and **scoring** methods to ensure the quality of the trained algorithm on almost real conditions.
- Manipulate **contracts** data coming from your ERP and turn it into timeseries you can use for analysis, visualisation and machine learning.
- Date-time **feature engineering** robust to timezone hazards.

## Where to get it

The source code is currently hosted on GitHub at: <https://github.com/enercoop/enda>. If you wish to run the examples it contains, you can clone enda from the Github repository

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/enda) (for now it is not directly on [Conda](https://docs.conda.io/en/latest/)). 


```sh
pip install enda
```

or using [poetry](https://python-poetry.org/):

```sh
poetry add enda
```

## Documentation and API

The complete API is available online [here](https://enercoop.github.io/enda).


## How to get started?

For a more comprehensive approach to enda, several [Jupyter notebooks](https://jupyter.org/) have been proposed in the [guides](<https://github.com/enercoop/enda/tree/main/guides>.).
Some dependencies are needed to run these examples, that you can easily install with poetry, running ```poetry install enda[examples]```


## Dependencies

### Hard dependencies

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [H2O](https://docs.h2o.ai/)
- [Numpy](https://numpy.org/)
- [Statsmodels](https://pypi.org/project/statsmodels/)
- Libraries that are recommended by the previous packages: [datatable](https://pypi.org/project/datatable/), [polars](https://pypi.org/project/polars/), [numexpr](https://pypi.org/project/numexpr/), [unidecode](https://pypi.org/project/Unidecode/)
- Libraries meant to get calendar data: [jours-feries-france](https://pypi.org/project/jours-feries-france/), [vacances-scolaires-france](https://pypi.org/project/vacances-scolaires-france/)


### Optional dependencies

If you want to run the examples, you may need extra dependencies. 
These dependencies can be installed using poetry: 

```sh
poetry install --with examples
```

or manually:

```sh
pip install numexpr bottleneck pandas enda jupyter h2o scikit-learn statsmodels joblib matplotlib
```

Accordingly, if you wish to develop into enda, we suggest some tools and linters that can be used. 
```sh
poetry install --with dev
```

## License

[MIT](LICENSE)
