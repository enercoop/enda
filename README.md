# enda


## What is it?

**enda** is a Python package that provides tools to manipulate **timeseries** data in conjunction with **contracts** data for analysis and **forecasts**. 

Its main goal is to help [Rescoop.eu](https://www.rescoop.eu/) members build various applications, such as short-term electricity load and production forecasts, specifically for the [RescoopVPP](https://www.rescoopvpp.eu/) project. Hence some tools in this package perform TSO (transmission network operator) and DNO (distribution network operator) data wrangling as well as weather data management. enda is mainly developed by [Enercoop](https://www.enercoop.fr/).

## Main Features
Here are some things **enda** does well :

  - Provide robust machine learning algorithms for short-term electricty load (and soon production) forecasts, developed by Enercoop and originally based on Komi Nagbe's thesis (http://www.theses.fr/s148364).
  - Manipulate **contracts** data coming from your ERP and turn it into timeseries you can use for analysis, visualisation and machine learning.  
  - Timeseries-specific detection of missing data, like time gaps and frequency changes.
  - Date-time feature engineering robust to timezone hazards.

## Where to get it
The source code is currently hosted on GitHub at : https://github.com/enercoop/enda

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/enda) (for now it is not directly on [Conda](https://docs.conda.io/en/latest/)).

```sh
# PyPI
pip install enda
```

## How to get started ?

Check out the guides : https://github.com/enercoop/enda/tree/main/guides .

## Hard dependencies
- [Pandas - the main dataframe manipulation tool for python, advanced timeseries management included.](https://pandas.pydata.org/)
- Pandas itself has hard dependencies and optional dependencies, checkout https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html . Hard dependencies of pandas include : `setuptools`, `NumPy`, `python-dateutil`, `pytz`.

## Optional dependencies 

Optional dependencies are used only for specific methods. Enda will give an error if the method called requires a dependency that is not installed. 

Enda can work with different machine learning "backends" :
- [Scikit-learn](https://scikit-learn.org/stable/)
- [H2O - an efficient machine learning framework](https://docs.h2o.ai/)

You can also easily implement your own ml-backend by implementing enda's ModelInterface. Checkout `enda.ml_backends.sklearn_linreg.py` for an example with `SKLearnLinearRegression`. 

Other optional dependencies :
- [statsmodel](https://pypi.org/project/statsmodels/)

Furthermore, don't hesitate to install pandas "Recommended dependencies" for speed-ups : `numexpr` and `bottleneck`.

If you want to save your trained models, we recommend `joblib`. See Scikit-learn's recommendations here : https://scikit-learn.org/stable/modules/model_persistence.html . 

An almost complete install looks like :
```
pip install numexpr bottleneck pandas enda jupyter h2o scikit-learn statsmodels joblib matplotlib
```


## License
[MIT](LICENSE)
