# enda


## What is it?

**enda** is a Python package that provides tools to manipulate **timeseries** data in conjunction 
with **contracts** data for analysis and **forecasts**. Its main goal is assist Rescoop.eu[https://www.rescoop.eu/] with various applications, such as short-term electricity load and production forecasts, specifically for the RescoopVPP[https://www.rescoopvpp.eu/] project. Hence some tools in this package perform TSO (transmission network operator) and DNO (distribution network operator) data wrangling for several European countries, as well as weather data. 

## Main Features
Here are some things **enda** does well:

  - Provide robust machine learning algorithms for short-term electricty load (and soon production) forecasts, originally based on Komi Nagbe's thesis (http://www.theses.fr/s148364).
  - Manipulate **contracts** data coming from your ERP and turn it into timeseries you can use for analysis, visualisation and machine learning.  
  - Timeseries-specific detection of missing data, like time gaps and frequency changes.
  - Date-time feature engineering robust to timezone hazards.  


## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/enercoop/enda

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/enda) and on [Conda](https://docs.conda.io/en/latest/).

```sh
# conda
conda install pandas
```

```sh
# or PyPI
pip install pandas
```

## Dependencies
- [Pandas - the main dataframe manipulation tool for python, advanced timeseries management included.](https://pandas.pydata.org/)
- [H2O - an efficient machine learning framework.](https://docs.h2o.ai/)

## License
[MIT](LICENSE)


#### Building this package

Go into a python >=3.7.3 venv
pip install --upgrade pip
pip install twine
python setup.py sdist bdist_wheel
 
twine check dist/*     # or:  tar tzf enda-0.1.0.tar.gz
twine upload --repository-url https://test.pypi.org/legacy/ dist/*




