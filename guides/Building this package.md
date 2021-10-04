# Building this package

This small guide is for the authors of this package. 

The package is built using a `setup.py` file following this tutorial :
https://packaging.python.org/tutorials/packaging-projects/ .

(The tutorial did not work using the `setup.cfg` file when tried. 
`python -m build` gave errors reading the example `pyproject.toml` and `setup.cfg`. 
Used other ressources for `setup.py` like https://www.freecodecamp.org/news/build-your-first-python-package/ )


### 1. Run unittests 

Make a python >=3.7.3 virtual environment (e.g. with `python3 -m venv {path-to-venv}`. Then :
```shell
source {path-to-venv}/bin/activate
which python  # check python path
```

Install the some dependencies in your virtual environment in order for the tests to succeed : 
```shell
pip install --upgrade pip 
pip install --upgrade pandas scikit-learn joblib h2o statsmodels
```

Run the unittests and check there is no error (they are in the folder `tests`):
```shell
python -m unittest discover tests/
```

### 2. Set the package version

Put the correct version number in `setup.py`. We follow
- https://packaging.python.org/guides/distributing-packages-using-setuptools/#standards-compliance-for-interoperability .
- https://www.python.org/dev/peps/pep-0440/ .

Use for instance : `version="0.0.5"` .

### 3. Build the package

Still in the virtual environment, install packages required to build this package :
```shell
pip install --upgrade twine setuptools wheel
```

Then : 
```shell
rm -r build/ dist/ enda.egg-info/  # clean any previous build
python setup.py sdist bdist_wheel  
twine check dist/*
```
(Some files should have appeared in folders `build/ dist/ enda.egg-info/`).

### 4. Test package using test.pypi.org

To upload to `test.pypi.org` (this should ask you for login information):
```shell
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```


Check that the package is there : https://test.pypi.org/project/enda/#history .

Test to install and use this test package. Go to another directory, and create/activate a virtualenv.
```shell
cd {other-dir}
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Enda requires `pandas` which is only available on the real pypi.org .
```shell
pip install pandas
```

Choose which version of enda you want to install (from test.pypi.org) : 
```shell
pip install --index-url https://test.pypi.org/simple/ enda  # latest stable release
pip install --index-url https://test.pypi.org/simple/ --no-deps enda # latest stable release without the dependencies
pip install --index-url https://test.pypi.org/simple/ enda==0.1.1.dev6 # specific development release
```

Then enter python shell, import enda, and test some enda functions :
```
python  # enter python shell 
```
```python
import pandas as pd
dti = pd.DatetimeIndex([pd.to_datetime('2019-01-01 00:00:00+01:00'), pd.to_datetime('2019-01-01 00:02:00+01:00')])
import enda
enda.TimeSeries.align_timezone(dti, tzinfo='Europe/Paris')
exit()  # exit python shell 
```

### 5. pypi.org

Before uploading to the real `pypi.org`, first double check that the enda version in `setup.py` is correct.

The steps are the same as in the previous section except for :
```shell
twine upload dist/*
```

The new version of the package is now officially released.

Then to test the package just get `enda` using pip without specifying `--index-url`. 
Also, the hard requirements (`pandas`) will be downloaded automatically :
```
pip install enda
```

