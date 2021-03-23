# Building this package

The official python packaging tutorial (https://packaging.python.org/tutorials/packaging-projects/) 
does not work, at least not easily. `python3 -m build` gave errors reading the example 
`pyproject.toml` and `setup.cfg`.

The package is built using a `setup.py` file following this tutorial: 
https://packaging.python.org/tutorials/packaging-projects/ .  

First put the correct version number in `setup.py`. We follow 
https://packaging.python.org/guides/distributing-packages-using-setuptools/#standards-compliance-for-interoperability .
https://www.python.org/dev/peps/pep-0440/ .
Use for instance : `version="0.1.1.dev1"` .

To build, make a python >=3.7.3 venv. Then:
```shell
source {path-to-venv}/bin/activate
pip install --upgrade pip twine setuptools
```

```shell
rm -r build/ dist/ enda.egg-info/  # clean any previous build
python setup.py sdist bdist_wheel  
twine check dist/*
```

## Test.pypi.org

To upload to `test.pypi.org`:
```shell
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Check that the package is there: https://test.pypi.org/project/enda/#history .

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

Then enter python shell and import enda
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

## Pypi.org 

Before uploading to the real `pypi.org`, first check that the enda version in `setup.py` is correct.

The steps are the same as in the previous section except for :
```shell
twine upload --repository-url https://pypi.org/ dist/*
```
Then to test the package just get `enda` using pip without specifying `--index-url`. 
Also, the hard requirements (`pandas`) will be downloaded automatically:
```
pip install enda
```

