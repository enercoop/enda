# Building this package

The official python packaging tutorial (https://packaging.python.org/tutorials/packaging-projects/) 
does not work, at least not easily. `python3 -m build` gave errors reading the example 
`pyproject.toml` and `setup.cfg`.

The package is built using a `setup.py` file following this tutorial: 
https://packaging.python.org/tutorials/packaging-projects/ .  

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

To upload to `test.pypi.org`:
```shell
twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
```

Test to install and use this test package. First create/activate a virtualenv.
Install the latest stable release
```shell
pip install --index-url https://test.pypi.org/simple/ --no-deps enda
```
Or install a specific (not necessarily stable) release like the lastest dev one:
```shell
pip install --index-url https://test.pypi.org/simple/ --no-deps enda==0.1.1.dev5
python  # enter python shell 
```
```python
import pandas as pd
dti = pd.DatetimeIndex([pd.to_datetime('2019-01-01 00:00:00+01:00'), pd.to_datetime('2019-01-01 00:02:00+01:00')])
from enda.timeseries import TimeSeries
TimeSeries.align_timezone(dti, tzinfo='Europe/Paris')
exit()  # exit python shell 
```

Before uploading to the real `pypi.org`, first check that the enda version in `setup.py` is correct.
Follow the official specification for versioning : https://www.python.org/dev/peps/pep-0440/.
Use for instance : `version="0.1.1.dev1"` . 
```shell
twine upload dist/*
```
