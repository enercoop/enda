# Contributing

_This small guide is for the authors of this package._

This package is built and published on PyPi using [poetry](https://python-poetry.org/).

### 1. Check install and package validity 

#### Create a virtual environment


```sh
# for instance, with pyenv 
pyenv virtualenv 3.9.10 publish_enda
pyenv activate publish_enda
```

#### Check the .pyproject.toml file

It is possible to check the .pyproject file running 

```sh
poetry check 
```

and the installation 

```sh
poetry lock  # create the poetry.lock file from the .toml file
poetry install  # install dependencies from the .lock file
poetry install --with dev  # install extra dependencies for dev (for example)
```


#### Run unittests

Make sure all unit tests pass with (they are in the folder `tests`):
```shell
python -m unittest discover tests/
```

### 2. Set the package version

Put the correct version number in `pyproject.toml` file, in the `[tool.poetry]` section. The good practice is to upgrade: 
- the third digit for minor fixes
- the second digit when some enhancement have been performed, but no major has been made
- the first digit for a major change of version, which might even break compatibility. 

Run ```poetry check``` to be sure tye `pyproject.toml` file is alright.

### 3. Build the package

In the virtual environment, run again (because version number changed)

```sh
poetry lock  # this create poetry.lock file from the pyproject.toml
poetry build  # this creates the dist/ folder, which contains a 
              # tar.gz file with the package content, and a .whl file.
```

To make sure the package is fine, it is possible to inspect the dist/ folder, and to run next command in a new virtualenv:

```sh
pip install dist/enda-<version>-py3-none-any.whl
```

all dependencies of the package must be installed out of this command.


### 4. Test package upload using TestPyPI

Before uploading to PyPI, it is relevant to publish the package on [TestPyPI](https://test.pypi.org/). 

By default, poetry does not publish on TestPyPI, and it must be configured to do so. 

```sh
poetry config repositories.test-pypi https://test.pypi.org/legacy/  # add test.pypi in the known repositories  
poetry config --list  # display the config file to check it's been added
```

Uploading to TestPyPI requires to set up a single-use API token (password authentication has been dismissed). To do so: 
- Go to TestPyPI and log in.
- Navigate to your account settings and create a new API token.
- Copy the generated token.

Note that the 2-Factors-Authentication must also be set-up for your account. 

Then, Poetry must be configured to use the API token by setting the `POETRY_PYPI_TOKEN_TEST_PYPI` environment variable:

```sh
export POETRY_PYPI_TOKEN_TEST_PYPI=pypi-<your-token>
```

Finally, to upload to `test.pypi.org`, simply run 
```sh
poetry publish -r test-pypi
```

Check that the package is there : https://test.pypi.org/project/enda/#history .

Try to install and use this test package. Go to another directory, and create/activate a virtualenv.
Use the next commands to install enda from TestPyPI and include dependencies from PyPI:

```
ENDA_NEW_VERSION='1.X.Y'  # Example to be updated to the version only available in test.pypi.org
pip install --extra-index-url https://test.pypi.org/simple/ enda==${ENDA_NEW_VERSION}
```

#### Test with poetry 

Create a new poetry project
```sh 
poetry new enda_test  # this creates a poetry project with a pyproject.toml file
cd enda_test
```

There, we might want to check the package has been correctly uploaded. To do so, it is relevant to modify the .pyproject.toml file: 

```
[[tool.poetry.source]]
name = "test-pypi"
url = "https://test.pypi.org/simple/"
priority = "explicit"

[tool.poetry.dependencies]
python = "^3.9.0"
enda = {version = <version>, source = "test-pypi"}  # this defines the source for the package enda
```

This makes test.pypi the source for enda. Note that only enda is likely to be downloaded, as other packages might not have been put on TestPyPI.

```
poetry lock
poetry install
```

#### Test with pip

Next command installs from pip. 

```sh
pip install --index-url https://test.pypi.org/simple/ enda 
```

Note this must check for hard dependencies. 

#### Test code

Then enter python shell, import enda, and test some enda functions, eg.:

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

### 5. Upload to PyPI

Before uploading to the real `pypi.org`, first double check that the enda version in `pyproject.toml` is correct.

```
poetry check 
poetry lock 
poetry build 
```

and check the content of `dist/`.

The steps are the same as in the previous section except: 
- the API token must be generated from PyPI directly.
- poetry must be configured with the token with ```export POETRY_PYPI_TOKEN_PYPI=pypi-<token>```
- simply run ```poetry publish``` without indicating the repository explicitly.

The new version of the package is now officially released.

Then to test the package just get `enda` using pip without specifying `--index-url`, or poetry without changing the .toml file. 
Hard requirements will be downloaded automatically.

```
pip install enda
```
