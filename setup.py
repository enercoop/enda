"""setup file meant to be used by pip, if this one does not rely on pyproject.toml file"""

import os
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
with open(os.path.join(HERE, "README.md"), "r") as f:
    README = f.read()

# This call to setup() does all the work
setup(
    name="enda",
    version="1.0.0",
    description="Tools to manipulate energy time-series and contracts, and to perform forecasts.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/enercoop/enda",
    author="Enercoop",
    author_email="team-data@enercoop.org",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    project_urls={"Bug Tracker": "https://github.com/enercoop/enda/issues"},
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.9,<3.11",
    install_requires=[
        "datatable~=1.1.0",
        "h2o~=3.36.2.2",
        "jours-feries-france~=0.7.0",
        "pandas~=1.4.0",
        "polars~=0.20.0",
        "pyarrow>=15.0.2",
        "scikit-learn~=1.0.2",
        "statsmodels<=0.13.5",
        "unidecode~=1.3.6",
        "vacances-scolaires-france~=0.10.0"
    ]
)
