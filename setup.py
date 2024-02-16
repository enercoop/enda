import pathlib
from setuptools import setup, find_packages
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
with open(os.path.join(HERE, "README.md"), "r") as f:
    README = f.read()

# This call to setup() does all the work
setup(
    name="enda",
    version="0.0.8",
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
    python_requires=">=3.7.3,<3.11",
    install_requires=[
        "pandas>=1.0.0,<2.0.0",
        "h2o~=3.42.0.4",
        "scikit-learn~=1.0.2",
        "statsmodels<=0.13.5",
        "bottleneck~=1.3.7",
        "joblib~=1.3.0",
        "jours-feries-france~=0.7.0",
        "numexpr~=2.8.4",
        "numpy~=1.18",
        "unidecode~=1.3.6",
        "vacances-scolaires-france~=0.10.0"
    ]
)
