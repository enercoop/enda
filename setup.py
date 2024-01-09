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
    python_requires=">=3.7.3",
    install_requires=["pandas>=1.0.0"],
)
