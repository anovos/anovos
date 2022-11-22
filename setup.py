import platform
from os import path

from setuptools import setup

DIR = path.dirname(path.abspath(__file__))

if "arm64" not in platform.version().lower():
    INSTALL_PACKAGES = open(path.join(DIR, "requirements.txt")).read().splitlines()
else:
    INSTALL_PACKAGES = open(path.join(DIR, "requirements_m1.txt")).read().splitlines()

with open(path.join(DIR, "README.md")) as f:
    README = f.read()

setup(
    name="anovos",
    package_dir={"anovos": "src/main/anovos"},
    packages=[
        "anovos",
        "anovos.shared",
        "anovos.data_transformer",
        "anovos.data_ingest",
        "anovos.data_analyzer",
        "anovos.drift_stability",
        "anovos.data_report",
        "anovos.feature_recommender",
        "anovos.feature_store",
    ],
    package_data={"anovos.feature_recommender": ["data/*.csv"]},
    description="An Open Source tool for Feature Engineering in Machine Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=INSTALL_PACKAGES,
    url="https://github.com/anovos/anovos.git",
    author="Team Anovos",
    author_email="info@anovos.ai",
    keywords=[
        "machine learning",
        "open source",
        "feature engineering",
        "analytics",
        "apache spark",
        "feature recommendation",
    ],
    tests_require=["pytest", "coverage"],
    python_requires=">=3.7",
)
