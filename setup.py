from os import path

from setuptools import setup

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name='anovos',
    package_dir={'anovos': 'src/main/anovos'},
    packages=['anovos', 'anovos.shared', 'anovos.data_transformer', 'anovos.data_ingest', 'anovos.data_analyzer',
              'anovos.data_drift', 'anovos.data_report'],
    description='An Open Source tool for Feature Engineering in Machine Learning',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
    version='0.1',
    url='https://github.com/anovos/anovos.git',
    author='Team Anovos',
    author_email='info@anovos.ai',
    keywords=['machine learning', 'open source', 'feature engineering', 'analytics'],
    tests_require=[
        'pytest',
        'coverage'
    ],
    python_requires='>=3.7'
)
