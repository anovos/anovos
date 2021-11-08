from setuptools import setup, find_packages

setup(
	name='anovos',
	package_dir={'anovos':'src/main/anovos'},
	packages=['anovos','anovos.shared','anovos.data_transformer','anovos.data_ingest','anovos.data_analyzer','anovos.data_drift','anovos.data_report'],
	description='An Open Source tool for Feature Engineering in Machine Learning',
	version='0.1',
	url='https://gitlab.com/mwengr/mw_ds_feature_machine.git',
	author='mobilewalla',
	author_email='hello@anovos.ai',
	keywords=['machine learning','open source','feature engineering']
)
