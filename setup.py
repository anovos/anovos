from setuptools import find_packages, setup

print("start setup")
setup(
    package_dir={'': 'src/main'},
    name="com.mw.ds.data_ingest", packages=find_packages()
)