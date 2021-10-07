from setuptools import find_packages, setup
import findspark
findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')

print("start setup")
# setup(
  
#     package_dir={'': 'src'},
#     packages=find_packages(where='src'),
# )