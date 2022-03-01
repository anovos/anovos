aws s3 cp s3://<bucket_name>/requirements.txt .
sudo pip-3.6 install --upgrade cython
sudo pip-3.6 install --upgrade pip-3.6 setuptools wheel --user
sudo pip-3.6 install -r ./requirements.txt
sudo pip-3.6 install --upgrade --no-deps --force-reinstall scikit-learn
sudo pip-3.6 install numpy==1.18.1
sudo pip-3.6 install pandas==1.1.5
sudo pip-3.6 install tensorflow
sudo pip-3.6 install --no-deps --force-reinstall matplotlib==3.3.4
