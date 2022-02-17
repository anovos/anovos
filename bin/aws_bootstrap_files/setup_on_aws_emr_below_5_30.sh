aws s3 cp s3://<bucket_name>/requirements.txt .
sudo pip-3.7 install --upgrade pip setuptools wheel --user
sudo pip-3.7 install --upgrade cython
sudo pip-3.7 install -r ./requirements.txt
sudo pip-3.7 install --no-deps --force-reinstall matplotlib==3.4.3
