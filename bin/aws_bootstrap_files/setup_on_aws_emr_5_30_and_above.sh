aws s3 cp s3://<bucket_name>/requirements.txt .
sudo pip3 install --upgrade pip setuptools wheel --user
sudo pip3 install -r ./requirements.txt
sudo pip3 install --no-deps --force-reinstall matplotlib==3.4.3
