aws s3 cp s3://<bucket_name>/requirements.txt .
sudo pip3 install --upgrade cython
sudo pip3 install --upgrade pip setuptools wheel --user
sudo pip3 install -r ./requirements.txt
sudo pip3 install --upgrade --no-deps --force-reinstall scikit-learn
sudo pip3 install numpy==1.18.1
sudo pip3 install pandas==1.2.5
sudo pip3 install tensorflow
sudo pip3 install --no-deps --force-reinstall matplotlib==3.4.3
