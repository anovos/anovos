help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "build - package"

all: default

default: clean dev_deps deps test build


clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf Dockerfile
	rm -fr dist/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -rf ./data/tmp/output 

test:
	pytest src/test

build: clean
	cp ./docker_build/Dockerfile ./Dockerfile
	rm -rf ./dist && mkdir ./dist && mkdir ./dist/data && mkdir ./dist/output
	cp ./src/main/main.py ./dist
	cp ./config/configs.yaml ./dist
	cp -rf ./src/main/com ./dist/com
	cd ./dist && zip -r com.zip spark.py ./com
	cp -rf ./data/income_dataset ./dist/data/income_dataset
	cp -rf ./data/data_report/data_dict ./dist/output/data_dict
	cp -rf ./data/data_report/feature_mp ./dist/output/feature_mp
	cp -rf ./data/data_report/metric_dict ./dist/output/metric_dict
	pip3 install -r requirements.txt
	cp ./bin/spark-submit.sh ./dist
