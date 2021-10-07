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
	cp -rf ./data/data_report/stability/ ./dist/output/stability/
	cd ./dist && zip -r com.zip ./com
	cd ./dist && tar -cvzf com.tar.gz ./com
	cp -rf ./data/income_dataset ./dist/data/income_dataset
	pip3 install -r requirements.txt --no-cache-dir
	cp ./bin/remove_overheads.sh ./dist
	cp ./bin/spark-submit.sh ./dist
