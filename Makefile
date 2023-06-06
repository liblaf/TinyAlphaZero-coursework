all:

install:
	poetry install
	conda install --channel conda-forge --yes gym==0.19.0

test:
	pytest --cov=$(CURDIR) --cov-report=term --cov-report=term-missing --cov-report=xml
