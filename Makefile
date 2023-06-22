all:

clean:
	@ find $(CURDIR) -type d -name "__pycache__"   | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type d -name ".pytest_cache" | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type f -name ".coverage"     | xargs --no-run-if-empty -- $(RM) --verbose
	@ find $(CURDIR) -type f -name "coverage.xml"  | xargs --no-run-if-empty -- $(RM) --verbose

install:
	poetry install
	conda install --channel conda-forge --yes gym==0.19.0

test:
	pytest --ignore-glob='**/run' --cov=$(CURDIR) --cov-report=term --cov-report=term-missing --cov-report=xml
