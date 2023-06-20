ARCHIVE := $(CURDIR)/2020012872_李钦_hw3.zip
CODE    := $(CURDIR)/alpha_zero

all:

archive: $(ARCHIVE)

clean:
	@ find $(CURDIR) -type d -name "__pycache__"   | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type d -name ".pytest_cache" | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type f -name ".coverage"     | xargs --no-run-if-empty -- $(RM) --verbose
	@ find $(CURDIR) -type f -name "*.zip"         | xargs --no-run-if-empty -- $(RM) --verbose

pretty: black prettier

test:
	pytest --capture no --cov $(CODE) --cov-report term --cov-report term-missing $(CURDIR)

#####################
# Auxiliary targets #
#####################

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier:
	prettier --write $(CURDIR)

CODE_LIST    != find $(CODE) -name '*.py' -type f
ARCHIVE_LIST += $(CODE_LIST)
ARCHIVE_LIST += $(CURDIR)/hw3.pdf
ARCHIVE_LIST += $(CURDIR)/Makefile
ARCHIVE_LIST += $(CURDIR)/README.md
$(ARCHIVE): $(ARCHIVE_LIST)
	zip --from-crlf --recurse-paths --update $@ $(patsubst $(CURDIR)/%, %, $^)
