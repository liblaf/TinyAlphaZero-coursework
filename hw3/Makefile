ARCHIVE := $(CURDIR)/2020012872_李钦_hw3.zip
CODE    := $(CURDIR)/alpha_zero
OUTPUT  := $(CURDIR)/output

PYTHON_FLAGS ?= # -OO
PYTHON       := python $(PYTHON_FLAGS)

MULTIPROCESSING ?= True

all:

anylearn: $(CODE)/exp_anylearn.py $(CURDIR)/requirements.txt
ifeq ($(BW_SESSION),)
	$(error Bitwarden Locked)
else
	$(MAKE) clean
ifeq ($(MULTIPROCESSING), True)
	$(PYTHON) $< --username "$(shell bw get username Anylearn)" --password "$(shell bw get password Anylearn)" --multiprocessing
else
	$(PYTHON) $< --username "$(shell bw get username Anylearn)" --password "$(shell bw get password Anylearn)"
endif
	$(MAKE) clean
endif

archive: $(ARCHIVE)

clean:
	@ find $(CURDIR) -type d -name "__pycache__"   | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type d -name ".pytest_cache" | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type f -name ".coverage"     | xargs --no-run-if-empty -- $(RM) --verbose
	@ find $(CURDIR) -type f -name "*.zip"         | xargs --no-run-if-empty -- $(RM) --verbose
	$(RM) --recursive $(CURDIR)/.git
	$(RM) --recursive $(OUTPUT)

pretty: black prettier

profile: $(CURDIR)/train.profile
	speedscope $<

test:
	pytest --capture no --cov $(CODE) --cov-report term --cov-report term-missing $(CURDIR)

train:
ifeq ($(MULTIPROCESSING), True)
	$(PYTHON) -m alpha_zero.main --multiprocessing
else
	$(PYTHON) -m alpha_zero.main
endif

#####################
# Auxiliary targets #
#####################

ALWAYS:

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

$(CURDIR)/train.profile: ALWAYS
	py-spy record --output $@ --format speedscope --duration 8 -- python -m alpha_zero.main

$(CURDIR)/requirements.txt: $(dir $(CURDIR))/poetry.lock
	poetry export --output $@ --without-hashes --without-urls --without pytorch
