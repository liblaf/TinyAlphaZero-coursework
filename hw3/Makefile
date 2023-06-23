ARCHIVE := $(CURDIR)/2020012872_李钦_hw3.zip
CODE    := $(CURDIR)/alpha_zero
OUTPUT  := $(CURDIR)/output
RUN     := $(CURDIR)/run

CUDA_VISIBLE_DEVICES ?= 0
MULTIPROCESSING      ?= True
PYTEST_FLAGS         ?= --capture no
PYTHON_FLAGS         ?= # -OO
export CUDA_VISIBLE_DEVICES
export MULTIPROCESSING
export PYTEST_FLAGS
export PYTHON_FLAGS

PYTHON       := env CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) python $(PYTHON_FLAGS)
PYTEST       := pytest $(PYTEST_FLAGS)

all:

anylearn: $(CURDIR)/requirements.txt
ifeq ($(BW_SESSION),)
	$(error Bitwarden Locked)
else
ifeq ($(MULTIPROCESSING), True)
	$(PYTHON) -m alpha_zero.exp_anylearn --username "$(shell bw get username Anylearn)" --password "$(shell bw get password Anylearn)" --multiprocessing
else
	$(PYTHON) -m alpha_zero.exp_anylearn --username "$(shell bw get username Anylearn)" --password "$(shell bw get password Anylearn)" --no-multiprocessing
endif
endif

archive: $(ARCHIVE)

clean:
	@ find $(CURDIR) -type d -name "__pycache__"   | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type d -name ".pytest_cache" | xargs --no-run-if-empty --verbose -- $(RM) --recursive
	@ find $(CURDIR) -type f -name ".coverage"     | xargs --no-run-if-empty -- $(RM) --verbose
	@ find $(CURDIR) -type f -name "*.zip"         | xargs --no-run-if-empty -- $(RM) --verbose
	$(RM) --recursive $(CURDIR)/.git
	# $(RM) --recursive $(OUTPUT)
	# $(RM) --recursive $(RUN)

pretty: black prettier

profile: $(CURDIR)/train.profile
	speedscope $<

TASK_ID        := $(shell date --iso-8601=seconds | tr ':' '-')
TASK_WORKSPACE := $(RUN)/$(TASK_ID)
submit: | $(RUN)
	mkdir --parents --verbose $(TASK_WORKSPACE)
	install -D --no-target-directory --verbose $(CURDIR)/Makefile $(TASK_WORKSPACE)/Makefile
	cp --recursive $(CODE) $(TASK_WORKSPACE)
	tmux new-session -d -c $(TASK_WORKSPACE) -s $(TASK_ID) '$(MAKE) $(MAKEFLAGS) train'
	tmux pipe-pane -o -t $(TASK_ID) 'cat > $(TASK_WORKSPACE)/train.log'

test:
	$(PYTEST) --cov=$(CODE) --cov-report=term --cov-report=term-missing $(CODE)

train:
ifeq ($(MULTIPROCESSING), True)
	$(PYTHON) -m alpha_zero.main --multiprocessing
else
	$(PYTHON) -m alpha_zero.main --no-multiprocessing
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

$(RUN):
	@ mkdir --parents --verbose $@
