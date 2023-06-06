MAKEFLAGS += --jobs

ASSETS  := $(CURDIR)/assets
CODE    := $(CURDIR)/code
RESULTS := $(CURDIR)/results
ARCHIVE := $(CURDIR)/2020012872_李钦_hw2.zip

all: archive

archive: $(ARCHIVE)

clean:
	@ $(RM) --verbose $(ARCHIVE)
	# $(RM) --recursive $(ASSETS)
	# $(RM) --recursive $(RESULTS)

pretty: black prettier

test:
	pytest --cov=$(CODE)/alphazero --cov-report=term --cov-report=term-missing $(CURDIR)

################################################################
#                      Auxiliary Targets                       #
################################################################

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier:
	prettier --write $(CURDIR)

$(ASSETS)/policy_gradient:
	@ mkdir --parents --verbose $@

ASSET_LIST   += $(ASSETS)/policy_gradient/loss.png
ASSET_LIST   += $(ASSETS)/policy_gradient/reward.png
CODE_LIST    != find $(CODE) -name '*.py' -type f
ARCHIVE_LIST += $(CODE_LIST)
ARCHIVE_LIST += $(CURDIR)/hw2.pdf
ARCHIVE_LIST += $(CURDIR)/Makefile
ARCHIVE_LIST += $(CURDIR)/README.md
ARCHIVE_LIST += $(CURDIR)/README.pdf
$(ARCHIVE): $(ARCHIVE_LIST)
	zip --from-crlf --recurse-paths --update $@ $(patsubst $(CURDIR)/%, %, $^)

POLICY_GRADIENT_RESULT_LIST += $(RESULTS)/policy_gradient/0.txt
POLICY_GRADIENT_RESULT_LIST += $(RESULTS)/policy_gradient/1.txt
POLICY_GRADIENT_RESULT_LIST += $(RESULTS)/policy_gradient/2.txt
$(ASSETS)/policy_gradient/loss.png $(ASSETS)/policy_gradient/reward.png: $(CODE)/policy_gradient/plot.py $(POLICY_GRADIENT_RESULT_LIST) | $(ASSETS)/policy_gradient
	python $< --input $(POLICY_GRADIENT_RESULT_LIST) --seed 0 1 2 --prefix $(@D)

$(RESULTS)/policy_gradient:
	@ mkdir --parents --verbose $@

$(RESULTS)/policy_gradient/%.txt: $(CODE)/policy_gradient/policy_gradient.py | $(RESULTS)/policy_gradient
	python $< --output $@ --seed $*
