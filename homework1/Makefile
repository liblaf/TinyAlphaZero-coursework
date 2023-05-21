CODE      := code
PARAMS    := params
RESULTS   := results
TASK_LIST := task-2 task-3 task-4
ZIP_FILE  := 2020012872_李钦_hw1.zip

FILE_LIST  += $(wildcard $(CODE)/*.py)
FILE_LIST  += $(wildcard $(RESULTS)/*.png)
FILE_LIST  += hw1.pdf
FILE_LIST  += Makefile
FILE_LIST  += README.md
FILE_LIST  += README.pdf
PARAM_LIST += $(PARAMS)/mcts_param_3x3_1.0_500.pkl
PARAM_LIST += $(PARAMS)/mcts_param_3x3_1.5_500.pkl
PARAM_LIST += $(PARAMS)/mcts_param_3x3_2.0_500.pkl

all: $(TASK_LIST)

archive: $(ZIP_FILE)

clean:
	$(RM) --recursive $(PARAMS)
	$(RM) --recursive $(RESULTS)
	$(RM) 2020012872_李钦_hw1.zip

deps: requirements.txt
	pip install --requirement=$<

pretty:
	isort --profile black $(CURDIR)
	black $(CURDIR)
	prettier --write $(CURDIR)

task-2: $(PARAMS)/mcts_param_3x3_1.0_500.pkl $(CODE)/eval.py
	python $(CODE)/eval.py --params $< MCTS Random
	python $(CODE)/eval.py --params $< MCTS AlphaBeta

task-3: $(RESULTS)/MCTS_AlphaBeta.png $(RESULTS)/AlphaBeta_MCTS.png $(RESULTS)/MCTS_Random.png $(RESULTS)/Random_MCTS.png

task-4: $(PARAMS)/mcts_param_3x4_1.0_2000.pkl $(CODE)/eval.py
	python $(CODE)/eval.py --width 3 --height 4 --no-bilateral --params $< MCTS AlphaBeta
	python $(CODE)/eval.py --width 3 --height 4 --no-bilateral --params $< AlphaBeta MCTS
	python $(CODE)/eval.py --width 3 --height 4 --no-bilateral --params $< MCTS Random
	python $(CODE)/eval.py --width 3 --height 4 --no-bilateral --params $< Random MCTS

$(PARAMS) $(RESULTS):
	mkdir --parents $@

$(PARAMS)/mcts_param_3x3_1.0_500.pkl: $(CODE)/train.py | $(PARAMS)
	python $< -C 1.0 --train-iter 500 --param-prefix $(@D)

$(PARAMS)/mcts_param_3x3_1.5_500.pkl: $(CODE)/train.py | $(PARAMS)
	python $< -C 1.5 --train-iter 500 --param-prefix $(@D)

$(PARAMS)/mcts_param_3x3_2.0_500.pkl: $(CODE)/train.py | $(PARAMS)
	python $< -C 2.0 --train-iter 500 --param-prefix $(@D)

$(PARAMS)/mcts_param_3x4_1.0_2000.pkl: $(CODE)/train.py | $(PARAMS)
	python $< --width 3 --height 4 -C 1.0 --train-iter 2000 --param-prefix $(@D)

$(RESULTS)/MCTS_AlphaBeta.png: $(CODE)/task-3.py $(PARAM_LIST) | $(RESULTS)
	python $< -C 1.0 1.5 2.0 --train-iter 500 --param-prefix $(PARAMS) --output $@ MCTS AlphaBeta

$(RESULTS)/AlphaBeta_MCTS.png: $(CODE)/task-3.py $(PARAM_LIST) | $(RESULTS)
	python $< -C 1.0 1.5 2.0 --train-iter 500 --param-prefix $(PARAMS) --output $@ AlphaBeta MCTS

$(RESULTS)/MCTS_Random.png: $(CODE)/task-3.py $(PARAM_LIST) | $(RESULTS)
	python $< -C 1.0 1.5 2.0 --train-iter 500 --param-prefix $(PARAMS) --output $@ MCTS Random

$(RESULTS)/Random_MCTS.png: $(CODE)/task-3.py $(PARAM_LIST) | $(RESULTS)
	python $< -C 1.0 1.5 2.0 --train-iter 500 --param-prefix $(PARAMS) --output $@ Random MCTS

$(ZIP_FILE): $(FILE_LIST)
	zip --from-crlf --recurse-paths --update $@ $^
