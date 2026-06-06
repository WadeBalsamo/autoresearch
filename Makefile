.PHONY: help install install-gpu data test smoke \
        cls nsp gen synth validate-synth sweep-cls sweep-nsp sweep-gen leaderboard clean

DATA ?= ./data
QRA  ?= ../Qualitative_Research_Algorithm
QRA_OUT ?= $(QRA)/data/output
OPUS ?= anthropic/claude-opus-4.8

help:
	@echo "make install        # CPU-only core (common/synth/optimize/tests)"
	@echo "make install-gpu    # + torch/transformers/peft/bitsandbytes (the 3090 box)"
	@echo "make data           # pull QRA exports into $(DATA)   (QRA_OUT=...)"
	@echo "make test           # torch-free unit tests"
	@echo "make cls|nsp|gen    # train one track (writes run.log)"
	@echo "make synth          # generate synthetic cue blocks via Claude Opus (needs OPENROUTER_API_KEY)"
	@echo "make sweep-cls|...  # hands-off hyperparameter sweep"
	@echo "make leaderboard    # render leaderboard.md"

install:
	uv sync

install-gpu:
	uv sync --extra gpu

data:
	python scripts/pull_qra_data.py --qra-output $(QRA_OUT) --dest $(DATA)

test:
	python -m pytest -q

smoke:
	uv run python tracks/classification/prepare.py --data-dir $(DATA)

cls:
	uv run python tracks/classification/train.py --data-dir $(DATA) 2>&1 | tee run.log

nsp:
	uv run python tracks/nsp/train.py --data-dir $(DATA) 2>&1 | tee run.log

gen:
	uv run python tracks/generative/train.py --data-dir $(DATA) 2>&1 | tee run.log

synth:
	python -m synth.generate --data-dir $(DATA) --n 2000 --model $(OPUS) --qra-repo $(QRA)

validate-synth:
	python -m synth.validate --data-dir $(DATA)

sweep-cls:
	python -m optimize.search --track classification --data-dir $(DATA) --trials 20

sweep-nsp:
	python -m optimize.search --track nsp --data-dir $(DATA) --trials 20

sweep-gen:
	python -m optimize.search --track generative --data-dir $(DATA) --trials 10 --timeout 2400

leaderboard:
	python -m optimize.leaderboard

clean:
	rm -rf runs/ checkpoints/ adapters/ run.log tracks/*/_sweep_train.py __pycache__ \
	       */__pycache__ */*/__pycache__ .pytest_cache
