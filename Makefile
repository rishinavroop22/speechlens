# SpeechLens - common tasks.  `make help` lists them.
PY ?= python

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*## /\t/'

setup:           ## install Python dependencies
	$(PY) -m pip install -r requirements.txt

model:           ## download the alignment model (378 MB)
	$(PY) tools/get_model.py

serve:           ## run the dashboard on http://localhost:8000
	uvicorn speechlens.app:app --port 8000

test:            ## unit tests
	$(PY) -m pytest -q tests

eval:            ## grounding evaluation -> results/evaluation.json
	$(PY) tools/evaluate.py

experiments:     ## ablations, counterfactual check, stress test, statistics
	$(PY) tools/ablate.py
	$(PY) tools/counterfactual_eval.py
	$(PY) tools/stress.py
	$(PY) tools/stats.py

reproduce:       ## rebuild dataset and every number from the raw speeches
	sh reproduce.sh

static:          ## build the hosted static demo into build/static
	$(PY) tools/make_space.py && $(PY) tools/make_static.py

docker:          ## build and run the Docker image
	docker build -t speechlens . && docker run -p 8000:8000 speechlens

.PHONY: help setup model serve test eval experiments reproduce static docker
