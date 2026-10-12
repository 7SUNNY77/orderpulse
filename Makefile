# OrderPulse pipeline. Each target has an equivalent `python -m orderpulse <cmd>`.
# Activate the virtual environment first, or override PY, e.g. `make PY=.venv/Scripts/python test`.
PY ?= python

.PHONY: ingest quality marts train evaluate predict pipeline test app

ingest:
	$(PY) -m orderpulse ingest

quality:
	$(PY) -m orderpulse quality

marts:
	$(PY) -m orderpulse marts

train:
	$(PY) -m orderpulse train

evaluate:
	$(PY) -m orderpulse evaluate

predict:
	$(PY) -m orderpulse predict

# All computational steps: ingest -> quality -> marts -> train -> evaluate -> predict
pipeline:
	$(PY) -m orderpulse pipeline

test:
	$(PY) -m pytest

app:
	$(PY) -m orderpulse app
