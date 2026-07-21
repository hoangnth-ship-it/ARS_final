# Convenience targets (Sec 9). Cross-platform driver is `python run.py <cmd>`.
PY ?= python

manifest:      ; $(PY) run.py manifest
confound:      ; $(PY) run.py confound
features:      ; $(PY) run.py features --which all
baselines:     ; $(PY) run.py baselines
train:         ; $(PY) run.py deep
deep:          ; $(PY) run.py deep
stats:         ; $(PY) run.py stats
sweep:         ; $(PY) run.py sweep
figures:       ; $(PY) run.py figures
model_stats:   ; $(PY) run.py sweep
report:        ; $(PY) run.py report
all:           ; $(PY) run.py all
fast:          ; $(PY) run.py all --fast
test:          ; $(PY) -m pytest -q

.PHONY: manifest confound features baselines train deep stats sweep figures model_stats report all fast test
