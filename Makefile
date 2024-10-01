.PHONY: install test

default: test

install:
	pip install -e .

test:
	PYTHONPATH=./ pytest -s

lr:
	PYTHONPATH=./ pytest -s ./tests/test_linear_regression.py
	
features:
	PYTHONPATH=./ pytest -s ./tests/test_features.py