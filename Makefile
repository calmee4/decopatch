.PHONY: test compile check

test:
	python -m pytest -q

compile:
	python -m compileall -q src tests

check: test compile
