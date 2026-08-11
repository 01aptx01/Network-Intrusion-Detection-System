.PHONY: clean data train test lint

python = python

data:
	$(python) -c "from src.data.preprocessing import synthesize_fallback_dataset; print('Dataset ready')"

train:
	$(python) main.py

test:
	$(python) -m pytest

lint:
	$(python) -m flake8 src tests

clean:
	rm -rf __pycache__ .pytest_cache *.pyc
