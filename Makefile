.PHONY: setup-data synthetic-seed test run evaluate evaluate-live

setup-data:
	python scripts/prepare_data.py

synthetic-seed:
	python scripts/create_synthetic_seed.py

test:
	python -m pytest -q

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

evaluate:
	python scripts/evaluate.py

evaluate-live:
	python scripts/evaluate.py --live
