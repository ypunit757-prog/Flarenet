.PHONY: install lint format test test-fast run fast experiment nasa docker clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

test-fast:
	pytest -m "not slow"

run:
	flarenet run

fast:
	flarenet run -c configs/fast.yaml

experiment:
	flarenet experiment -e configs/experiments/stage3_final.yaml --out runs/stage3 --jobs 2

# Real NASA/NOAA GOES data (needs internet and the netcdf extra)
nasa:
	pip install -e ".[netcdf]"
	flarenet run -c configs/nasa_goes_ncei.yaml

docker:
	docker build -t flarenet:latest .

clean:
	rm -rf runs data build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
