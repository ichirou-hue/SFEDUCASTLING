install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	cd ../../../.. && python -m pip install -r requirements.txt

install-test: ## [Local development] Install test requirements
	cd ../../../.. && python -m pip install -r requirements.txt

test: ## [Local development] Run unit tests
	cd ../../../.. && python -m pytest -x -s -v tests/ml/integration/chessclip --run-ml-integration
