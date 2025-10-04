.PHONY: run install clean runner svd_runner build_svd svd nmf_builder nmf_runner nmf
.DEFAULT_GOAL=runner
run: install
	poetry run python src/runner_prediction.py

build: install
	poetry run python src/runner_builder.py
install:
	poetry install --no-root

clean: pyproject.toml
	poetry run python clean.py

runner: run clean

svd_runner: install
	poetry run python src/runner_prediction_svd.py

build_svd: install
	poetry run python src/runner_builder_svd.py

svd: svd_runner clean

nmf_runner: install
	poetry run python src/runner_prediction_nmf.py

nmf_builder: install
	poetry run python src/runner_builder_nmf.py

nmf: nmf_runner clean