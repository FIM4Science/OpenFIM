.PHONY: local_install local_start local build clean serve

local_install:
	uv lock --upgrade --project ..
	uv sync --project .. --group tutorial

local_start:
	uv run --project .. --group tutorial jupyter-book start

build: local_install
	uv run --project .. --group tutorial jupyter-book build --check-links --html --strict

check_notebooks: clean build # requires HF_TOKEN
	uv run --group tutorial pytest --nbmake --nbmake-timeout=600 .

clean:
	rm -rf _build
	rm -rf .myst
	rm -rf .pytest_cache

local: clean local_install local_start

serve: build
	@echo "Serving the tutorial locally at http://localhost:8000"
	uv run python -m http.server --directory _build/html 8000
