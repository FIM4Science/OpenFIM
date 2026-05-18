VENV = ../.venv
PYTHON = $(VENV)/bin/python
JB = $(VENV)/bin/jupyter-book

.PHONY: local_install local_start local clean

local_install:
	uv pip install -r requirements.txt --python $(PYTHON)

local_start:
	$(JB) start _build/html

local: clean local_install
	$(JB) init
	$(JB) build . --watch & $(JB) start _build/html

clean:
	rm -rf _build
	rm -f myst.yml
	rm -f ._config.yml.bak
	rm -f ._toc.yml.bak
	rm -f .gitignore
	git checkout _config.yml
	git checkout _toc.yml
