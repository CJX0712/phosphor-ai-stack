PY ?= python
SRC := src

.PHONY: install install-ai install-dev guard test e2e eval verify serve lint clean openapi

install:
	$(PY) -m pip install -r requirements.txt

install-ai:
	$(PY) -m pip install -r requirements-ai.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

guard:
	$(PY) scripts/scan_guard.py

test:
	$(PY) -m pytest -q tests

e2e:
	$(PY) scripts/e2e_http.py

eval:
	$(PY) -m phosphor.cli eval

lint:
	$(PY) -m ruff check $(SRC) tests scripts || true

openapi:
	$(PY) -c "import sys,yaml;sys.path.insert(0,'src');from phosphor.api.app import create_app;open('docs/openapi.yaml','w',encoding='utf-8').write(yaml.safe_dump(create_app().openapi(),allow_unicode=True,sort_keys=False))"

verify: guard test e2e eval

serve:
	$(PY) -m phosphor.cli serve --port 8080

clean:
	$(PY) -c "import shutil;shutil.rmtree('.artifacts',ignore_errors=True);shutil.rmtree('.cache',ignore_errors=True)"
