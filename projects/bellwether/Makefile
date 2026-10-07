PY := .venv/bin/python

.PHONY: init-db test run-watcher setup-external-pinger lint local-prod

init-db:
	$(PY) scripts/init_db.py

test:
	.venv/bin/pytest -q

run-watcher:
	$(PY) scripts/run_watcher.py $(CMD)

dashboard:
	.venv/bin/streamlit run dashboard.py

export-web:
	$(PY) scripts/export_web.py

setup-external-pinger:
	$(PY) scripts/setup_external_pinger.py

web:
	npm --prefix bellwether-web run dev

lint:
	@echo "lint not configured yet"

local-prod:
	$(PY) scripts/local_daemon.py
