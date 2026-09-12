check:
	python -m py_compile scripts/*.py
	python -m json.tool config/config.example.json > /dev/null

list-results:
	find results/reference -maxdepth 1 -type f -print
