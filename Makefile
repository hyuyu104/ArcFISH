.PHONY: devel
.PHONY: git-add
.PHONY: log

devel:
	python -m pip install -e .

git-add:
	git add -A notebooks
	git add -A snapfish2
	git add LICENSE
	git add .gitattributes
	git add Makefile
	git add README.md
	git add pyproject.toml
	git add 123ACElog/log.py
	git add 123ACElog/utils.py

log: data/chiapet_mesc/chiapet_all_replicates_all_targets_intersection.csv
	python 123ACElog/log.py

clean:
	rm -r 123ACElog/110124

data/chiapet_mesc/chiapet_all_replicates_all_targets_intersection.csv:
	python 123ACElog/utils.py