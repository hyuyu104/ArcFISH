.PHONY: devel
.PHONY: git-add

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