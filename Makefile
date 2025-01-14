.PHONY: devel
.PHONY: git-add
.PHONY: log
.PHONY: to-server

devel:
	python -m pip install -e .

init-sphinx:
	python -m pip install sphinx sphinx-rtd-theme
	mkdir docs
	sphinx-quickstart docs
	sphinx-build -M html docs/source/ docs/build/

html:
	cd docs; sphinx-apidoc -o ./source ../snapfish2;\
	make clean; make html; cd ..

git-add:
	git add -A snapfish2
	git add -A docs/source
	git add LICENSE
	git add .gitattributes
	git add Makefile
	git add README.md
	git add pyproject.toml

log:
	python 123ACElog/log.py

log110124: data/chiapet_mesc/chiapet_all_replicates_all_targets_intersection.csv
	python 123ACElog/log.py

to-server:
	scp -r figures \
	snapfish2 \
	Makefile \
	hongyuyu@longleaf.unc.edu:\
	/proj/yunligrp/users/hongyuyu/AxisWiseTest/.

	rsync -a --ignore-existing 123ACElog/011625 \
	hongyuyu@longleaf.unc.edu:\
	/proj/yunligrp/users/hongyuyu/AxisWiseTest/123ACElog/.

	rsync -a --ignore-existing data/* \
	hongyuyu@longleaf.unc.edu:\
	/proj/yunligrp/users/hongyuyu/AxisWiseTest/data/

# Will not overwrite existing files
from-server:
	rsync -a --ignore-existing hongyuyu@longleaf.unc.edu:\
	/proj/yunligrp/users/hongyuyu/AxisWiseTest/123ACElog/011625/* \
	123ACElog/011625

	rsync -a --ignore-existing hongyuyu@longleaf.unc.edu:\
	/proj/yunligrp/users/hongyuyu/AxisWiseTest/data/bonev_2017/FitHiC_filtered \
	data/bonev_2017/.

	scp hongyuyu@longleaf.unc.edu:/proj/yunligrp/users/hongyuyu/AxisWiseTest/Makefile .

# clean:
# 	rm -r 123ACElog/110124

data/chiapet_mesc/chiapet_all_replicates_all_targets_intersection.csv:
	python 123ACElog/utils.py