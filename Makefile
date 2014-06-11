default: nutils numeric

all: nutils numeric docs

nutils:
	git pull

numeric:
	git submodule init nutils/numeric
	git submodule update nutils/numeric
	rm -f nutils/numeric.pyc # for transition
	$(MAKE) -C nutils/numeric test_c

docs:
	$(MAKE) -C docs html

test:
	python tests/test.py

clean:
	$(MAKE) -C nutils/numeric clean
	rm -f nutils/*.pyc tests/*.pyc

.PHONY: default all nutils numeric test clean
