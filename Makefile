all: nutils numeric

nutils:
	git pull

numeric:
	git submodule init nutils/numeric
	git submodule update nutils/numeric
	rm -f nutils/numeric.pyc # for transition
	$(MAKE) -C nutils/numeric test_c

test:
	python tests/test.py

clean:
	$(MAKE) -C nutils/numeric clean
	rm -f nutils/*.pyc tests/*.pyc

.PHONY: all nutils numeric test clean

# vim:noexpandtab
