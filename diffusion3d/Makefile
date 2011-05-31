###
###
###

### Do not write any implementation specific stuff in this file!!!

###
###
###

SHELL = /bin/sh

top_srcdir = .
srcdir = $(top_srcdir)

# FLOATS = float double
FLOATS = float
SOLVER_IMPL_NAMES = $(shell for f in $(top_srcdir)/solver/*@*@*; do basename "$$f"; done)

SUBMAKE_VARS =

.PHONY: all
all:
	for float in $(FLOATS); do \
	  for impl in $(SOLVER_IMPL_NAMES); do $(MAKE) -f Makefile.impl $(SUBMAKE_VARS) FLOAT=$$float SOLVER_IMPL_NAME=$$impl FLOAT=$$float all; done; \
	done

.PHONY: realclean
realclean:
	-rm -f bin/*@*@*@*
	cd solver && for d in *@*@*; do ( cd $$d && $(MAKE) -f Makefile.impl $(SUBMAKE_VARS) realclean); done

.PHONY: clean
clean:
	for float in $(FLOATS); do \
	  for impl in $(SOLVER_IMPL_NAMES); do $(MAKE) -f Makefile.impl $(SUBMAKE_VARS) SOLVER_IMPL_NAME=$$impl FLOAT=$$float clean; done; \
	done
