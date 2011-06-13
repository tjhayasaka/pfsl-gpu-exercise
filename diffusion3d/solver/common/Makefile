###
###
###

SOLVER_IMPL_NAME = $(shell basename $(shell pwd))
SOLVER_OPTIONS =

.PHONY: all
all:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) SOLVER_OPTIONS=$(SOLVER_OPTIONS) all

.PHONY: clean
clean:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) SOLVER_OPTIONS=$(SOLVER_OPTIONS) clean

.PHONY: realclean
realclean:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) SOLVER_OPTIONS=$(SOLVER_OPTIONS) realclean
