###
###
###

SOLVER_IMPL_NAME = $(shell basename $(shell pwd))

.PHONY: all
all:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) all

.PHONY: clean
clean:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) clean

.PHONY: realclean
realclean:
	cd ../.. && $(MAKE) -f Makefile.impl SOLVER_IMPL_NAME=$(SOLVER_IMPL_NAME) realclean
