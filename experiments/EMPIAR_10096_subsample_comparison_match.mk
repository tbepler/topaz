#!/usr/bin/env make -f

SOURCES := $(shell find results/EMPIAR-10096_subsample_comparison/ -type f -name '*_extracted_test.txt')
TARGETS := $(SOURCES:_extracted_test.txt=_extracted_test_matched.txt)


all: $(TARGETS)

clean:
	rm -f $(TARGETS)

%_extracted_test_matched.txt: %_extracted_test.txt
	python scripts/match_extracted_particles.py -r7 --targets=data/EMPIAR-10096/picks_4a_test.txt -o $@ $^


.PHONY: all clean
