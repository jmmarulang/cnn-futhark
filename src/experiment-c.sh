#!/bin/sh
#
# Run the various experiments.
#
# Currently has some stuff specific to the cluster the script was written for.

gcc -O3 -ffast-math -march=native -mtune=native -o clang/microgpt clang/microgpt.c -lm
clang/microgpt