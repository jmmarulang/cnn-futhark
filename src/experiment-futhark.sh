#!/bin/sh
#
# Run the various experiments.
#
# Currently has some stuff specific to the cluster the script was written for.

export CC=gcc
export CFLAGS="-O3 -ffast-math -march=native -mtune=native"

# export CC=./test-script

futhark c --server futhark/microgpt.fut

# futhark c --server futhark/test.fut

# python futhark/mainMGPT.py
