#!/bin/sh
#
# Run the various experiments.
#
# Currently has some stuff specific to the cluster the script was written for.

export CC=gcc
# export CFLAGS="-O3 -std=c99 -ffast-math -march=native -mtune=native"
export CFLAGS="-O3 -ffast-math -march=native -mtune=native"

futhark c --server futhark/microgpt.fut

python futhark/mainMGPT.py
