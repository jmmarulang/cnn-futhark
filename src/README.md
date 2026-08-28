# Experimental validation

The `futhark/microgpt.fut` file contains the Futhark program. The
`grad_loss` function has been extracted from our Agda code. The rest
of the code is hand-written scaffolding and monomorphic variants of
rank-polymorphic operations. The `futhark/mainMGPT.py` contains Python
code for loading data and invoking the Python program.
Use `src/experiment-futhark.sh` to compile and run.


The `clang/microgpt.c` contains a c implementation for benchmarkig.
Use `src/experiment-futhark.sh` to compile and run.

