#!/bin/bash -eu
# ClusterFuzzLite build: compile the results-loader fuzz target.
# The target inserts the repo root into sys.path itself, so no package
# install step is needed; atheris is provided by the base builder image.
compile_python_fuzzer fuzz/fuzz_parse_results.py
