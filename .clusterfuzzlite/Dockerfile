# ClusterFuzzLite builder image for model-protocol-lab (cycle 217).
# Standard base-builder-python image, pinned by digest (Scorecard
# Pinned-Dependencies requires hash-pinned container images); the
# project is copied in by the ClusterFuzzLite runner, which then
# executes build.sh from $SRC. No pip upgrade step: the base image's
# pip is sufficient for compile_python_fuzzer, and an unpinned
# upgrade would itself trip the same check.
FROM gcr.io/oss-fuzz-base/base-builder-python:v1@sha256:21f6fab17ace4dc76e1c56de91ee95d3d03da021df94b41056ba036f6c6f55e9
COPY . $SRC/model-protocol-lab
WORKDIR $SRC/model-protocol-lab
COPY .clusterfuzzlite/build.sh $SRC/build.sh
