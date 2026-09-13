# APRIL source attribution

- Upstream: [APRIL-MedSeg](https://github.com/juntaoJianggavin/APRIL-MedSeg)
- License: [Apache-2.0](LICENSE)
- Upstream commit: unavailable
- Included source identity: [SOURCE_SHA256SUMS.txt](SOURCE_SHA256SUMS.txt)

This directory contains the model implementations and dependencies for the 17
APRIL-backed architectures in LUSBench. Upstream source headers and references
are preserved.

LUSBench supplies minimal package initializers and a lazy model builder so that
using one architecture does not import unrelated models. These modified adapter
files remain under Apache-2.0. The retained model class implementations are
unchanged; benchmark-specific runtime configuration is supplied by the training
and evaluation backends.
