# Third-party notices

The root PolyForm Noncommercial license applies to repository-authored code and
documentation except where a separate license is specified. It does not replace
the licenses of the third-party components below or restrict rights granted by
their licensors. Permission from LUSBench maintainers does not waive any
third-party requirements. Model weights and datasets require their own
applicable permissions.

## APRIL-MedSeg

- Upstream: https://github.com/juntaoJianggavin/APRIL-MedSeg
- License: Apache License 2.0
- Distribution: minimal source subset under `third_party/april_medseg`
- Source identity: file hashes in `third_party/april_medseg/SOURCE_SHA256SUMS.txt`; upstream commit unavailable

The subset contains the original implementations and transitive dependencies
needed by the 17 APRIL-backed LUSBench architectures. Package initializers and
`medseg/model_builder.py` were replaced with minimal, lazy LUSBench adapters to
avoid importing unrelated APRIL architectures. Those modified files state their
LUSBench purpose and remain under Apache-2.0.

Several APRIL model files identify additional research-code sources in their
headers. Those source comments have been preserved. Users should inspect the
upstream projects' terms before redistributing individual files outside this
benchmark bundle.

## Segment Anything Model 3 (SAM3)

- Upstream: https://github.com/facebookresearch/sam3
- Frozen source reference: `46957e47805eaa273f4aa7bbbd25a88bca9108ce`
- License: SAM License, copied to `third_party/sam3_patches/LICENSE`
- Distribution: patch logic and setup documentation only

Official SAM3 source and `sam3.pt` are not distributed. The LUSBench patch runs
Hungarian matching in FP32, validates finite matching costs, and clamps
zero-area IoU/GIoU denominators. Applying or distributing SAM3 materials is
subject to the SAM License.

## Other external implementations

MONAI, PyTorch, torchvision, timm, Transformers, Ultralytics, SAM2, MedSAM,
SAMUS, S2DENet, USFM, segmentation-models-pytorch, mamba-ssm, and their model
weights are external dependencies and are not redistributed here. Their own
licenses govern installation and use.
