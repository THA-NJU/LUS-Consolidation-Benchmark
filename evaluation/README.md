# Evaluation

Run commands from the repository root after following the
[environment guide](../docs/ENVIRONMENTS.md). Supply the dataset and weights
as local paths. Each track uses its own test partition.

## Five-image inference

Prepare five test images from five distinct patients for each track:

```bash
python tools/prepare_examples.py --data-root /path/to/Size_512 \
  --size 512 --output ./outputs/examples/512
python tools/prepare_examples.py --data-root /path/to/Size_224 \
  --size 224 --output ./outputs/examples/224
```

The output directory must be new. Patient IDs are read from the `pNNN_` filename
prefix. Selection uses the track size as its random seed and is independent
between tracks. Each folder contains copied test images, reference masks and a
`case_manifest.json` with content hashes. Reuse the same folder for every model
within that track. Source data are not modified.

Configure the model interpreters in `configs/runtime.local.json`, then run:

```bash
python tools/evaluate.py --model segformer_b2 --size 512 --sample \
  --pth /path/to/Pth --subset ./outputs/examples/512 \
  --out ./outputs/predictions --runtime-config configs/runtime.local.json

python tools/evaluate.py --model segformer_b2 --size 224 --sample \
  --pth /path/to/Pth --subset ./outputs/examples/224 \
  --out ./outputs/predictions --runtime-config configs/runtime.local.json
```

Replace `segformer_b2` with a model identifier from `python tools/train.py --list`.
All 36 models support this mode. Keep the distributed weight directory layout
under `Pth/<family>/<size>/<model>/`; some model directories contain nested
checkpoint folders. Inference writes binary PNG masks under
`<out>/<size>/<model>/masks/`, using 0 for background and 255 for foreground.
Use a new output root when changing model weights or input images.

Reference masks are used for metric calculation, not as prediction inputs.
Five-image metrics describe only the selected examples; use the full test split
for benchmark results. This mode does not measure inference latency.

## Full-split evaluation

Activate the model's environment. List standalone evaluators with:

```bash
python tools/evaluate.py --list
```

Options after `--` are forwarded to the selected backend. A SegFormer example:

```bash
python tools/evaluate.py --model segformer_b2 --size 512 -- \
  --checkpoint /path/to/best.pt --size 512 \
  --base-model ./configs/segformer-b2 --local-files-only \
  --data-root /path/to/Size_512 \
  --output-dir ./outputs/evaluation/segformer_512 \
  --splits test --labels 1 255
```

For the native YOLO26s semantic evaluator:

```bash
python evaluation/native/evaluate_yolo26s_sem.py \
  --checkpoint /path/to/best.pt --size 224 \
  --data-root /path/to/Size_224 \
  --output-dir ./outputs/evaluation/yolo26s_224 \
  --splits test --labels 1 255
```

Backend options differ. Request their help through the selector, for example
`python tools/evaluate.py --model segformer_b2 --size 512 -- --help`.
An entry displayed as `-` has no standalone full-split evaluator: use the
evaluation integrated in its training backend, or the five-image mode above
for checkpoint inference. The latter does not replace full-split evaluation.

SAM3 official-trainer checkpoints and interactive-trainer checkpoints have
different formats. See the [SAM3 guide](../experiments/foundation/sam3/README.md).

## Metrics

HD and HD95 are measured in pixels at the native track resolution. For a
nonempty reference and an empty prediction, both distances equal 512 on the
512 track or 224 on the 224 track. These values are included in the mean and
population standard deviation; ordinary nonempty-mask distances are not clipped.
Empty-reference cases are excluded from foreground segmentation aggregates.
Connected-component delta is reference count minus prediction count.
The shared definitions are in
[configs/benchmark_protocol.json](../configs/benchmark_protocol.json).
