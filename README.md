# LUS Consolidation Segmentation Benchmark

Training and evaluation code for pulmonary-consolidation segmentation in lung
ultrasound images. The benchmark includes 36 models on 512 x 512 images and
224 x 224 patches, covering CNN, Transformer, Mamba, RWKV, foundation-model,
YOLO and other architectures.

## Installation

```bash
git clone https://github.com/THA-NJU/LUS-Consolidation-Benchmark.git
cd LUS-Consolidation-Benchmark
```

Create the environment for your chosen model using the
[environment guide](docs/ENVIRONMENTS.md). Model families use separate Python
environments. In a Python 3.10+ environment, install the repository package and
fetch the external sources:

```bash
python -m pip install -e .
python tools/setup_sources.py
```

The package installs shared utilities; model frameworks require the additional
dependencies listed in the environment guide. APRIL model implementations are
included under `third_party/april_medseg`. External source revisions are specified
in [configs/external_sources.json](configs/external_sources.json).

## Dataset

Images are extracted from patient videos. The 224 track comprises sliding crops
of the 512 images and is repartitioned by patient after cropping. Training,
validation and test patients are disjoint within each track; corresponding
partitions differ across tracks.

| Track | Training images | Validation images | Test images | Total |
|---:|---:|---:|---:|---:|
| 512 | 16,131 | 2,111 | 1,539 | 19,781 |
| 224 | 26,017 | 2,796 | 3,557 | 32,370 |

Each track contains 93 training, 12 validation and 11 test patients. Two
physicians annotate independently. If their Dice exceeds 0.93, the union of
their masks is used; otherwise a third physician supplies the reference mask.

Arrange each track as follows, with matching image and mask stems:

```text
Size_512/                 # or Size_224/
  train/images/
  train/masks/
  val/images/
  val/masks/
  test/images/
  test/masks/
```

Use binary masks and specify the foreground labels required by the selected
evaluator. The examples below accept foreground values 1 and 255. Supply the
dataset root explicitly; a folder named `Size_224_filtered` is also accepted
and does not require an additional filtering step. Dataset access will be
provided after paper acceptance.

## Training

List model identifiers and available training entry points:

```bash
python tools/train.py --list
```

Activate the model's environment before training. For example:

```bash
python tools/train.py \
  --model pvtb2_emcad --size 512 \
  --data-root /path/to/Size_512 \
  --output-dir ./outputs/training
```

Pass model-specific options after `--`. For SegFormer, supply an initialization
model directory as needed:

```bash
python tools/train.py \
  --model segformer_b2 --size 224 \
  --data-root /path/to/Size_224 \
  --output-dir ./outputs/training -- \
  --base-model /path/to/segformer-b2
```

The common training configuration uses at most 600 epochs, 10 warmup epochs,
validation every epoch, patience of 15 epochs without Dice improvement, and
a default learning rate of `1e-4`. Model selection uses the highest validation
per-image mean Dice. See [configs/benchmark_protocol.json](configs/benchmark_protocol.json)
for the shared settings and [SAM3 instructions](experiments/foundation/sam3/README.md)
for SAM3 source and checkpoint requirements.

The ResNet34 transfer models and USFM at 512 have evaluation entry points only.
USFM at 224 uses transfer training. The SAM3 training entry point uses interactive
prompts; checkpoint formats must match the selected training or evaluation backend.

## Evaluation

Use the [evaluation guide](evaluation/README.md) for full-split evaluation and
five-image inference. The latter supports all 36 models on either track and
saves predicted masks for the same five patients within a track.

For full-test SegFormer evaluation:

```bash
python tools/evaluate.py --model segformer_b2 --size 512 -- \
  --checkpoint /path/to/best.pt --size 512 \
  --base-model ./configs/segformer-b2 --local-files-only \
  --data-root /path/to/Size_512 \
  --output-dir ./outputs/evaluation/segformer_512 \
  --splits test --labels 1 255
```

Metrics include Dice, IoU, precision, recall, HD, HD95 and connected-component
differences. Binary probability outputs use a fixed threshold of 0.5. For a
nonempty reference and an empty prediction, HD and HD95 equal the track side
length (512 or 224 pixels) and contribute to the per-image mean and population
standard deviation. Other distances are not clipped. Empty-reference cases are
excluded from foreground segmentation aggregates.

## Weights

Weights are distributed separately from this code repository. Place checkpoints
under `Pth/<family>/<224,512>/<model>/`, preserving the model folder names and
checkpoint filenames. The evaluation guide describes how to supply this root.

## License

Except for separately licensed third-party material, repository-authored code
and documentation are provided under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

The license permits noncommercial use, modification and redistribution. It also
expressly permits use by the organizations listed in its Noncommercial
Organizations section, including educational institutions and public research
organizations, regardless of their funding source or funding obligations.
Redistributions must include the license text or its URL and the `Required Notice:`
lines in [NOTICE](NOTICE). The full license text governs these permissions.

Commercial uses outside the license's permitted purposes require separate prior
written permission from the relevant rights holders. Contact the repository
maintainers through GitHub to discuss a licensing request.

Third-party code retains its own licenses; this license does not replace or
restrict rights granted by those licensors. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and datasets are
licensed separately under their accompanying terms. The code license does not
automatically grant rights to those assets.
