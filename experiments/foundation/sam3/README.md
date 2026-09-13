# SAM3

Use the SAM3 environment described in the
[environment guide](../../../docs/ENVIRONMENTS.md). Obtain the official SAM3
source at commit `46957e47805eaa273f4aa7bbbd25a88bca9108ce` and the standard
`sam3.pt` initialization checkpoint under the upstream SAM License.

From the repository root, apply the source patch:

```bash
python third_party/sam3_patches/apply.py --repo-root /path/to/sam3
```

The patch uses FP32 Hungarian matching costs, checks finite matcher inputs and
clamps zero-area IoU/GIoU denominators. It checks the source revision and can be
rerun on the same checkout.

## Training

```bash
python tools/train.py --model sam3 --size 224 \
  --data-root /path/to/Size_224 --output-dir ./outputs/training -- \
  --sam3-source-dir /path/to/sam3 --checkpoint /path/to/sam3.pt
```

The interactive trainer uses a fixed full-image box, input size 1008 and a local
checkpoint. Its default trainable components are the interactive prompt encoder
and mask decoder. Training runs for at most 600 epochs, with 10 warmup epochs
and early-stopping patience of 15. Use `--size 512` with the corresponding
dataset root for the 512 track.

## Checkpoint evaluation

The interactive trainer accepts `--eval-only --resume` for its own delta
checkpoint format. Official-trainer checkpoints use
`evaluation/foundation/evaluate_sam3_official.py`. These checkpoint formats are
distinct; select the evaluator matching the checkpoint producer.
For five-image inference, follow the
[evaluation guide](../../../evaluation/README.md) with `--model sam3` and set
the `sam3` interpreter and source path in the runtime configuration.
