# SAM3 source patch

- Upstream: [Segment Anything Model 3](https://github.com/facebookresearch/sam3)
- Required commit: `46957e47805eaa273f4aa7bbbd25a88bca9108ce`
- License: [SAM License](LICENSE)
- Targets: `sam3/model/box_ops.py` and `sam3/train/matcher.py`

Only patch logic is included in this directory. Obtain the official source
separately, then run from the LUSBench repository root:

```bash
python third_party/sam3_patches/apply.py --repo-root /path/to/sam3
```

The command checks the Git revision, applies guarded replacements and keeps
local backups. It can be rerun on the same checkout. Matching costs use FP32,
nonfinite inputs are rejected, and IoU/GIoU denominators handle zero-area boxes.
Use and redistribution of upstream SAM3 materials remain subject to the SAM License.
