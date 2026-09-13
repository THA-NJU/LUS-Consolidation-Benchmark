# Manual setup of model-family environments

Use separate environments for the model families you need. Existing working
environments can be used directly. There is no combined installation for all
models. The table lists the Python and PyTorch versions used for each model group.
Install the dependencies required by the selected model; the table is not a
complete dependency lock file.

| Key | Models | Python | PyTorch / CUDA wheel |
|---|---|---|---|
| `base` | CNN/general Transformer/SegFormer/YOLO/S2DENet/other | 3.10 | 2.5.1 / cu124 |
| `ssm` | Mamba and RWKV | 3.11 | 2.5.1 / cu121 |
| `usfm` | USFM | 3.9 | 2.4.1 / cu118 |
| `sam` | SAM2 and MedSAM | 3.10 | 2.5.1 / cu124 |
| `samus` | SAMUS | 3.10 | 2.5.1 / cu124 |
| `sam3` | SAM3 | 3.12 | 2.10.0 / cu128 |

## Create only the environments you need

With Conda installed, run the applicable lines:

```bash
conda create -n lus-base python=3.10 pip
conda create -n lus-ssm python=3.11 pip
conda create -n lus-usfm python=3.9 pip
conda create -n lus-sam python=3.10 pip
conda create -n lus-samus python=3.10 pip
conda create -n lus-sam3 python=3.12 pip
```

Activate the chosen environment, for example:

```bash
conda activate lus-base
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch/torchvision for the group's Python, CUDA build and GPU driver,
using the official [PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
Mamba/RWKV extensions additionally require a compatible local CUDA compiler;
the wheel's CUDA runtime alone does not provide that compiler. Keep the group's
PyTorch version fixed while resolving additional dependencies.

## Install dependencies for your models

From the repository root, the following is a starting point for general
CNN/Transformer adapters after installing PyTorch, not a complete all-model setup:

```bash
python -m pip install -e .
python -m pip install timm einops monai segmentation-models-pytorch scikit-image scikit-learn
```

Add dependencies for the selected models below. Preview a large resolution with
`python -m pip install --dry-run ...`. Install a single OpenCV distribution
supplying `cv2`; the [OpenCV installation guide](https://pypi.org/project/opencv-python/4.11.0.86/)
explains why multiple variants should not share an environment.

| Model/group | Additional requirements |
|---|---|
| Shared native adapters | NumPy, Pillow, SciPy, tqdm, OpenCV; pandas/Matplotlib for native report paths that use them |
| SegFormer | `transformers`, `safetensors`; local `configs/segformer-b2` supplies the architecture; supply the trained checkpoint explicitly |
| YOLO | Compatible Ultralytics including the semantic task used by this benchmark; version 8.4.92. Use a separate YOLO environment if its dependencies conflict with other base models |
| Mamba | Bundled APRIL sources, `timm`, `einops`, `mamba-ssm`; extension version 2.2.4. Install/build only after matching PyTorch, CUDA and compiler |
| RWKV | Bundled APRIL sources, `timm`, `einops`, Ninja and CUDA compiler; the runtime builds WKV |
| USFM | Pinned `.external/usfm` and custom `.external/usfm_mmseg`; matching MMCV/MMEngine and their runtime dependencies. MMCV version 2.2.0. Do not substitute an unrelated MMSegmentation fork |
| SAM2 | Pinned `.external/sam2`, its `pyproject.toml`/`setup.py` dependencies and compatible CUDA build |
| MedSAM | Pinned `.external/medsam`, its `setup.py` dependencies and MONAI when required by the native trainer |
| SAMUS | Pinned `.external/samus` and the packages required by its models/trainer. Use dependencies compatible with the PyTorch 2.5.1 stack shown above |
| SAM3 | Pinned `.external/sam3` and its `pyproject.toml` dependencies in Python 3.12; the native interactive trainer requires standard `sam3.pt` |
| S2DENet | Pinned `.external/s2denet`, its model imports and shared evaluation dependencies |

Fetch recorded source commits once using a Python 3.10+ controller:

```bash
python tools/setup_sources.py
```

Source revisions are listed in `configs/external_sources.json`. Runtime dispatch
adds selected source paths to child processes; package dependencies must still be
installed. Use the specified source revisions for model compatibility.

USFM's Python 3.9 backend runs directly from the checkout. Do not install the
repository's Python 3.10+ metadata package into that backend. Use a Python 3.10+
controller and the `usfm` executable mapping for the child process.

### NumPy and OpenCV

Choose mutually compatible NumPy and OpenCV versions. Models requiring NumPy 1.x
need an OpenCV version that supports NumPy 1.x. Keep a separate environment when
another model requires NumPy 2. Do not force incompatible dependencies with
`--no-deps`.

## Map environments to models

In each active environment, get its executable:

```bash
python -c "import sys; print(sys.executable)"
```

Copy the template and replace the Python paths with your own:

```bash
cp configs/runtime.example.json configs/runtime.local.json
```

Paths may be absolute or relative to the JSON file's directory. A model-specific
entry overrides the family default, so additional environments are supported:

```json
{
  "python": {
    "base": "/path/to/lus-base/bin/python",
    "ssm": "/path/to/lus-ssm/bin/python",
    "usfm": "/path/to/lus-usfm/bin/python",
    "sam": "/path/to/lus-sam/bin/python",
    "samus": "/path/to/lus-samus/bin/python",
    "sam3": "/path/to/lus-sam3/bin/python",
    "segformer_b2": "/path/to/lus-segformer/bin/python"
  }
}
```

Keep this configuration local. Five-image evaluation with
`tools/evaluate.py --sample` selects these interpreters automatically. Ordinary
training and full-split evaluation use the current Python interpreter; activate
the model environment or invoke its Python executable explicitly.

## Check the selected environment

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Continue with the [evaluation guide](../evaluation/README.md) to prepare examples
and save predictions, or the [training instructions](../README.md#training).
