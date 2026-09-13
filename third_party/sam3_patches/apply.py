#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import subprocess
from pathlib import Path


EXPECTED_SAM3_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"[already patched] {label}")
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence for {label}, found {count}. "
            "Inspect the source before patching."
        )
    print(f"[patch] {label}")
    return text.replace(old, new, 1)


def patch_box_ops(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.before_lus_stability")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        '''    union = area1[..., None] + area2[..., None, :] - inter

    iou = inter / union
    return iou, union
''',
        '''    union = area1[..., None] + area2[..., None, :] - inter

    # Numerical guard for degenerate / zero-area boxes. Hungarian matching is
    # non-differentiable, and a 0/0 here otherwise becomes NaN and crashes
    # scipy.optimize.linear_sum_assignment.
    eps = torch.finfo(union.dtype).eps
    iou = inter / union.clamp_min(eps)
    return iou, union
''',
        "box_iou safe denominator",
    )

    text = replace_once(
        text,
        '''    area = wh[..., 0] * wh[..., 1]  # (..., N, M)

    return iou - (area - union) / area
''',
        '''    area = wh[..., 0] * wh[..., 1]  # (..., N, M)

    eps = torch.finfo(area.dtype).eps
    return iou - (area - union) / area.clamp_min(eps)
''',
        "generalized_box_iou safe denominator",
    )

    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)


def patch_matcher(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.before_lus_stability")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        '''        _, num_queries = outputs["pred_logits"].shape[:2]

        out_score = outputs["pred_logits"].squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))
''',
        '''        _, num_queries = outputs["pred_logits"].shape[:2]
        # Matching is @torch.no_grad(); use FP32 even when the model trains
        # under BF16 autocast to avoid non-finite cdist / GIoU costs.
        out_score = outputs["pred_logits"].squeeze(-1).float()  # (B, Q)
        out_bbox = outputs["pred_boxes"].float()  # (B, Q, 4)
''',
        "BinaryHungarianMatcherV2 FP32 predictions",
    )

    text = replace_once(
        text,
        '''        tgt_bbox = batched_targets["boxes_padded"]
        if self.remove_samples_with_0_gt:
''',
        '''        tgt_bbox = batched_targets["boxes_padded"].float()
        if self.remove_samples_with_0_gt:
''',
        "BinaryHungarianMatcherV2 FP32 targets",
    )

    text = replace_once(
        text,
        '''        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]

        # Compute the L1 cost between boxes
''',
        '''        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]

        for tensor_name, tensor in (
            ("pred_logits", out_score),
            ("pred_boxes", out_bbox),
            ("target_boxes", tgt_bbox),
        ):
            if not torch.isfinite(tensor).all():
                bad = int((~torch.isfinite(tensor)).sum().item())
                raise FloatingPointError(
                    f"Non-finite {tensor_name} before Hungarian matching: "
                    f"count={bad}, shape={tuple(tensor.shape)}"
                )

        # Compute the L1 cost between boxes
''',
        "BinaryHungarianMatcherV2 finite input checks",
    )

    text = replace_once(
        text,
        '''        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
''',
        '''        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        if not torch.isfinite(C).all():
            bad = int((~torch.isfinite(C)).sum().item())
            raise FloatingPointError(
                "Non-finite Hungarian cost after FP32 stabilization: "
                f"count={bad}, "
                f"bbox_finite={bool(torch.isfinite(cost_bbox).all())}, "
                f"class_finite={bool(torch.isfinite(cost_class).all())}, "
                f"giou_finite={bool(torch.isfinite(cost_giou).all())}"
            )
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
''',
        "BinaryHungarianMatcherV2 finite cost checks",
    )

    text = replace_once(
        text,
        '''        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = self.norm(outputs["pred_logits"]).squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))
''',
        '''        bs, num_queries = outputs["pred_logits"].shape[:2]
        logits_fp32 = outputs["pred_logits"].float()
        out_prob = self.norm(logits_fp32).squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"].float()  # (B, Q, 4)
''',
        "BinaryOneToManyMatcher FP32 predictions",
    )

    text = replace_once(
        text,
        '''        tgt_bbox = batched_targets["boxes_padded"]
        assert len(tgt_bbox) == bs
''',
        '''        tgt_bbox = batched_targets["boxes_padded"].float()
        assert len(tgt_bbox) == bs
''',
        "BinaryOneToManyMatcher FP32 targets",
    )

    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Official SAM3 Git checkout at the frozen reference commit.",
    )
    parser.add_argument("--allow-source-mismatch", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot determine SAM3 source revision below {repo}") from exc
    if commit != EXPECTED_SAM3_COMMIT and not args.allow_source_mismatch:
        raise RuntimeError(
            f"SAM3 commit mismatch: found {commit}, expected {EXPECTED_SAM3_COMMIT}. "
            "Checkout the frozen commit or explicitly declare a non-reference patch run."
        )
    matcher = repo / "sam3/train/matcher.py"
    box_ops = repo / "sam3/model/box_ops.py"

    if not matcher.is_file():
        raise FileNotFoundError(matcher)
    if not box_ops.is_file():
        raise FileNotFoundError(box_ops)

    patch_box_ops(box_ops)
    patch_matcher(matcher)

    print("\nSAM3 matcher stability patch completed.")
    print(f"matcher: {matcher}")
    print(f"box ops: {box_ops}")
    print("Backups use suffix: .py.before_lus_stability")


if __name__ == "__main__":
    main()
