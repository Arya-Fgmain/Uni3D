#!/usr/bin/env python3
"""Compare perturbed_pcs against perturbed_pcs_comp using Uni3D FPD.

This is a self-comparison / sanity-check script for layouts like:

  extracted_parts/<category>/perturbed_pcs/*.ply
  extracted_parts/<category>/perturbed_pcs_comp/*.ply

Example:
  python scripts/fpd_uni3d_self_compare_ply.py \
      --extracted-parts-dir ../TRELLIS.2/extracted_parts \
      --ckpt models/model.pt \
      --scale base \
      --out-csv uni3d_fpd_self_compare.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

POINTCLOUD_SUFFIXES = {".ply", ".pt", ".pth"}


def parse_args():
    parser = argparse.ArgumentParser("Compare perturbed_pcs vs perturbed_pcs_comp with Uni3D FPD")
    parser.add_argument("--extracted-parts-dir", required=True, help="Root folder containing perturbed_pcs folders.")
    parser.add_argument("--pc-dir-name", default="perturbed_pcs", help="Original perturbed PC folder name.")
    parser.add_argument("--comp-dir-name", default="perturbed_pcs_comp", help="Comparison perturbed PC folder name.")
    parser.add_argument("--out-csv", default="uni3d_fpd_self_compare.csv")

    parser.add_argument("--ckpt", required=True, help="Path to Uni3D model.pt checkpoint.")
    parser.add_argument("--scale", default="base", choices=["tiny", "small", "base", "large", "giant"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--no-xyz-normalize", action="store_true")
    parser.add_argument("--require-count", type=int, default=5, help="Expected files per distribution; 0 disables warnings.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def default_key(stem):
    """Turn 'basket_x_perturbed_00' or 'basket_x_perturbed_comp_00' into 'basket_x'."""
    patterns = [
        r"^(.+?)_perturbed_comp[_-]?\d+$",
        r"^(.+?)_perturbed(?:_pc)?[_-]?\d+$",
        r"^(.+?)_comp[_-]?\d+$",
        r"^(.+?)_pc[_-]?\d+$",
        r"^(.+?)_sample[_-]?\d+$",
        r"^(.+?)[_-]\d+$",
    ]
    for pattern in patterns:
        m = re.match(pattern, stem)
        if m:
            return m.group(1)
    return stem


def collect_groups(root, target_dir_name):
    groups = defaultdict(list)
    categories = {}

    root = Path(root)
    for pc_dir in sorted(root.rglob(target_dir_name)):
        if not pc_dir.is_dir() or pc_dir.name != target_dir_name:
            continue

        category = pc_dir.parent.name
        for path in sorted(pc_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in POINTCLOUD_SUFFIXES:
                continue
            key = default_key(path.stem)
            groups[key].append(path)
            categories[key] = category

    return dict(groups), categories


def load_pointcloud(path):
    import numpy as np
    import torch

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        pc = torch.load(path, map_location="cpu")
        if isinstance(pc, dict):
            for key in ("pointclouds", "points", "pc", "xyz"):
                if key in pc:
                    pc = pc[key]
                    break
            else:
                raise ValueError(f"{path} is a dict but has none of pointclouds/points/pc/xyz.")
        pc = torch.as_tensor(pc).float()
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        if pc.dim() != 3 or pc.shape[-1] not in (3, 6):
            raise ValueError(f"{path} expected [N,3], [N,6], [B,N,3], or [B,N,6], got {tuple(pc.shape)}")
        return pc

    if suffix == ".ply":
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("Reading .ply requires trimesh: pip install trimesh") from exc

        obj = trimesh.load(path, process=False)

        # Point-cloud PLYs usually load as trimesh.points.PointCloud.
        if hasattr(obj, "vertices"):
            points = np.asarray(obj.vertices, dtype=np.float32)
        elif hasattr(obj, "geometry"):
            parts = []
            for geom in obj.geometry.values():
                if hasattr(geom, "vertices"):
                    parts.append(np.asarray(geom.vertices, dtype=np.float32))
            if not parts:
                raise ValueError(f"{path} has no vertices.")
            points = np.concatenate(parts, axis=0)
        else:
            raise ValueError(f"{path} has no vertices.")

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"{path} expected vertices shaped [N,3+], got {points.shape}")

        points = points[:, :3]
        return torch.tensor(points, dtype=torch.float32).unsqueeze(0)

    raise ValueError(f"Unsupported point-cloud suffix: {suffix}")


def load_distribution(paths):
    import torch

    clouds = [load_pointcloud(p) for p in paths]
    return torch.cat(clouds, dim=0)


def calculate_score(calculate_fpd_uni3d, original, comp, args, torch):
    return calculate_fpd_uni3d(
        original,
        pointclouds2=comp,
        ckpt_path=args.ckpt,
        scale=args.scale,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None,
        statistic_save_path=None,
        normalize=not args.no_xyz_normalize,
        l2_normalize=not args.no_l2_normalize,
    )


def main():
    args = parse_args()

    import torch
    from FPD import calculate_fpd_uni3d

    orig_groups, orig_categories = collect_groups(args.extracted_parts_dir, args.pc_dir_name)
    comp_groups, comp_categories = collect_groups(args.extracted_parts_dir, args.comp_dir_name)

    all_keys = sorted(set(orig_groups) | set(comp_groups))
    rows = []

    print(f"Found {len(orig_groups)} {args.pc_dir_name} groups and {len(comp_groups)} {args.comp_dir_name} groups.")

    for key in all_keys:
        orig_paths = orig_groups.get(key, [])
        comp_paths = comp_groups.get(key, [])
        category = orig_categories.get(key, comp_categories.get(key, ""))

        status = "ok"
        warning = ""
        score = ""

        if not orig_paths or not comp_paths:
            status = "missing"
            warning = f"missing {'perturbed_pcs' if not orig_paths else 'perturbed_pcs_comp'}"
        elif args.require_count and (len(orig_paths) != args.require_count or len(comp_paths) != args.require_count):
            status = "count_warning"
            warning = f"expected {args.require_count}/{args.require_count}, got perturbed_pcs={len(orig_paths)}, perturbed_pcs_comp={len(comp_paths)}"

        if status == "missing" and args.strict:
            raise ValueError(f"{key}: {warning}")

        if orig_paths and comp_paths:
            try:
                print(f"[{key}] perturbed_pcs={len(orig_paths)} perturbed_pcs_comp={len(comp_paths)}")
                original = load_distribution(orig_paths)
                comp = load_distribution(comp_paths)
                score = calculate_score(calculate_fpd_uni3d, original, comp, args, torch)
                print(f"[{key}] self Uni3D FPD = {score}")
            except Exception as exc:
                if args.strict:
                    raise
                status = "error"
                warning = repr(exc)
                print(f"[{key}] ERROR: {warning}")

        rows.append({
            "category": category,
            "key": key,
            "self_fpd": score,
            "status": status,
            "warning": warning,
            "num_perturbed_pcs": len(orig_paths),
            "num_perturbed_pcs_comp": len(comp_paths),
            "perturbed_pc_files": ";".join(str(p) for p in orig_paths),
            "perturbed_pc_comp_files": ";".join(str(p) for p in comp_paths),
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "key",
                "self_fpd",
                "status",
                "warning",
                "num_perturbed_pcs",
                "num_perturbed_pcs_comp",
                "perturbed_pc_files",
                "perturbed_pc_comp_files",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["self_fpd"] != "")
    print(f"Saved {len(rows)} rows to {out_csv} ({ok} computed scores).")


if __name__ == "__main__":
    main()
