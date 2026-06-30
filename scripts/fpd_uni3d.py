#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MESH_SUFFIXES = {".obj", ".glb", ".gltf"}


def parse_args():
    p = argparse.ArgumentParser("Category-level Uni3D FPD for real vs generated meshes")

    p.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Folder of real .glb/.gltf/.obj meshes",
    )
    p.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Folder of generated .glb/.gltf/.obj meshes",
    )
    p.add_argument("--out-csv", type=Path, default=Path("uni3d_category_fpd.csv"))

    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to Uni3D model.pt checkpoint",
    )
    p.add_argument("--scale", default="base", choices=["tiny", "small", "base", "large", "giant"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")

    p.add_argument("--points-per-cloud", type=int, default=10000)
    p.add_argument("--mesh-clouds-per-file", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument(
        "--no-xyz-normalize",
        action="store_true",
        help=(
            "Disable the default per-cloud centering and unit-sphere scaling. "
            "Not recommended when comparing generators with different coordinates."
        ),
    )

    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional subset, e.g. dishwasher oven piano tabletop_clock",
    )

    return p.parse_args()


def category_from_path(path, root):
    # dishwasher.3dc200....glb -> dishwasher
    # tabletop_clock.fpModel....glb -> tabletop_clock
    if "." in path.stem:
        return path.stem.split(".", 1)[0]

    # Older SpaceControl output trees use generic names such as generated.glb:
    # <object-id>/<run-name>/generated.glb. Find the nearest ancestor that
    # carries the original dotted object ID.
    relative = path.relative_to(root)
    for parent_name in reversed(relative.parts[:-1]):
        if "." in parent_name:
            return parent_name.split(".", 1)[0]

    return path.stem


def collect_by_category(root):
    root = Path(root).expanduser().resolve()
    groups = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MESH_SUFFIXES:
            continue

        category = category_from_path(path, root)
        groups[category].append(path)

    return dict(groups)


def _parse_obj_vertex_index(token, num_vertices):
    index = int(token.split("/")[0])
    if index < 0:
        return num_vertices + index
    return index - 1


def _load_obj_triangles(path):
    import torch

    vertices = []
    triangles = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif parts[0] == "f" and len(parts) >= 4:
                face = [_parse_obj_vertex_index(tok, len(vertices)) for tok in parts[1:]]
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

    if not vertices:
        raise ValueError(f"{path} has no vertices")
    if not triangles:
        raise ValueError(f"{path} has no faces")

    return (
        torch.tensor(vertices, dtype=torch.float32),
        torch.tensor(triangles, dtype=torch.long),
    )


def _load_trimesh_triangles(path):
    import numpy as np
    import torch
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)

    meshes = []

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geometry_name)

            if geom is None or not isinstance(geom, trimesh.Trimesh):
                continue

            geom = geom.copy()
            geom.apply_transform(transform)
            meshes.append(geom)

    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)

    meshes = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]

    if not meshes:
        raise ValueError(f"{path} has no triangle geometry")

    vertices_parts = []
    faces_parts = []
    offset = 0

    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path} has non-triangular faces")

        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        offset += vertices.shape[0]

    vertices = torch.tensor(np.concatenate(vertices_parts, axis=0), dtype=torch.float32)
    triangles = torch.tensor(np.concatenate(faces_parts, axis=0), dtype=torch.long)

    return vertices, triangles


def load_mesh_triangles(path):
    suffix = Path(path).suffix.lower()

    if suffix == ".obj":
        return _load_obj_triangles(path)

    if suffix in {".glb", ".gltf"}:
        return _load_trimesh_triangles(path)

    raise ValueError(f"Unsupported mesh type: {suffix}")


def sample_mesh_surface(path, points_per_cloud, mesh_clouds):
    import torch

    vertices, triangles = load_mesh_triangles(path)
    tri_vertices = vertices[triangles]

    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]

    areas = torch.linalg.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1) * 0.5
    valid = areas > 0

    if not valid.any():
        raise ValueError(f"{path} only has zero-area triangles")

    tri_vertices = tri_vertices[valid]
    areas = areas[valid]
    probs = areas / areas.sum()

    clouds = []

    for _ in range(mesh_clouds):
        tri_idx = torch.multinomial(probs, points_per_cloud, replacement=True)
        chosen = tri_vertices[tri_idx]

        r1 = torch.rand(points_per_cloud, 1)
        r2 = torch.rand(points_per_cloud, 1)
        sqrt_r1 = torch.sqrt(r1)

        points = (
            (1.0 - sqrt_r1) * chosen[:, 0]
            + sqrt_r1 * (1.0 - r2) * chosen[:, 1]
            + sqrt_r1 * r2 * chosen[:, 2]
        )

        clouds.append(points)

    return torch.stack(clouds, dim=0)


def load_distribution(paths, points_per_cloud, mesh_clouds_per_file):
    import torch

    clouds = [
        sample_mesh_surface(path, points_per_cloud, mesh_clouds_per_file)
        for path in paths
    ]

    return torch.cat(clouds, dim=0)


def calculate_score(calculate_fpd_uni3d, generated, reference, args, torch):
    return calculate_fpd_uni3d(
        generated,
        pointclouds2=reference,
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

    args.real_dir = args.real_dir.expanduser().resolve()
    args.generated_dir = args.generated_dir.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.out_csv = args.out_csv.expanduser()

    for label, path in (
        ("real directory", args.real_dir),
        ("generated directory", args.generated_dir),
    ):
        if not path.is_dir():
            raise SystemExit(f"ERROR: {label} does not exist: {path}")

    if not args.ckpt.is_file():
        raise SystemExit(f"ERROR: checkpoint does not exist: {args.ckpt}")

    import torch
    from FPD import calculate_fpd_uni3d

    if args.seed is not None:
        torch.manual_seed(args.seed)

    real_groups = collect_by_category(args.real_dir)
    gen_groups = collect_by_category(args.generated_dir)

    categories = sorted(set(real_groups) | set(gen_groups))

    if args.categories is not None:
        categories = [c for c in categories if c in set(args.categories)]

    rows = []

    print(f"Found real categories: {sorted(real_groups)}")
    print(f"Found generated categories: {sorted(gen_groups)}")
    print(
        "XYZ normalization:",
        "disabled" if args.no_xyz_normalize else "centered unit sphere",
    )

    for category in categories:
        real_paths = real_groups.get(category, [])
        gen_paths = gen_groups.get(category, [])

        status = "ok"
        warning = ""
        score = ""

        if not real_paths or not gen_paths:
            status = "missing"
            warning = f"missing {'real' if not real_paths else 'generated'}"
            print(f"[{category}] SKIP: {warning}")

        else:
            try:
                print(f"[{category}] real={len(real_paths)} generated={len(gen_paths)}")

                reference = load_distribution(
                    real_paths,
                    args.points_per_cloud,
                    args.mesh_clouds_per_file,
                )

                generated = load_distribution(
                    gen_paths,
                    args.points_per_cloud,
                    args.mesh_clouds_per_file,
                )

                score = calculate_score(
                    calculate_fpd_uni3d,
                    generated,
                    reference,
                    args,
                    torch,
                )

                print(f"[{category}] Uni3D FPD = {score}")

            except Exception as exc:
                status = "error"
                warning = repr(exc)
                print(f"[{category}] ERROR: {warning}")

        rows.append(
            {
                "category": category,
                "fpd": score,
                "status": status,
                "warning": warning,
                "num_real": len(real_paths),
                "num_generated": len(gen_paths),
                "real_files": ";".join(str(p) for p in real_paths),
                "generated_files": ";".join(str(p) for p in gen_paths),
            }
        )

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "fpd",
                "status",
                "warning",
                "num_real",
                "num_generated",
                "real_files",
                "generated_files",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
