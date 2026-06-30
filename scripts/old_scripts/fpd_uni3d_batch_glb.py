"""CLI for calculating Uni3D FPD for either one pair or many matched distributions.

Single-pair example:
  python scripts/fpd_uni3d_batch_glb.py \
      --generated bat.glb \
      --reference m2.glb \
      --ckpt checkpoints/uni3d-b/model.pt \
      --scale base

Batch example for your folder layout:
  python scripts/fpd_uni3d_batch_glb.py \
      --batch \
      --extracted-parts-dir extracted_parts \
      --generations-dir generations \
      --ckpt checkpoints/uni3d-b/model.pt \
      --scale base \
      --out-csv uni3d_fpd_results.csv

Expected batch layout:
  extracted_parts/<category>/perturbed_pcs/*.ply, *.pt, or *.pth
  generations/**/*.glb, *.gltf, or *.obj

Default matching behavior:
  - generation key: filename stem before '_sample_'
      drum.3dc200-bf_01f_sample_0_1234.glb -> drum.3dc200-bf_01f
  - perturbed PC key: filename stem with common replicate suffixes removed
      drum.3dc200-bf_01f_0.pt              -> drum.3dc200-bf_01f
      drum.3dc200-bf_01f_perturbed_0.pt    -> drum.3dc200-bf_01f
      drum.3dc200-bf_01f_sample_0.pt       -> drum.3dc200-bf_01f

If your names differ, use --gen-key-regex and/or --pc-key-regex. The first
capturing group is used as the key.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
import copy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MESH_SUFFIXES = {".obj", ".glb", ".gltf"}
TENSOR_SUFFIXES = {".pt", ".pth"}
POINTCLOUD_SUFFIXES = TENSOR_SUFFIXES | {".ply"}
INPUT_SUFFIXES = MESH_SUFFIXES | POINTCLOUD_SUFFIXES


def parse_args():
    parser = argparse.ArgumentParser("Calculate FPD with Uni3D features")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--batch", action="store_true", help="Evaluate all matched generated/GT distributions.")
    mode.add_argument("--generated", help="Single-mode path to .obj, .glb, .gltf, .ply, .pt, or .pth point-cloud data.")
    mode.add_argument("--reference", help="Single-mode path to .obj, .glb, .gltf, .ply, .pt, or .pth point-cloud data.")
    mode.add_argument("--stats", help="Single-mode optional Uni3D .npz stats made with save_uni3d_statistics().")

    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--extracted-parts-dir", help="Root folder containing */perturbed_pcs/ directories.")
    batch.add_argument("--generations-dir", help="Root folder containing generated mesh files.")
    batch.add_argument("--out-csv", default="uni3d_fpd_results.csv", help="Where to save batch results.")
    batch.add_argument(
        "--gen-key-regex",
        default=r"^(.+?)_sample_",
        help="Regex used on generation filename stems. First capture group becomes the object key.",
    )
    batch.add_argument(
        "--pc-key-regex",
        default=None,
        help="Optional regex used on perturbed PC filename stems. First capture group becomes the object key.",
    )
    batch.add_argument(
        "--batch-mesh-clouds",
        type=int,
        default=1,
        help="Point clouds sampled per generated mesh in batch mode. Use 1 for a true 5-vs-5 setup.",
    )
    batch.add_argument(
        "--require-count",
        type=int,
        default=5,
        help="Expected number of generated files and perturbed PC files per object. Set 0 to disable warnings.",
    )
    batch.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error instead of skipping keys with missing/invalid files.",
    )

    uni3d = parser.add_argument_group("Uni3D / FPD")
    uni3d.add_argument("--ckpt", required=True, help="Path to downloaded Uni3D model.pt checkpoint.")
    uni3d.add_argument(
        "--scale",
        default="base",
        choices=["tiny", "small", "base", "large", "giant"],
        help="Must match the downloaded checkpoint family.",
    )
    uni3d.add_argument("--batch-size", type=int, default=32)
    uni3d.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu.")
    uni3d.add_argument(
        "--points-per-cloud",
        type=int,
        default=10000,
        help="Number of surface points to sample per cloud when an input is a mesh.",
    )
    uni3d.add_argument(
        "--mesh-clouds",
        type=int,
        default=64,
        help="Number of point clouds to sample from each mesh in single mode.",
    )
    uni3d.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable CLIP-style feature normalization before Frechet statistics.",
    )
    uni3d.add_argument(
        "--no-xyz-normalize",
        action="store_true",
        help="Disable unit-sphere XYZ normalization if your tensors are already preprocessed.",
    )
    uni3d.add_argument("--seed", type=int, default=None, help="Optional seed for mesh surface sampling.")
    uni3d.add_argument(
        "--icp-threshold",
        type=float,
        default=0.05,
        help="ICP max correspondence distance after pre-normalization.",
    )

    return parser.parse_args()


def _parse_obj_vertex_index(token, num_vertices):
    index = int(token.split("/")[0])
    if index < 0:
        return num_vertices + index
    return index - 1


def _load_obj_triangles(path):
    import torch

    vertices = []
    triangles = []

    with open(path, "r") as obj_file:
        for line in obj_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face = [_parse_obj_vertex_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

    if not vertices:
        raise ValueError(f"{path} does not contain OBJ vertices.")
    if not triangles:
        raise ValueError(f"{path} does not contain OBJ faces to sample.")

    vertices = torch.tensor(vertices, dtype=torch.float32)
    triangles = torch.tensor(triangles, dtype=torch.long)
    return vertices, triangles


def _load_trimesh_triangles(path):
    import numpy as np
    import torch

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("Loading .glb/.gltf requires trimesh. Install it with: pip install trimesh") from exc

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
        raise ValueError(f"{path} does not contain any triangle mesh geometry.")

    vertices_parts = []
    faces_parts = []
    vertex_offset = 0

    for mesh in meshes:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path} contains non-triangular faces after loading.")
        vertices_parts.append(vertices)
        faces_parts.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]

    vertices = torch.tensor(np.concatenate(vertices_parts, axis=0), dtype=torch.float32)
    triangles = torch.tensor(np.concatenate(faces_parts, axis=0), dtype=torch.long)
    return vertices, triangles


def _load_mesh_triangles(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return _load_obj_triangles(path)
    if suffix in {".glb", ".gltf"}:
        return _load_trimesh_triangles(path)
    raise ValueError(f"Unsupported mesh input type '{suffix}'.")


def _sample_mesh_surface(path, points_per_cloud, mesh_clouds):
    import torch

    vertices, triangles = _load_mesh_triangles(path)
    tri_vertices = vertices[triangles]

    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]
    areas = torch.linalg.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1) * 0.5

    valid = areas > 0
    if not valid.any():
        raise ValueError(f"{path} only contains zero-area triangles.")

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


def _load_ply_pointcloud(path):
    """Load XYZ or XYZRGB points from a PLY point cloud.

    Assumes the PLY represents a point cloud through its vertex array. If RGB is
    present, returns [1, N, 6]; otherwise returns [1, N, 3].
    """
    import numpy as np
    import torch

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("Loading .ply requires trimesh. Install it with: pip install trimesh") from exc

    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geometry_name)
            if geom is None or not hasattr(geom, "vertices"):
                continue
            geom = geom.copy()
            if hasattr(geom, "apply_transform"):
                geom.apply_transform(transform)
            verts = np.asarray(geom.vertices, dtype=np.float32)
            if verts.size:
                parts.append(verts)
        if not parts:
            raise ValueError(f"{path} does not contain PLY vertices.")
        points = np.concatenate(parts, axis=0)
        return torch.tensor(points, dtype=torch.float32).unsqueeze(0)

    if not hasattr(loaded, "vertices"):
        raise ValueError(f"{path} does not contain PLY vertices.")

    points = np.asarray(loaded.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} has invalid PLY vertex shape {points.shape}.")
    points = points[:, :3]

    # Preserve RGB if trimesh exposes vertex colors. Uni3D FPD can also fill
    # neutral RGB later if only XYZ is given, so missing color is fine.
    colors = None
    visual = getattr(loaded, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vertex_colors is not None:
        colors = np.asarray(vertex_colors)
        if colors.ndim == 2 and colors.shape[0] == points.shape[0] and colors.shape[1] >= 3:
            colors = colors[:, :3].astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            points = np.concatenate([points, colors], axis=1)

    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)


def load_pointcloud_input(path, points_per_cloud, mesh_clouds):
    import torch

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TENSOR_SUFFIXES:
        pointclouds = torch.load(path, map_location="cpu")
        if isinstance(pointclouds, dict):
            for key in ("pointclouds", "points", "pc", "xyz"):
                if key in pointclouds:
                    pointclouds = pointclouds[key]
                    break
            else:
                raise ValueError(f"{path} is a dict, but none of pointclouds/points/pc/xyz were found.")

        if pointclouds.dim() == 2:
            pointclouds = pointclouds.unsqueeze(0)
        if pointclouds.dim() != 3 or pointclouds.shape[-1] not in (3, 6):
            raise ValueError(f"{path} should have shape [B,N,3], [B,N,6], [N,3], or [N,6], got {tuple(pointclouds.shape)}.")
        return pointclouds.float()

    if suffix == ".ply":
        return _load_ply_pointcloud(path)

    if suffix in MESH_SUFFIXES:
        return _sample_mesh_surface(path, points_per_cloud, mesh_clouds)

    raise ValueError(f"Unsupported input type '{suffix}'. Use .obj, .glb, .gltf, .ply, .pt, or .pth.")


def load_distribution(paths, points_per_cloud, mesh_clouds):
    """Load many files and concatenate along the distribution/B dimension."""
    import torch

    clouds = [load_pointcloud_input(path, points_per_cloud, mesh_clouds) for path in paths]
    return torch.cat(clouds, dim=0)



def normalize_xyz_for_icp(points):
    """Center and scale one [N,3] tensor by its bbox diagonal for ICP only."""
    import torch

    center = points.mean(dim=0, keepdim=True)
    points = points - center

    bbox_diag = torch.linalg.norm(
        points.max(dim=0).values - points.min(dim=0).values
    )

    return points / (bbox_diag + 1e-8)


def icp_align_distribution(generated, reference, threshold=0.05):
    """
    ICP-align each generated cloud to the first reference cloud.

    Inputs:
        generated: [B,N,3] or [B,N,6]
        reference: [B,N,3] or [B,N,6]

    Output:
        aligned generated tensor, same shape as generated.

    Notes:
        - ICP is performed on XYZ only.
        - RGB channels, if present, are preserved from generated.
        - Each generated cloud and the first reference cloud are normalized
          before ICP, so the returned generated XYZ is in the normalized
          target/reference coordinate frame.
    """
    import torch
    import numpy as np

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("ICP alignment requires open3d. Install it with: pip install open3d") from exc

    ref_xyz_t = reference[0, :, :3].detach().cpu().float()
    ref_xyz_norm = normalize_xyz_for_icp(ref_xyz_t).numpy()

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(ref_xyz_norm)

    aligned_clouds = []

    for i in range(generated.shape[0]):
        gen_xyz_t = generated[i, :, :3].detach().cpu().float()
        gen_xyz_norm = normalize_xyz_for_icp(gen_xyz_t).numpy()

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(gen_xyz_norm)

        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        src_aligned = o3d.geometry.PointCloud(src)
        src_aligned.transform(result.transformation)

        aligned_xyz = np.asarray(src_aligned.points, dtype=np.float32)

        if generated.shape[-1] == 6:
            rgb = generated[i, :, 3:].detach().cpu().numpy()
            aligned = np.concatenate([aligned_xyz, rgb], axis=1)
        else:
            aligned = aligned_xyz

        aligned_clouds.append(torch.tensor(aligned, dtype=generated.dtype))

    return torch.stack(aligned_clouds, dim=0)


def key_from_regex(stem, regex):
    match = re.search(regex, stem)
    if not match:
        return None
    return match.group(1)


def default_pc_key(stem):
    """Strip common replicate suffixes from a perturbed point-cloud filename stem."""
    patterns = [
        r"^(.+?)_perturbed(?:_pc)?[_-]?\d+$",
        r"^(.+?)_pc[_-]?\d+$",
        r"^(.+?)_sample[_-]?\d+$",
        r"^(.+?)_copy[_-]?\d+$",
        r"^(.+?)[_-]\d+$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            return match.group(1)
    return stem


def collect_generated_groups(generations_dir, gen_key_regex):
    groups = defaultdict(list)
    for path in sorted(Path(generations_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MESH_SUFFIXES:
            continue
        key = key_from_regex(path.stem, gen_key_regex)
        if key is None:
            key = path.stem
        groups[key].append(path)
    return dict(groups)


def collect_perturbed_pc_groups(extracted_parts_dir, pc_key_regex=None):
    groups = defaultdict(list)
    categories = {}

    for pc_dir in sorted(Path(extracted_parts_dir).rglob("perturbed_pcs")):
        if not pc_dir.is_dir():
            continue
        category = pc_dir.parent.name
        for path in sorted(pc_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in POINTCLOUD_SUFFIXES:
                continue
            key = key_from_regex(path.stem, pc_key_regex) if pc_key_regex else default_pc_key(path.stem)
            if key is None:
                key = default_pc_key(path.stem)
            groups[key].append(path)
            categories[key] = category

    return dict(groups), categories


def calculate_score(calculate_fpd_uni3d, generated, reference, args, torch):
    return calculate_fpd_uni3d(
        generated,
        pointclouds2=reference,
        ckpt_path=args.ckpt,
        scale=args.scale,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None,
        statistic_save_path=args.stats if not args.batch else None,
        normalize=not args.no_xyz_normalize,
        l2_normalize=not args.no_l2_normalize,
    )


def run_single(args):
    if not args.generated:
        raise ValueError("Single mode requires --generated.")
    if not args.reference and not args.stats:
        raise ValueError("Single mode requires either --reference or --stats.")

    import torch
    from FPD import calculate_fpd_uni3d

    if args.seed is not None:
        torch.manual_seed(args.seed)

    generated = load_pointcloud_input(args.generated, args.points_per_cloud, args.mesh_clouds)
    reference = (
        load_pointcloud_input(args.reference, args.points_per_cloud, args.mesh_clouds)
        if args.reference
        else None
    )

    generated = icp_align_distribution(generated, reference, threshold=args.icp_threshold)

    score = calculate_score(calculate_fpd_uni3d, generated, reference, args, torch)
    print(f"Uni3D FPD: {score}")


def run_batch(args):
    if not args.extracted_parts_dir or not args.generations_dir:
        raise ValueError("Batch mode requires --extracted-parts-dir and --generations-dir.")

    import torch
    from FPD import calculate_fpd_uni3d

    if args.seed is not None:
        torch.manual_seed(args.seed)

    gen_groups = collect_generated_groups(args.generations_dir, args.gen_key_regex)
    pc_groups, categories = collect_perturbed_pc_groups(args.extracted_parts_dir, args.pc_key_regex)

    all_keys = sorted(set(gen_groups) | set(pc_groups))
    rows = []

    print(f"Found {len(gen_groups)} generated groups and {len(pc_groups)} perturbed-PC groups.")

    for key in all_keys:
        gen_paths = gen_groups.get(key, [])
        pc_paths = pc_groups.get(key, [])
        category = categories.get(key, "")

        status = "ok"
        warning = ""
        score = ""

        if not gen_paths or not pc_paths:
            status = "missing"
            warning = f"missing {'generated' if not gen_paths else 'perturbed_pcs'}"
        elif args.require_count and (len(gen_paths) != args.require_count or len(pc_paths) != args.require_count):
            status = "count_warning"
            warning = f"expected {args.require_count}/{args.require_count}, got generated={len(gen_paths)}, perturbed_pcs={len(pc_paths)}"

        if status == "missing" and args.strict:
            raise ValueError(f"{key}: {warning}")

        if gen_paths and pc_paths:
            try:
                print(f"[{key}] generated={len(gen_paths)} perturbed_pcs={len(pc_paths)}")
                generated = load_distribution(gen_paths, args.points_per_cloud, args.batch_mesh_clouds)
                reference = load_distribution(pc_paths, args.points_per_cloud, 1)
                generated = icp_align_distribution(generated, reference, threshold=args.icp_threshold)

                score = calculate_score(calculate_fpd_uni3d, generated, reference, args, torch)
                print(f"[{key}] Uni3D FPD = {score}")
            except Exception as exc:
                if args.strict:
                    raise
                status = "error"
                warning = repr(exc)
                print(f"[{key}] ERROR: {warning}")

        rows.append(
            {
                "category": category,
                "key": key,
                "fpd": score,
                "status": status,
                "warning": warning,
                "num_generated": len(gen_paths),
                "num_perturbed_pcs": len(pc_paths),
                "generated_files": ";".join(str(p) for p in gen_paths),
                "perturbed_pc_files": ";".join(str(p) for p in pc_paths),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "key",
                "fpd",
                "status",
                "warning",
                "num_generated",
                "num_perturbed_pcs",
                "generated_files",
                "perturbed_pc_files",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] in {"ok", "count_warning"} and row["fpd"] != "")
    print(f"Saved {len(rows)} rows to {out_csv} ({ok} computed scores).")


def main():
    args = parse_args()
    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
