import os
import pathlib
from types import SimpleNamespace

import numpy as np
import torch

try:
    from pointnet import PointNetCls
except ImportError:
    PointNetCls = None

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

"""Calculate Frechet Pointcloud Distance referened by Frechet Inception Distance."
    [ref] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    github code  : (https://github.com/bioinf-jku/TTUR)
    paper        : (https://arxiv.org/abs/1706.08500)

"""


# Uni3D checkpoint configs from scripts/inference.sh. The model scale must match
# the checkpoint you downloaded, otherwise checkpoint loading will fail.
UNI3D_SCALE_CONFIG = {
    "tiny": {"pc_model": "eva02_tiny_patch14_224", "pc_feat_dim": 192},
    "small": {"pc_model": "eva02_small_patch14_224", "pc_feat_dim": 384},
    "base": {"pc_model": "eva02_base_patch14_448", "pc_feat_dim": 768},
    "large": {"pc_model": "eva02_large_patch14_448", "pc_feat_dim": 1024},
    "giant": {"pc_model": "eva_giant_patch14_560", "pc_feat_dim": 1408},
}


def get_activations(pointclouds, model, batch_size=100, dims=1808,
                    device=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    n_batches = pointclouds.size(0) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    pointclouds = pointclouds.transpose(1,2)
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        pointcloud_batch = pointclouds[start:end]

        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        _, _, actv = model(pointcloud_batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = actv.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def normalize_pointcloud_xyz(pointclouds, eps=1e-6):
    """Center and scale XYZ to the unit sphere, matching Uni3D datasets."""
    xyz = pointclouds[..., :3]
    centroid = xyz.mean(dim=1, keepdim=True)
    xyz = xyz - centroid
    scale = torch.sqrt((xyz ** 2).sum(dim=-1)).amax(dim=1, keepdim=True)
    xyz = xyz / scale.clamp_min(eps).unsqueeze(-1)

    # Preserve RGB if it is already present; only XYZ is normalized.
    if pointclouds.shape[-1] > 3:
        return torch.cat([xyz, pointclouds[..., 3:]], dim=-1)
    return xyz


def prepare_uni3d_pointclouds(pointclouds, normalize=True, rgb_fill=0.4):
    """Prepare point clouds for Uni3D's encode_pc interface.

    PointNet FPD uses [B, N, 3] and transposes to [B, 3, N].
    Uni3D uses [B, N, 6], where the channels are XYZ followed by RGB.
    If your generated samples do not have color, use the neutral 0.4 RGB
    value used by the repository datasets.
    """
    if isinstance(pointclouds, np.ndarray):
        pointclouds = torch.from_numpy(pointclouds)

    pointclouds = pointclouds.float()

    if pointclouds.dim() != 3:
        raise ValueError("Uni3D FPD expects a tensor shaped [B, N, 3] or [B, N, 6].")

    if pointclouds.shape[-1] not in (3, 6):
        raise ValueError("Last dimension must be 3 for XYZ or 6 for XYZRGB.")

    if normalize:
        pointclouds = normalize_pointcloud_xyz(pointclouds)

    if pointclouds.shape[-1] == 3:
        rgb = torch.ones_like(pointclouds) * rgb_fill
        pointclouds = torch.cat([pointclouds, rgb], dim=-1)

    return pointclouds


def build_uni3d_model(ckpt_path, scale="base", device=None, strict=True):
    """Build Uni3D and load model-zoo weights for feature extraction.

    This intentionally does not build the CLIP text/image model. FPD only needs
    the point-cloud encoder exposed by model.encode_pc().
    """
    import models.uni3d as uni3d_models

    if scale not in UNI3D_SCALE_CONFIG:
        raise ValueError("Unknown Uni3D scale '{}'. Choose one of {}.".format(
            scale, sorted(UNI3D_SCALE_CONFIG.keys())
        ))

    cfg = UNI3D_SCALE_CONFIG[scale]
    args = SimpleNamespace(
        pc_model=cfg["pc_model"],
        pc_feat_dim=cfg["pc_feat_dim"],
        pretrained_pc="",
        drop_path_rate=0.0,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        embed_dim=1024,
        patch_dropout=0.0,
    )

    model = uni3d_models.create_uni3d(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Model-zoo checkpoints use "module"; bare state dicts also work here.
    state_dict = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_uni3d_activations(pointclouds, model, batch_size=32, dims=1024,
                          device=None, verbose=False, normalize=True,
                          rgb_fill=0.4, l2_normalize=True):
    """Calculate Uni3D point-cloud embeddings for FPD statistics.

    Uni3D embeddings are CLIP-like. L2 normalization matches the zero-shot
    evaluation path in main.py and usually makes this metric less sensitive to
    feature magnitude drift.
    """
    model.eval()
    pointclouds = prepare_uni3d_pointclouds(
        pointclouds,
        normalize=normalize,
        rgb_fill=rgb_fill,
    )

    n_batches = int(np.ceil(pointclouds.size(0) / float(batch_size)))
    pred_arr = []

    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating Uni3D batch %d/%d" % (i + 1, n_batches),
                  end="", flush=True)

        start = i * batch_size
        end = min(start + batch_size, pointclouds.size(0))
        pointcloud_batch = pointclouds[start:end]

        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        actv = model.encode_pc(pointcloud_batch)
        if l2_normalize:
            actv = actv / actv.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        actv = actv.float().cpu().numpy()
        if actv.shape[-1] != dims:
            raise ValueError("Expected Uni3D feature dim {}, got {}.".format(dims, actv.shape[-1]))
        pred_arr.append(actv)

    if verbose:
        print(" done")

    return np.concatenate(pred_arr, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Covariances are rank deficient when a category has fewer samples than
    # embedding dimensions.  sqrtm(sigma1 @ sigma2) is then numerically
    # unstable because the product is not symmetric and may acquire enormous
    # imaginary components.  Use the equivalent PSD expression:
    # Tr(sqrt(sqrt(sigma1) @ sigma2 @ sqrt(sigma1))).
    sigma1 = (sigma1 + sigma1.T) * 0.5
    sigma2 = (sigma2 + sigma2.T) * 0.5
    eigenvalues1, eigenvectors1 = np.linalg.eigh(sigma1)
    eigenvalues1 = np.clip(eigenvalues1, 0.0, None)
    sqrt_sigma1 = (
        eigenvectors1 * np.sqrt(eigenvalues1)
    ).dot(eigenvectors1.T)
    covariance_product = sqrt_sigma1.dot(sigma2).dot(sqrt_sigma1)
    covariance_product = (covariance_product + covariance_product.T) * 0.5
    product_eigenvalues = np.linalg.eigvalsh(covariance_product)
    tr_covmean = np.sqrt(np.clip(product_eigenvalues, 0.0, None)).sum()

    distance = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * tr_covmean
    )
    return float(max(distance, 0.0))


def calculate_activation_statistics(pointclouds, model, batch_size=100,
                                    dims=1808, device=None, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(pointclouds, model, batch_size, dims, device, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_uni3d_activation_statistics(pointclouds, model, batch_size=32,
                                          dims=1024, device=None,
                                          verbose=False, normalize=True,
                                          rgb_fill=0.4, l2_normalize=True):
    """Calculate Frechet statistics from Uni3D embeddings."""
    act = get_uni3d_activations(
        pointclouds,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        verbose=verbose,
        normalize=normalize,
        rgb_fill=rgb_fill,
        l2_normalize=l2_normalize,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['m'][:], f['s'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s

def save_statistics(real_pointclouds, path, model, batch_size, dims, cuda):
    m, s = calculate_activation_statistics(real_pointclouds, model, batch_size,
                                         dims, cuda)
    np.savez(path, m = m, s = s)
    print('save done !!!')


def save_uni3d_statistics(real_pointclouds, path, model, batch_size=32,
                          dims=1024, device=None, normalize=True,
                          rgb_fill=0.4, l2_normalize=True):
    """Save real-set Uni3D statistics for reuse.

    Do not reuse the old PointNet pre_statistics.npz with Uni3D FPD; the
    covariance and mean must come from the same embedding model.
    """
    m, s = calculate_uni3d_activation_statistics(
        real_pointclouds,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        normalize=normalize,
        rgb_fill=rgb_fill,
        l2_normalize=l2_normalize,
    )
    np.savez(path, m=m, s=s)
    print("save Uni3D statistics done")

def calculate_fpd(pointclouds1, pointclouds2=None, batch_size=100, dims=1808, device=None):
    """Calculates the FPD of two pointclouds"""

    PointNet_path = './evaluation/cls_model_39.pth'
    statistic_save_path = './evaluation/pre_statistics.npz'
    model = PointNetCls(k=16)
    map_location = device if device is not None else torch.device('cpu')
    model.load_state_dict(torch.load(PointNet_path, map_location=map_location))

    if device is not None:
        model.to(device)

    m1, s1 = calculate_activation_statistics(pointclouds1, model, batch_size, dims, device)
    if pointclouds2 is not None:
        m2, s2 = calculate_activation_statistics(pointclouds2, model, batch_size, dims, device)
    else: # Load saved statistics of real pointclouds.
        f = np.load(statistic_save_path)
        m2, s2 = f['m'][:], f['s'][:]
        f.close()

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fpd_uni3d(pointclouds1, pointclouds2=None, ckpt_path=None,
                        scale="base", batch_size=32, dims=1024,
                        device=None, statistic_save_path=None,
                        normalize=True, rgb_fill=0.4, l2_normalize=True):
    """Calculate FPD with Uni3D instead of PointNet.

    Args:
        pointclouds1: Generated/evaluated point clouds shaped [B, N, 3] or [B, N, 6].
        pointclouds2: Reference point clouds shaped [B, N, 3] or [B, N, 6].
        ckpt_path: Path to the downloaded Uni3D model.pt checkpoint.
        scale: One of tiny/small/base/large/giant, matching the checkpoint.
        statistic_save_path: Optional .npz containing Uni3D real-set statistics.

    Either pass pointclouds2 directly or pass statistic_save_path containing
    statistics produced by save_uni3d_statistics().
    """
    if ckpt_path is None:
        raise ValueError("ckpt_path is required for Uni3D FPD.")

    if pointclouds2 is None and statistic_save_path is None:
        raise ValueError("Provide pointclouds2 or statistic_save_path.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_uni3d_model(ckpt_path, scale=scale, device=device)

    m1, s1 = calculate_uni3d_activation_statistics(
        pointclouds1,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        normalize=normalize,
        rgb_fill=rgb_fill,
        l2_normalize=l2_normalize,
    )

    if pointclouds2 is not None:
        m2, s2 = calculate_uni3d_activation_statistics(
            pointclouds2,
            model,
            batch_size=batch_size,
            dims=dims,
            device=device,
            normalize=normalize,
            rgb_fill=rgb_fill,
            l2_normalize=l2_normalize,
        )
    else:
        f = np.load(statistic_save_path)
        m2, s2 = f["m"][:], f["s"][:]
        f.close()

    return calculate_frechet_distance(m1, s1, m2, s2)
