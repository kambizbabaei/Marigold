# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape
    logger.info(f"Original shape: {ori_shape}")

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Log input statistics
    logger.info(f"GT stats - min: {np.min(gt):.4f}, max: {np.max(gt):.4f}, mean: {np.mean(gt):.4f}")
    logger.info(f"Pred stats - min: {np.min(pred):.4f}, max: {np.max(pred):.4f}, mean: {np.mean(pred):.4f}")
    logger.info(f"Valid mask - True count: {np.sum(valid_mask)}, Total: {valid_mask.size}")

    # Check for NaN and Inf values
    if np.isnan(gt).any():
        logger.error("NaN values found in GT!")
    if np.isnan(pred).any():
        logger.error("NaN values found in prediction!")
    if np.isinf(gt).any():
        logger.error("Inf values found in GT!")
    if np.isinf(pred).any():
        logger.error("Inf values found in prediction!")

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        logger.info(f"Downsampling with scale factor: {scale_factor:.4f}")
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )
            logger.info(f"Downsampled shape: {gt.shape}")

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # Log masked data statistics
    logger.info(f"Masked GT stats - min: {np.min(gt_masked):.4f}, max: {np.max(gt_masked):.4f}, mean: {np.mean(gt_masked):.4f}")
    logger.info(f"Masked Pred stats - min: {np.min(pred_masked):.4f}, max: {np.max(pred_masked):.4f}, mean: {np.mean(pred_masked):.4f}")
    logger.info(f"Number of valid points: {len(gt_masked)}")

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    
    # Log matrix condition
    try:
        condition_number = np.linalg.cond(A)
        logger.info(f"Matrix condition number: {condition_number:.4e}")
    except:
        logger.error("Failed to compute condition number")

    try:
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X
        logger.info(f"Computed scale: {scale:.4f}, shift: {shift:.4f}")
    except np.linalg.LinAlgError as e:
        logger.error(f"Least squares failed: {str(e)}")
        # Try with a more stable rcond
        try:
            X = np.linalg.lstsq(A, gt_masked, rcond=1e-4)[0]
            scale, shift = X
            logger.info(f"Computed scale (with rcond=1e-4): {scale:.4f}, shift: {shift:.4f}")
        except np.linalg.LinAlgError as e:
            logger.error(f"Least squares failed even with rcond=1e-4: {str(e)}")
            # Fallback to simple scaling
            scale = np.mean(gt_masked) / np.mean(pred_masked)
            shift = 0
            logger.info(f"Using fallback scale: {scale:.4f}, shift: {shift:.4f}")

    aligned_pred = pred_arr * scale + shift

    # Log aligned prediction statistics
    logger.info(f"Aligned pred stats - min: {np.min(aligned_pred):.4f}, max: {np.max(aligned_pred):.4f}, mean: {np.mean(aligned_pred):.4f}")

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
