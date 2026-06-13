from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence


def multiple_max_set_coverage(
    kf_seen_voxel_num: list,
    kf_voxel_indices: list,
    kf_unoptimized_voxels: Optional[torch.Tensor],
    kf_all_voxels: Optional[torch.Tensor],
    num_selections: int,
    num_voxels: int,
    device: str,
    padded_tensor: Optional[torch.Tensor] = None,
):
    """
    Overwrite all voxels contained in the keyframe multiple times

    Note that there is a related implementation outside of this function.
    In the insert_keyframe class method in mapping, The function is to add the new voxels contained
    in each newly added key frame to the set of unoptimized voxels.
    Cover all the voxels contained in the key frame multiple times
    Args:
        kf_seen_voxel_num (list): This is a list, each element is the number of the corresponding voxels contained in
                                  the keyframe in key_frames.
        kf_voxel_indices (list): indices of voxels contained in each keyframe.
        kf_unoptimized_voxels (tensor, N + 1): mask of all unoptimized voxels, N=max number of voxels.
        kf_all_voxels (tensor, N + 1): mask of all voxels to be optimized.
        num_selections (int): Number of keyframes to be selected.
        num_voxels (int): Number of total voxels in the octree.
        device (str): device to run the computation.
    Returns:
        selected_frame_indices (list): indices of selected keyframes.
        kf_unoptimized_voxels (tensor, N + 1): mask of unoptimized voxels after selection.
        kf_all_voxels (tensor, N + 1): mask of all voxels to be optimized.
    """

    cnt = 0
    selected_frame_indices = []

    # padded_tensor (B, M): each keyframe's voxel indices, right-padded with -1.
    # The -1 entries address the sink slot kf_unoptimized_voxels[-1] (kept False),
    # so they are no-ops in every gather/index_fill below — letting us use rows of
    # padded_tensor in place of re-transferring kf_voxel_indices[i] to the GPU. The
    # caller may pass a cached GPU tensor (it only changes when a keyframe is added)
    # to skip the per-call pad_sequence + H2D; result is identical either way.
    if padded_tensor is None:
        padded_tensor = pad_sequence(kf_voxel_indices, batch_first=True, padding_value=-1).long().to(device)
    if kf_unoptimized_voxels is None:
        kf_unoptimized_voxels = torch.zeros(num_voxels + 1, dtype=torch.bool).to(device)  # unoptimized voxels
        kf_all_voxels = torch.zeros(num_voxels + 1, dtype=torch.bool).to(device)  # All voxels to be optimized

        kf_seen_voxel_num = torch.tensor(kf_seen_voxel_num)  # (B), on CPU
        value, index = torch.max(kf_seen_voxel_num, dim=0)
        idx = index.item()
        selected_frame_indices.append(idx)

        kf_unoptimized_voxels.index_fill_(0, padded_tensor.view(-1), True)
        kf_unoptimized_voxels[-1] = False

        kf_unoptimized_voxels.index_fill_(0, padded_tensor[idx], False)

        cnt += 1

    kf_all_voxels.index_fill_(0, padded_tensor.view(-1), True)
    kf_all_voxels[-1] = False

    while cnt < num_selections:
        result_num = torch.sum(kf_unoptimized_voxels[padded_tensor].long(), dim=-1)  # (B)
        value, index = torch.max(result_num, dim=0)
        idx = index.item()
        selected_frame_indices.append(idx)

        voxel_indices = padded_tensor[idx]  # GPU row (incl. -1 padding -> sink)
        kf_unoptimized_voxels.index_fill_(0, voxel_indices, False)

        cnt += 1

        if not kf_unoptimized_voxels.any():  # If all are optimized

            # Unoptimized voxels = all voxels that need to be optimized - \
            # voxels seen by the latest selected key frame.
            kf_unoptimized_voxels[...] = kf_all_voxels
            kf_unoptimized_voxels.index_fill_(0, voxel_indices, False)

    return selected_frame_indices, kf_unoptimized_voxels, kf_all_voxels
