import numpy as np
from ..mag1c_sas_base import compute_base_mag1c_SAS
from .utils import tile_image, stitch_tiles, compute_valid_mask, preprocess_image, compute_sampling_indices, postprocess_result


def mag1c_sas(image, methane_spectrum, sample_ratio=0.01, tiling=False, tile_size=512):
    """
    Pipeline for creating Mag1c-SAS product from a CHW hyperspectral image.
    
    Args:
        image: Hyperspectral image of shape (C, H, W).
        methane_spectrum: Methane absorption spectrum template.
        sample_ratio: Ratio of pixels to sample for SAS (default 0.01 = 1%).
        tiling: Whether to process the image in tiles (default False).
        tile_size: Size of tiles when tiling is enabled (default 512).
    
    Returns:
        Mag1c-SAS result reshaped back to (H, W).
    """
    if tiling:
        # Tile the image
        tiles, tiling_info = tile_image(image, tile_size)
        
        # Process each tile
        result_tiles = []
        for tile in tiles:
            tile_result = process_tile_mag1c_sas(tile, methane_spectrum, sample_ratio)
            # Add channel dimension for stitching: (H, W) -> (1, H, W)
            result_tiles.append(np.expand_dims(tile_result, axis=0))
        
        # Stitch results back together
        # Update tiling_info for single-channel output
        tiling_info['original_shape'] = (1, image.shape[1], image.shape[2])
        result = stitch_tiles(result_tiles, tiling_info)
        return result.squeeze(axis=0)  # Remove channel dimension: (1, H, W) -> (H, W)
    else:
        return process_tile_mag1c_sas(image, methane_spectrum, sample_ratio)


def process_tile_mag1c_sas(image, methane_spectrum, sample_ratio):
    """
    Process a single tile/image through Mag1c-SAS.
    
    Args:
        image: Hyperspectral image of shape (C, H, W).
        methane_spectrum: Methane absorption spectrum template.
        sample_ratio: Ratio of pixels to sample for SAS.
    
    Returns:
        Mag1c-SAS result of shape (1, H, W).
    """
    c, h, w = image.shape
    
    # Create mask for valid pixels
    valid_mask_flat = compute_valid_mask(image)
    
    # Check if there are any valid pixels
    if not np.any(valid_mask_flat):
        return np.zeros((h, w), dtype=np.float32)
    
    image_batched = preprocess_image(image, valid_mask_flat)
    
    # Compute sampling indices
    indices = compute_sampling_indices(image_batched.shape[1], sample_ratio)
    
    # Run Mag1c-SAS
    result = compute_base_mag1c_SAS(image_batched, methane_spectrum, indices)
    
    return postprocess_result(result, valid_mask_flat, (1, h, w))
