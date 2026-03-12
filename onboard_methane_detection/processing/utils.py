import numpy as np


def tile_image(image, tile_size=512):
    """
    Tile an image of shape CHW or BCHW into tiles of specified size.
    If image is not divisible by tile_size, it will be padded evenly from all sides.
    
    Args:
        image: Input image of shape (C, H, W) or (B, C, H, W).
        tile_size: Size of each tile (default 512).
    
    Returns:
        tiles: List of tiles.
        tiling_info: Dictionary containing information needed for stitching.
    """
    is_bchw = image.ndim == 4
    if is_bchw:
        b, c, h, w = image.shape
        original_shape = (b, c, h, w)
    else:
        c, h, w = image.shape
        original_shape = (c, h, w)
    
    # Calculate padding needed to make dimensions divisible by tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    
    # Distribute padding evenly on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Pad the image (pad only H and W dimensions)
    if is_bchw:
        pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        
    padded_image = np.pad(
        image,
        pad_width,
        mode='constant',
        constant_values=0,
    )
    
    padded_h, padded_w = padded_image.shape[-2:]
    
    # Calculate grid shape
    n_rows = padded_h // tile_size
    n_cols = padded_w // tile_size
    
    # Extract tiles
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * tile_size
            x_start = col * tile_size
            if is_bchw:
                tile = padded_image[:, :, y_start:y_start + tile_size, x_start:x_start + tile_size]
            else:
                tile = padded_image[:, y_start:y_start + tile_size, x_start:x_start + tile_size]
            tiles.append(tile)
    
    tiling_info = {
        'original_shape': original_shape,
        'padding': (pad_top, pad_bottom, pad_left, pad_right),
        'tile_size': tile_size,
        'grid_shape': (n_rows, n_cols),
    }
    
    return tiles, tiling_info


def stitch_tiles(tiles, tiling_info):
    """
    Stitch tiles back into the original image shape, removing padding.
    Always returns CHW format.
    """
    n_rows, n_cols = tiling_info['grid_shape']
    tile_size = tiling_info['tile_size']
    original_shape = tiling_info['original_shape']
    pad_top, _, pad_left, _ = tiling_info['padding']
    
    # Ensure original_shape variables for CHW
    orig_c, orig_h, orig_w = original_shape
    
    # Create output image of original shape (always CHW)
    output = np.zeros((orig_c, orig_h, orig_w), dtype=tiles[0].dtype)
    
    # Place tiles directly into output, accounting for padding
    tile_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            # Position in padded space
            y_pad_start = row * tile_size
            x_pad_start = col * tile_size
            
            # Calculate overlap with original image
            y_src_start = max(0, pad_top - y_pad_start)
            x_src_start = max(0, pad_left - x_pad_start)
            y_src_end = min(tile_size, pad_top + orig_h - y_pad_start)
            x_src_end = min(tile_size, pad_left + orig_w - x_pad_start)
            
            # Position in output image
            y_dst_start = max(0, y_pad_start - pad_top)
            x_dst_start = max(0, x_pad_start - pad_left)
            y_dst_end = y_dst_start + (y_src_end - y_src_start)
            x_dst_end = x_dst_start + (x_src_end - x_src_start)
            
            if y_src_start < y_src_end and x_src_start < x_src_end:
                current_tile = tiles[tile_idx]
                output[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
                    current_tile[:, y_src_start:y_src_end, x_src_start:x_src_end]
            
            tile_idx += 1
    
    return output


def compute_valid_mask(image, censor_value=0):
    """Create a mask for invalid pixels (zeros in all channels).

    Args:
        image: Input image of shape (C, H, W).
        censor_value: Value that indicates invalid data (default 0).

    Returns:
        mask: Boolean array of shape (H*W) where True indicates valid pixels.
    """
    # A pixel is invalid if all channels equal the censor value
    valid_mask = ~np.all(image == censor_value, axis=0)
    return valid_mask.reshape(-1)

def compute_sampling_indices(pixel_n, sample_ratio=0.01):
    """
    Compute indices for sampling pixels based on the given ratio.
    """
    sample_size = max(1, int(sample_ratio * pixel_n))
    step_size = max(1, pixel_n // sample_size)
    return np.arange(0, pixel_n, step_size)[:sample_size]


def preprocess_image(image, valid_mask_flat):
    """
    Hyper-optimized for ARM A53: Uses strictly sequential memory access.
    """
    # 1. View as (C, H*W) - 0 cost, no memory moved.
    image_flat = image.reshape(image.shape[0], -1)
    
    # 2. Extract valid pixels using compress. 
    # This evaluates channel-by-channel sequentially. 
    # The A53 cache prefetcher will love this.
    valid_pixels = np.compress(valid_mask_flat, image_flat, axis=1) # Shape: (C, N_valid)
    
    # 3. Transpose to (N_valid, C) and add batch dim - 0 cost views.
    return valid_pixels.T[None, ...]


def postprocess_result(result, valid_mask_flat, shape):
    """
    Postprocess the flat batched result back to the original image shape.
    
    Args:
        result: The batched flat result.
        valid_mask_flat: Flattened boolean mask of valid pixels.
        shape: Original shape of the image in (C, H, W) format.
    """
    _, h, w = shape
    output = np.zeros(h * w, dtype=result.dtype)
    output[valid_mask_flat] = result.squeeze()
    return output.reshape(1, h, w)
