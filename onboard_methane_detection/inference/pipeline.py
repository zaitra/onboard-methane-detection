import numpy as np
from .utils import run_inference_full, pad_to_32, reverse_32_padding
from ..processing.utils import tile_image, stitch_tiles

def model_inference(session, input_name, image, use_tiling=True, tile_size=512, logits_to_probs=False):
    """
    Pipeline for performing model inference on a CHW image with optional tiling.
    
    Args:
        session: Initialized ONNX inference session.
        input_name: Input node name for the ONNX session.
        image: Input image of shape (C, H, W) or (B, C, H, W).
        use_tiling: If True, process the image in tiles. If False, process the whole image at once.
        tile_size: Size of tiles (default 512).
        logits_to_probs: If True, apply sigmoid to convert logits to probabilities.
    
    Returns:
        Model inference result of shape (1, H, W).
    """
    is_chw = image.ndim == 3
    if is_chw:
        image = np.expand_dims(image, axis=0)

    if not use_tiling:
        padded_image, pad_info = pad_to_32(image)
        result = _process_tile_inference(session, input_name, padded_image, logits_to_probs)
        result_unpadded = reverse_32_padding(result, pad_info)
        return result_unpadded

    # Tile the image
    tiles, tiling_info = tile_image(image, tile_size)
    
    # Process each tile
    result_tiles = []
    for tile in tiles:
        tile_result = _process_tile_inference(session, input_name, tile, logits_to_probs)
        result_tiles.append(tile_result)
    
    # Stitch results back together
    # Update tiling_info for single-channel output
    orig_bchw = tiling_info['original_shape']
    orig_h, orig_w = orig_bchw[-2:]
    tiling_info['original_shape'] = (1, orig_h, orig_w)
    
    result = stitch_tiles(result_tiles, tiling_info)
    return result


def _process_tile_inference(session, input_name, image_batched, logits_to_probs):
    """
    Process a single tile/image through model inference.
    
    Args:
        session: Initialized ONNX inference session.
        input_name: Input node name for the ONNX session.
        image_batched: Input image of shape (B, C, H, W).
        logits_to_probs: If True, apply sigmoid to convert logits to probabilities.
    
    Returns:
        Model inference result of shape (1, H, W).
    """
    image_batched = image_batched.astype(np.float32)
    
    # Run inference
    result = run_inference_full(session, input_name, image_batched, logits_to_probs=logits_to_probs)
    
    return result