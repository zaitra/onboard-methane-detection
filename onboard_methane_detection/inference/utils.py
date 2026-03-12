import os
import numpy as np
import onnxruntime as ort


def initialize_model(model_path: str = None, dynamic_output_size: bool = False) -> tuple[ort.InferenceSession, str]:
    """
    Initialize and return the ONNX inference session and input name.
    """
    if model_path is None:
        model_filename = "linknet_exported_dynamic.onnx" if dynamic_output_size else "linknet_exported.onnx"
        print(f"Using model: {model_filename}")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_sole_inference(session: ort.InferenceSession, input_name: str, image: np.ndarray) -> np.ndarray:
    """
    Run the inference session with the given image.
    """
    outputs = session.run(None, {input_name: image})
    return outputs[0]


def postprocess_result(result: np.ndarray, logits_to_probs: bool = False) -> np.ndarray:
    """
    Postprocess the inference results, converting logits to probabilities if requested.
    """
    if logits_to_probs:
        result = 1 / (1 + np.exp(-result))  # Sigmoid function
    return result[0]  # Remove batch dimension if present


def normalize_image(image: np.ndarray, sensor_or_factors: str | list | np.ndarray = 'emit') -> np.ndarray:
    """
    Normalize a CHW image and return BCHW.

    Args:
        image: Numpy array of shape (C, H, W). If BCHW is provided, it is used as-is.
        sensor_or_factors: 'emit', 'aviris_ng', or a list/array of length 4 containing custom division factors.
    """
    if isinstance(sensor_or_factors, str):
        if sensor_or_factors.lower() == 'emit':
            factors = np.array([98.91703491210939, 112.33824462890625, 119.84940185546876, 2436.562744140625], dtype=image.dtype)
        elif sensor_or_factors.lower() == 'aviris_ng':
            factors = np.array([247.29258728, 280.84561157, 299.62350464, 1218.28137207], dtype=image.dtype)
        else:
            raise ValueError("String must be 'emit' or 'aviris_ng'")
    elif isinstance(sensor_or_factors, (list, np.ndarray)) and len(sensor_or_factors) == 4:
        factors = np.array(sensor_or_factors, dtype=image.dtype)
    else:
        raise ValueError("Must be 'emit', 'aviris_ng', or a custom list/array of length 4")
        
    if image.ndim == 3:
        image_bchw = image[None, :, :, :]
    elif image.ndim == 4:
        image_bchw = image
    else:
        raise ValueError("Image must be CHW or BCHW")

    factors = factors[None, :, None, None]
    normalized = image_bchw / factors
    return np.clip(normalized, 0, 2)


def run_inference_full(session: ort.InferenceSession, input_name: str, image: np.ndarray, logits_to_probs: bool = False) -> np.ndarray:
    """
    Load an ONNX model and perform inference on a numpy image.
    
    Args:
        session: Initialized ONNX inference session.
        input_name: Input node name for the ONNX session.
        image: Input numpy image array of shape (B, C, H, W) where:
               - B is batch size
               - C is number of channels (4)
               - H is image height
               - W is image width
        logits_to_probs: If True, apply sigmoid to convert logits to probabilities 
                         (for binary segmentation).
    
    Returns:
        Inference output as numpy array of shape (B, 1, H, W).
        Contains probabilities if logits_to_probs=True, else raw logits.
    """
    result = run_sole_inference(session, input_name, image)
    return postprocess_result(result, logits_to_probs)


def pad_to_32(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Pad a BCHW image so its spatial dimensions (H, W) are divisible by 32.
    Returns the padded image and a dictionary with padding information.
    """
    b, c, orig_h, orig_w = image.shape
    pad_h = (32 - (orig_h % 32)) % 32
    pad_w = (32 - (orig_w % 32)) % 32
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded_image = np.pad(image, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    pad_info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'pad_h': pad_h,
        'pad_w': pad_w
    }
    return padded_image, pad_info


def reverse_32_padding(result_image: np.ndarray, pad_info: dict) -> np.ndarray:
    """
    Crop the padded inference result back to its original shape.
    Assumes result_image is of shape (1, H, W).
    """
    if pad_info['pad_h'] > 0 or pad_info['pad_w'] > 0:
        pad_top = pad_info['pad_top']
        pad_left = pad_info['pad_left']
        orig_h = pad_info['orig_h']
        orig_w = pad_info['orig_w']
        return result_image[:, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]
    return result_image
