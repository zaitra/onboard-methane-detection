# Onboard Methane Detection

This library provides an end-to-end pipeline for processing hyperspectral imagery to detect methane emissions. It includes tools for band selection, methane spectrum generation, preprocessing via Mag1c SAS, and model inference.

## Citation [![arXiv:2606.03675](https://img.shields.io/badge/arXiv-2606.03675-blue)](https://doi.org/10.48550/arXiv.2606.03675)
If you find our research useful, please cite our article:
```bibtex
@misc{herec2026fastmethanedetectionpipeline,
      title={A Fast Methane Detection Pipeline on Board Satellites Based on Mag1c-SAS and LinkNet}, 
      author={Jonáš Herec and Vít Růžička and Rado Pitoňák and Jan Sedmidubsky},
      year={2026},
      eprint={2606.03675},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03675}, 
}
```

## Installation

```bash
pip install onboard-methane-detection
```

### Dependencies and Performance

Onboard Methane Detection is lightweight and depends only on NumPy for processing. ONNX Runtime is required only if you want inference. You can omit ONNX Runtime by using the lightweight build (see `Lightweight Onboard Package Build` below).

Important: NumPy can be very slow without an acceleration library, so make sure one is installed on your device. We tested with [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS); the build was about 3.9 MB on ARM Cortex-A53 when compiled without LAPACK. After installing OpenBLAS, reinstall NumPy via `pip` so it can be detected and used.

## End-to-End Example

The following example demonstrates a complete workflow, divided into distinct stages: on-ground preparation, and onboard session execution. The same example is available in
<a href="https://colab.research.google.com/github/zaitra/onboard-methane-detection/blob/main/showcase.ipynb">showcase.ipynb <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a>.

### On-Ground

```python
# Core imports for on-ground processing
import numpy as np
from onboard_methane_detection import (
    select_the_bands_by_transmittance, 
    generate_methane_spectrum, 
    select_rgb_bands
)

# 1. Define or load sensor specifications
# wavelengths = [...]  # List of all available sensor wavelengths
# fwhms = [...]        # List of corresponding FWHMs

# 2. Select RGB bands
rgb_wavelengths, rgb_indices = select_rgb_bands(wavelengths)

# 3. Select bands specifically for Methane detection
# First, generate the baseline CH4 spectrum across all sensor bands
ch4_spectrum_full = generate_methane_spectrum(wavelengths, fwhms)

# Then, select the most informative bands based on transmittance
selected_wvs_ch4, selected_ch4_spectrum = select_the_bands_by_transmittance(
    wavelengths, ch4_spectrum_full, N=50, strategy="highest-variance"
)
selected_indices_ch4 = [wavelengths.tolist().index(w) for w in selected_wvs_ch4]

# 4. Export artifacts for the onboard inference session
np.save('rgb_indices.npy', np.array(rgb_indices))
np.save('selected_indices_ch4.npy', np.array(selected_indices_ch4))
np.save('selected_ch4_spectrum.npy', selected_ch4_spectrum)
```

### Onboard

```python
import numpy as np

# Core imports for onboard execution
from onboard_methane_detection import (
    mag1c_sas,
    initialize_model,
    normalize_image,
    model_inference
)

# 1. Load exported artifacts (indices and spectrum) from On-Ground preparation
rgb_indices = np.load('rgb_indices.npy')
selected_indices_ch4 = np.load('selected_indices_ch4.npy')
selected_ch4_spectrum = np.load('selected_ch4_spectrum.npy')

# 2. Session Initialization
# Initialize the inference model before the processing loop begins
session, input_name = initialize_model(dynamic_output_size=True)

# 3. Processing Images (Onboard Loop)
# acquired_images: list of images in shape (C, H, W)

for image in acquired_images:
    # Extract RGB channels using previously found indices
    rgb_image = image[rgb_indices, :, :]
    
    # Extract the CH4 detection bands from the image
    methane_bands_image = image[selected_indices_ch4, :, :]
    
    # Apply Mag1c SAS preprocessing.
    # Optional tiling args (`tiling`, `tile_size`) can be used if the image is too large.
    mag1c_output = mag1c_sas(methane_bands_image, selected_ch4_spectrum, tiling=False)
    
    # Concatenate Mag1c SAS output with local RGB context.
    # Correct channel order is R, G, B, Mag1c-SAS.
    model_input = np.concatenate([rgb_image, mag1c_output], axis=0)
    
    # Normalize input: sensor_or_factors can be "emit", "aviris_ng", 
    # or a custom array of 4 division factors for a different sensor.
    model_input = normalize_image(model_input, sensor_or_factors='emit') 
    
    # Run model inference.
    # Tiling arguments can also be provided here if the image is too large.
    predictions = model_inference(session, input_name, model_input, use_tiling=False, logits_to_probs=True)
    
    print("Processed image and generated predictions.")
```

## Lightweight Onboard Package Build

For deployment on resource-constrained satellite hardware, a build script is provided that generates a minimal version of this library with only the modules you need. It supports optional subpackages (`processing`, `inference`, `onground`) and depends only on NumPy by default, adding ONNX Runtime only when inference is included.

For full usage instructions, available options, and details on what each subpackage includes, see the [GitHub repository](https://github.com/zaitra/onboard-methane-detection).

# Acknowledgments
A huge thank you to the creators of [Mag1c](https://github.com/markusfoote/mag1c). This project uses core files and edited code from their original repository, and it wouldn't have been possible without their foundational work!
