# ==============================================================================
# Credits
# This file contains edited code and use edited files from the Mag1c repository 
# (https://github.com/markusfoote/mag1c).
# 
# Original Copyright: 
# BSD 3-Clause License
#
# Copyright (c) 2019,
#   Scientific Computing and Imaging Institute and
#   Utah Remote Sensing Applications Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import numpy as np
import os

def get_mask_bad_bands(wave: np.ndarray, use_all_bands: bool) -> np.ndarray:
    """Filters out bands outside of the main methane transmittance regions seen in STARCOP.
    Rejects wavelengths: - Below 1570 nm
                         - Above 2490 nm
                         - Between 1700-2000 nm (water absorption / low CH4 sensitivity region)

    :param wave: Vector of wavelengths to evaluate.
    :param use_all_bands: If True, all bands are kept regardless of wavelength.
    :return: Boolean mask array where True indicates a band to keep.
    """
    if use_all_bands:
        return np.ones_like(wave, dtype=bool)
    else:
        return ~(
            np.logical_or(
                np.logical_or(wave < 1570, wave > 2490),
                np.logical_and(wave > 1700, wave < 2000),
            )
        )

def generate_methane_spectrum(
    centers, fwhm, save=False,
):
    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers.
    :param fwhm: full width half maximum for the gaussian kernel of each band.
    :return template: the unit absorption spectum
    """
    # import scipy.stats
    SCALING = 1e5
    centers = np.asarray(centers)
    fwhm = np.asarray(fwhm)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError(
            "Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf)."
        )
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError(
            "Length of band center wavelengths and band fwhm arrays must be equal."
        )
    rads = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rads.npy"))
    wave = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wv.npy"))
    concentrations = np.asarray([0, 500, 1000, 2000, 4000, 8000, 16000])
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Evaluate normal distribution explicitly
    var = sigma**2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-((wave[:, None] - centers[None, :]) ** 2) / (2 * var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(
        response, response.sum(axis=0), where=response.sum(axis=0) > 0, out=response
    )
    # implement resampling as matrix multiply
    resampled = rads.dot(response)
    lograd = np.log(resampled, out=np.zeros_like(resampled), where=resampled > 0)
    slope, _, _, _ = np.linalg.lstsq(
        np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None
    )
    spectrum = slope[1, :] * SCALING
    return spectrum

def select_the_bands_by_transmittance(
    wavelengths,
    ch4_transmittance,
    N,
    strategy,
    use_all_bands: bool = False,
):
    """
    Selects N spectral bands based on methane (CH₄) transmittance using different selection strategies.
    
    Parameters
    ----------
    wavelengths : numpy.ndarray
        1D array of available wavelengths.
    
    ch4_transmittance : numpy.ndarray
        1D array of CH₄ transmittance values corresponding to the given wavelengths.
    
    N : int
        The number of bands to select.
    
    strategy : str
        The method used for selecting bands:
        - 'highest-transmittance': Selects bands with the highest absolute transmittance.
        - 'highest-variance': Selects bands that maximize transmittance variance across selected bands.
        - 'evenly-spaced': Selects bands evenly spaced within the methane-sensitive wavelength range (2122-2488 nm).
    use_all_bands : bool, optional
        If True, do not censor bad wavelength regions. Defaults to False.

    Returns
    -------
    selected_wavelengths : numpy.ndarray
        1D array of the selected wavelengths.
    
    selected_transmittance : numpy.ndarray
        1D array of the corresponding CH₄ transmittance values.
    """
    # Ensure input arrays have matching lengths
    if len(wavelengths) != len(ch4_transmittance):
        raise ValueError("Wavelengths and transmittance arrays must have the same length.")

    # Convert to arrays and keep originals for index mapping
    wavelengths = np.asarray(wavelengths)
    ch4_transmittance = np.asarray(ch4_transmittance)
    wavelengths_all = wavelengths
    ch4_transmittance_all = ch4_transmittance

    # Extract red RGB index from the original (unfiltered) wavelength list
    if np.min(wavelengths_all) <= 800:
        _, rgb_indices = select_rgb_bands(wavelengths_all, [640])
        red_idx_orig = int(rgb_indices[0])
    else:
        red_idx_orig = None

    # Censor bad wavelength regions before selection (keep original indices)
    good_band_mask = get_mask_bad_bands(wavelengths_all, use_all_bands)
    good_band_indices = np.where(good_band_mask)[0]
    wavelengths = wavelengths_all[good_band_mask]
    ch4_transmittance = ch4_transmittance_all[good_band_mask]

    if N > len(wavelengths):
        raise ValueError("The number of bands to be selected is greater than number of bands provided.")

    if strategy == 'highest-transmittance':
        # Select N bands with the highest absolute transmittance values
        selected_indices = np.argsort(np.abs(ch4_transmittance))[::-1][:N]

    elif strategy == 'highest-variance':
        # Build candidate pool from good bands but ensure red band (original
        # index) is included even if filtered out by the mask. Selection is
        # performed on transmittance values pulled from the original arrays
        # so we preserve the original index space for the final output.
        candidate_indices = list(good_band_indices)
        if red_idx_orig not in candidate_indices and red_idx_orig is not None:
            candidate_indices = [red_idx_orig] + candidate_indices

        candidate_trans = ch4_transmittance_all[candidate_indices]

        if N > len(candidate_indices):
            raise ValueError(f"Requested {N} bands, but only {len(candidate_indices)} available in candidate pool.")

        # Start with red at position 0 in the candidate list
        if red_idx_orig is not None:
            selected_rel = [0]
        else:
            selected_rel = [np.argmax(np.abs(candidate_trans))]

        # Iteratively select the next N-1 bands to maximize variance
        for _ in range(N - 1):
            remaining = [i for i in range(len(candidate_trans)) if i not in selected_rel]
            min_diffs = [min(np.abs(candidate_trans[selected_rel] - candidate_trans[i])) for i in remaining]
            selected_rel.append(remaining[int(np.argmax(min_diffs))])

        # Map relative candidate indices back to original indices
        selected_indices = np.asarray(candidate_indices)[selected_rel].astype(int)

    elif strategy == 'evenly-spaced':
        # Filter indices within the methane-sensitive range
        methane_mask = (wavelengths >= 2122) & (wavelengths <= 2488)
        methane_indices = np.where(methane_mask)[0]

        if N > len(methane_indices):
            raise ValueError(f"Requested {N} bands, but only {len(methane_indices)} available in range.")

        # Select N evenly spaced indices
        selected_indices = methane_indices[np.linspace(0, len(methane_indices) - 1, N, dtype=int)]

    else:
        raise ValueError("Invalid strategy. Choose from 'highest_transmittance', 'highest_variance', or 'evenly_spaced'.")

    # Map selected indices back to original arrays. For 'highest-variance'
    # we already produced original indices; for other strategies we produced
    # indices relative to the filtered/good-band list and must translate via
    # `good_band_indices`.
    if strategy == 'highest-variance':
        selected_orig_indices = np.asarray(selected_indices)
    else:
        selected_orig_indices = good_band_indices[np.asarray(selected_indices)]

    # Final selection of wavelengths and transmittance (original index space)
    return wavelengths_all[selected_orig_indices], ch4_transmittance_all[selected_orig_indices]

def select_rgb_bands(wavelengths, rgb_wavelengths=[640, 550, 460]):
    """
    Selects the bands closest to the specified RGB wavelengths.
    
    Parameters
    ----------
    wavelengths : list or numpy.ndarray
        Array of available wavelengths.
    rgb_wavelengths : list, optional
        List of target wavelengths. Defaults to the [640, 550, 460].

    Returns
    -------
    selected_wavelengths : list
        List of the closest matching wavelengths.
    selected_indices : list
        List of indices corresponding to the selected wavelengths.
    """
    wavelengths_arr = np.asarray(wavelengths)
    selected_indices = []
    selected_wavelengths = []
    
    for target in rgb_wavelengths:
        idx = np.argmin(np.abs(wavelengths_arr - target))
        selected_indices.append(int(idx))
        selected_wavelengths.append(float(wavelengths_arr[idx]))
        
    return np.asarray(selected_wavelengths), np.asarray(selected_indices)
