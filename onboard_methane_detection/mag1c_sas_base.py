# ==============================================================================
# Credits
# This file contains edited code from the Mag1c repository 
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

def compute_base_mag1c_SAS(rdn_data, spec, indices):
    """
    Compute Mag1c-SAS (Matched Filter with Iterative Covariance) on radiance data.
    
    Args:
        rdn_data: Radiance data of shape (B, N, C) where B is batch size,
                  N is number of pixels, C is number of channels.
        spec: Methane absorption spectrum template of shape (1, 1, C).
        indices: Array of pixel indices to sample for covariance estimation.
    
    Returns:
        Matched filter output of shape (B, N, 1).
    """
    rdn_data_sample = rdn_data[:, indices, :]
    mu, Cit, normalizer = acrwl1mf(
        x=rdn_data_sample,
        template=spec,
        num_iter=30,
        sample=True,
    )
    mf_out = acrwl1mf_compact(
        x=rdn_data,
        normalizer=normalizer,
        num_iter=3,
        mu=mu,
        Cit=Cit,
    )
    return mf_out

def acrwl1mf(x, template, num_iter, sample=False):
    """
    Adaptive Covariance Regularized Weighted L1 Matched Filter.
    
    Args:
        x: Input data of shape (B, N, C) where B is batch size,
           N is number of pixels, C is number of channels.
        template: Methane absorption spectrum template of shape (1, 1, C).
        num_iter: Number of refinement iterations.
        sample: If True, return intermediate values (mu, Cit, normalizer) 
                for use with acrwl1mf_compact.
    
    Returns:
        If sample=True:
            mu: Mean of shape (B, 1, C).
            Cit: Covariance inverse times template of shape (B, C, 1).
            normalizer: Normalizer of shape (B, 1, 1).
        If sample=False:
            mf: Matched filter output of shape (B, N, 1).
            R: Reflectance ratio of shape (B, N, 1).
    """
    N = x.shape[1]
    
    scaling = 1e5
    epsilon = 1e-9

    # 1. Initial Statistics
    mu = np.mean(x, axis=1, keepdims=True)
    mu_T = np.swapaxes(mu, 1, 2)
    R = (x @ mu_T) / (mu @ mu_T)
    
    target = template * mu
    xmean = x - mu
    
    # C: (B, C, C) Covariance matrix
    xmean_T = np.swapaxes(xmean, 1, 2)
    C = (xmean_T @ xmean) / N

    # 2. Initial Solve (VECTORIZED OPTIMIZATION)
    target_T = np.swapaxes(target, 1, 2)
    
    # Replaced the slow 'for b in range(B)' loop with a single batched call.
    # np.linalg.solve(C, target_T) solves the system C * Cit = target_T for all batches.
    Cit = np.linalg.solve(C, target_T)

    normalizer = target @ Cit
    
    # Initial Matched Filter
    mf = ((x - mu) @ Cit) / (R * normalizer)
    mf = np.maximum(mf, 0) # ReLU

    # 3. Iterative Refinement
    for i in range(num_iter):
        regularizer = 1.0 / (R * (mf + epsilon))
        
        # Update Modified X
        modx = x - (R * mf * target)
        
        # Update Statistics
        mu = np.mean(modx, axis=1, keepdims=True)
        target = template * mu
        xmean = modx - mu
        
        xmean_T = np.swapaxes(xmean, 1, 2)
        C = (xmean_T @ xmean) / N
        
        # Update Cit (VECTORIZED OPTIMIZATION)
        target_T = np.swapaxes(target, 1, 2)
        Cit = np.linalg.solve(C, target_T)

        # Update Normalizer
        normalizer = target @ Cit
        
        # Clamp normalizer to min=1
        normalizer = np.maximum(normalizer, 1)

        # Check for sample return condition
        if sample and (i + 1 == num_iter):
            return mu, Cit, normalizer

        # Update Matched Filter with Regularization
        mf_numerator = ((x - mu) @ Cit) - regularizer
        mf = mf_numerator / (R * normalizer)
        mf = np.maximum(mf, 0) # ReLU

    mf = mf * scaling
    return mf, R

def acrwl1mf_compact(x, normalizer, num_iter, mu, Cit):
    """
    Compact version of ACRWL1MF using precomputed statistics.
    
    Args:
        x: Input data of shape (B, N, C) where B is batch size,
           N is number of pixels, C is number of channels.
        normalizer: Precomputed normalizer of shape (B, 1, 1).
        num_iter: Number of regularization iterations.
        mu: Precomputed mean of shape (B, 1, C).
        Cit: Precomputed covariance inverse times template of shape (B, C, 1).
    
    Returns:
        mf: Matched filter output of shape (B, N, 1).
    """
    scaling = 1e5
    epsilon = 1e-9

    # 1. Compute R
    mu_T = np.swapaxes(mu, 1, 2)
    R = (x @ mu_T) / (mu @ mu_T)

    # 2. Clamp Normalizer
    normalizer = np.maximum(normalizer, 1)

    # 3. Compute Initial Matched Filter (mf_0)
    mf_numerator = (x - mu) @ Cit
    mf_denominator = R * normalizer
    
    mf_0 = mf_numerator / mf_denominator
    mf_0 = np.maximum(mf_0, 0) # ReLU
    
    mf = mf_0.copy()

    # 4. Iterative Regularization
    for i in range(num_iter):
        regularizer = 1.0 / (R * (mf + epsilon))
        
        # mf = mf_0 - (regularizer / mf_denominator)
        mf = mf_0 - (regularizer / mf_denominator)
        
        mf = np.maximum(mf, 0) # ReLU

    # 5. Final Scaling
    mf = mf * scaling
    
    return mf