r"""
Signal Processing Domain Module

This module maps theta to signal processing systems including
compression, filtering, and reconstruction.

Theta Mapping:
    theta -> 0: Corrupted/noisy/compressed signal
    theta -> 1: Clean/reconstructed/lossless signal
    theta = SNR / SNR_target: Signal quality
    theta = 1 - distortion: Compression quality
    theta = sparsity: Compressive sensing recovery

Key Features:
    - SNR and signal quality
    - Rate-distortion analysis
    - Compressive sensing
    - Filter analysis

References:
    @book{Oppenheim2009,
      author = {Oppenheim, Alan V. and Schafer, Ronald W.},
      title = {Discrete-Time Signal Processing},
      publisher = {Pearson},
      year = {2009}
    }
    @article{Candes2006,
      author = {Cand\`{e}s, Emmanuel J. and Romberg, Justin and Tao, Terence},
      title = {Robust uncertainty principles: exact signal reconstruction from
               highly incomplete frequency information},
      journal = {IEEE Trans. Inf. Theory},
      year = {2006}
    }
    @article{Shannon1948,
      author = {Shannon, Claude E.},
      title = {A mathematical theory of communication},
      journal = {Bell Syst. Tech. J.},
      year = {1948}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

# Target SNR in dB
SNR_TARGET_DB = 40.0

# Maximum compression ratio for "acceptable" quality
COMPRESSION_RATIO_MAX = 10.0


# =============================================================================
# Enums for Classification
# =============================================================================

class SignalQuality(Enum):
    """Classification of signal quality."""
    CORRUPTED = "corrupted"          # SNR < 0 dB
    NOISY = "noisy"                  # 0 <= SNR < 20 dB
    CLEAN = "clean"                  # 20 <= SNR < 40 dB
    PRISTINE = "pristine"            # SNR >= 40 dB


class ProcessingDomain(Enum):
    """Signal processing domain."""
    TIME = "time"                    # Time domain
    FREQUENCY = "frequency"          # Frequency domain
    TIME_FREQUENCY = "time_frequency"  # STFT, wavelets
    SPARSE = "sparse"                # Sparse domain


class CompressionLevel(Enum):
    """Compression level classification."""
    LOSSLESS = "lossless"            # No loss
    NEAR_LOSSLESS = "near_lossless"  # Imperceptible loss
    LOSSY = "lossy"                  # Noticeable loss
    EXTREME = "extreme"              # Significant loss


class FilterType(Enum):
    """Filter type classification."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    ALLPASS = "allpass"


# =============================================================================
# Dataclass for Signal Systems
# =============================================================================

@dataclass
class SignalSystem:
    """
    A signal processing system.

    Attributes:
        name: Descriptive name
        sampling_rate: Sampling rate in Hz
        snr_db: Signal-to-noise ratio in dB
        compression_ratio: Compression ratio
        bandwidth: Signal bandwidth in Hz
        dynamic_range_db: Dynamic range in dB
        sparsity: Signal sparsity (fraction of nonzero)
        bit_depth: Quantization bits
    """
    name: str
    sampling_rate: float = 44100.0
    snr_db: float = 40.0
    compression_ratio: float = 1.0
    bandwidth: float = 20000.0
    dynamic_range_db: float = 96.0
    sparsity: float = 0.1
    bit_depth: int = 16


# =============================================================================
# SNR Analysis
# =============================================================================

def snr_db_to_linear(snr_db: float) -> float:
    """
    Convert SNR from dB to linear ratio.

    Args:
        snr_db: SNR in dB

    Returns:
        Linear SNR ratio
    """
    return 10**(snr_db / 10)


def snr_linear_to_db(snr_linear: float) -> float:
    """
    Convert SNR from linear to dB.

    Args:
        snr_linear: Linear SNR ratio

    Returns:
        SNR in dB
    """
    if snr_linear <= 0:
        return float('-inf')
    return 10 * np.log10(snr_linear)


def compute_snr_theta(
    snr_db: float,
    target_db: float = SNR_TARGET_DB
) -> float:
    r"""
    Compute theta from SNR.

    Higher SNR = higher theta.

    Args:
        snr_db: Actual SNR in dB
        target_db: Target SNR in dB

    Returns:
        theta in [0, 1]

    Reference: \cite{Shannon1948}
    """
    if target_db <= 0:
        return 0.0

    # Normalize: 0 dB -> 0, target_db -> 1
    theta = snr_db / target_db
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Compression Analysis
# =============================================================================

def compute_compression_theta(
    ratio: float,
    max_ratio: float = COMPRESSION_RATIO_MAX,
    is_lossless: bool = False
) -> float:
    r"""
    Compute theta from compression ratio.

    Lower ratio (less compression) = higher theta for quality.
    Lossless gives theta = 1.

    Args:
        ratio: Compression ratio (uncompressed/compressed size)
        max_ratio: Maximum acceptable ratio
        is_lossless: Whether compression is lossless

    Returns:
        theta in [0, 1]
    """
    if is_lossless:
        return 1.0

    if ratio <= 1:
        return 1.0  # No compression
    if max_ratio <= 1:
        return 0.0

    theta = 1 - (ratio - 1) / (max_ratio - 1)
    return np.clip(theta, 0.0, 1.0)


def rate_distortion(
    rate: float,
    sigma2: float = 1.0
) -> float:
    r"""
    Compute distortion from rate for Gaussian source.

    D(R) = sigma^2 * 2^(-2R)

    Args:
        rate: Rate in bits per sample
        sigma2: Source variance

    Returns:
        Distortion

    Reference: \cite{Shannon1948}
    """
    if rate <= 0:
        return sigma2
    return sigma2 * 2**(-2 * rate)


def compute_distortion_theta(
    distortion: float,
    target_distortion: float = 0.01
) -> float:
    r"""
    Compute theta from distortion.

    Lower distortion = higher theta.

    Args:
        distortion: Actual distortion
        target_distortion: Target distortion

    Returns:
        theta in [0, 1]
    """
    if distortion <= 0:
        return 1.0
    if target_distortion <= 0:
        return 0.0

    theta = target_distortion / distortion
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Compressive Sensing
# =============================================================================

def compute_sparsity_theta(
    sparsity: float,
    measurements_ratio: float
) -> float:
    r"""
    Compute theta for compressive sensing recovery.

    Recovery possible if m >= C * k * log(n/k)
    where m = measurements, k = sparsity, n = signal length.

    Args:
        sparsity: Fraction of nonzero coefficients (k/n)
        measurements_ratio: Fraction of measurements (m/n)

    Returns:
        theta in [0, 1]: 1 = recovery guaranteed

    Reference: \cite{Candes2006}
    """
    if sparsity <= 0:
        return 1.0  # Zero signal, trivially sparse
    if sparsity >= 1:
        return 0.0  # Not sparse

    # Approximate condition: m/n >= C * (k/n) * log(1/sparsity)
    required_ratio = 4 * sparsity * np.log(1 / sparsity)

    if measurements_ratio >= required_ratio:
        theta = 1.0
    else:
        theta = measurements_ratio / required_ratio

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Filter Analysis
# =============================================================================

def compute_filter_theta(
    passband_ripple_db: float,
    stopband_attenuation_db: float,
    transition_width: float
) -> float:
    r"""
    Compute theta for filter quality.

    Good filter: low ripple, high attenuation, narrow transition.

    Args:
        passband_ripple_db: Passband ripple in dB (smaller = better)
        stopband_attenuation_db: Stopband attenuation in dB (larger = better)
        transition_width: Transition band width (fraction of bandwidth)

    Returns:
        theta in [0, 1]

    Reference: \cite{Oppenheim2009}
    """
    # Ripple contribution (0.1 dB target)
    ripple_theta = min(1.0, 0.1 / max(0.01, passband_ripple_db))

    # Attenuation contribution (60 dB target)
    atten_theta = min(1.0, stopband_attenuation_db / 60)

    # Transition width contribution (5% target)
    width_theta = min(1.0, 0.05 / max(0.01, transition_width))

    return (ripple_theta * atten_theta * width_theta)**(1/3)


# =============================================================================
# Quantization
# =============================================================================

def quantization_snr(bits: int) -> float:
    r"""
    Compute SNR from quantization bits.

    SNR = 6.02 * n + 1.76 dB for n bits

    Args:
        bits: Number of quantization bits

    Returns:
        SNR in dB

    Reference: \cite{Oppenheim2009}
    """
    if bits <= 0:
        return 0.0
    return 6.02 * bits + 1.76


def compute_quantization_theta(
    bits: int,
    target_bits: int = 16
) -> float:
    r"""
    Compute theta from quantization level.

    Args:
        bits: Actual bit depth
        target_bits: Target bit depth

    Returns:
        theta in [0, 1]
    """
    if target_bits <= 0:
        return 0.0
    if bits >= target_bits:
        return 1.0

    theta = bits / target_bits
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified Signal Processing Theta
# =============================================================================

def compute_signal_theta(system: SignalSystem) -> float:
    r"""
    Compute unified theta for signal processing system.

    Args:
        system: SignalSystem dataclass

    Returns:
        theta in [0, 1]
    """
    thetas = []

    # SNR contribution
    theta_snr = compute_snr_theta(system.snr_db)
    thetas.append(theta_snr)

    # Compression contribution
    theta_comp = compute_compression_theta(system.compression_ratio)
    thetas.append(theta_comp)

    # Quantization contribution
    theta_quant = compute_quantization_theta(system.bit_depth)
    thetas.append(theta_quant)

    # Sparsity contribution (if sparse processing)
    if system.sparsity < 1:
        theta_sparse = compute_sparsity_theta(system.sparsity, 0.5)
        thetas.append(theta_sparse)

    if not thetas:
        return 0.5

    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_signal_quality(snr_db: float) -> SignalQuality:
    """
    Classify signal quality from SNR.

    Args:
        snr_db: SNR in dB

    Returns:
        SignalQuality enum
    """
    if snr_db < 0:
        return SignalQuality.CORRUPTED
    elif snr_db < 20:
        return SignalQuality.NOISY
    elif snr_db < 40:
        return SignalQuality.CLEAN
    else:
        return SignalQuality.PRISTINE


def classify_compression(
    ratio: float,
    is_lossless: bool
) -> CompressionLevel:
    """
    Classify compression level.

    Args:
        ratio: Compression ratio
        is_lossless: Whether lossless

    Returns:
        CompressionLevel enum
    """
    if is_lossless or ratio <= 1:
        return CompressionLevel.LOSSLESS
    elif ratio < 2:
        return CompressionLevel.NEAR_LOSSLESS
    elif ratio < 10:
        return CompressionLevel.LOSSY
    else:
        return CompressionLevel.EXTREME


# =============================================================================
# Example Systems Dictionary
# =============================================================================

SIGNAL_SYSTEMS: Dict[str, SignalSystem] = {
    "cd_audio": SignalSystem(
        name="CD Audio",
        sampling_rate=44100,
        snr_db=96.0,  # 16-bit
        compression_ratio=1.0,  # Lossless
        bandwidth=20000,
        bit_depth=16
    ),
    "mp3_320": SignalSystem(
        name="MP3 320kbps",
        sampling_rate=44100,
        snr_db=60.0,
        compression_ratio=4.5,
        bandwidth=20000,
        bit_depth=16
    ),
    "mp3_128": SignalSystem(
        name="MP3 128kbps",
        sampling_rate=44100,
        snr_db=45.0,
        compression_ratio=11.0,
        bandwidth=16000,
        bit_depth=16
    ),
    "jpeg_quality_90": SignalSystem(
        name="JPEG Q90",
        sampling_rate=1.0,  # N/A for images
        snr_db=35.0,
        compression_ratio=10.0,
        sparsity=0.2
    ),
    "radar_processing": SignalSystem(
        name="Radar Signal Processing",
        sampling_rate=100e6,
        snr_db=20.0,
        compression_ratio=1.0,
        bandwidth=10e6,
        dynamic_range_db=60.0
    ),
    "speech_telephony": SignalSystem(
        name="Telephone Speech",
        sampling_rate=8000,
        snr_db=30.0,
        compression_ratio=8.0,
        bandwidth=3400,
        bit_depth=8
    ),
    "mri_reconstruction": SignalSystem(
        name="MRI Compressive Sensing",
        sampling_rate=1.0,
        snr_db=30.0,
        compression_ratio=4.0,
        sparsity=0.1
    ),
    "hdtv_broadcast": SignalSystem(
        name="HDTV Broadcast",
        sampling_rate=27e6,
        snr_db=50.0,
        compression_ratio=50.0,
        bandwidth=6e6,
        bit_depth=10
    ),
    "studio_master": SignalSystem(
        name="Studio Master Audio",
        sampling_rate=192000,
        snr_db=144.0,  # 24-bit
        compression_ratio=1.0,
        bandwidth=48000,
        bit_depth=24
    ),
    "ultrasound_imaging": SignalSystem(
        name="Medical Ultrasound",
        sampling_rate=50e6,
        snr_db=25.0,
        compression_ratio=3.0,
        bandwidth=10e6,
        dynamic_range_db=80.0
    ),
}


# Precomputed theta values
SIGNAL_THETA_VALUES: Dict[str, float] = {
    name: compute_signal_theta(system)
    for name, system in SIGNAL_SYSTEMS.items()
}
