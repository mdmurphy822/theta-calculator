"""
Tests for Signal Processing Domain Module

Tests cover:
- SNR calculations
- Compression analysis
- Compressive sensing
- Theta range validation [0, 1]
"""

import pytest

from theta_calculator.domains.signal_processing import (
    SignalSystem,
    SignalQuality,
    ProcessingDomain,
    CompressionLevel,
    FilterType,
    compute_signal_theta,
    compute_snr_theta,
    compute_compression_theta,
    compute_distortion_theta,
    compute_sparsity_theta,
    compute_filter_theta,
    compute_quantization_theta,
    snr_db_to_linear,
    snr_linear_to_db,
    rate_distortion,
    quantization_snr,
    classify_signal_quality,
    classify_compression,
    SIGNAL_SYSTEMS,
)


class TestSignalSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """SIGNAL_SYSTEMS dict should exist."""
        assert SIGNAL_SYSTEMS is not None
        assert isinstance(SIGNAL_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(SIGNAL_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["cd_audio", "mp3_320", "radar_processing", "studio_master"]
        for name in expected:
            assert name in SIGNAL_SYSTEMS, f"Missing system: {name}"


class TestSNRConversions:
    """Test SNR dB/linear conversions."""

    def test_db_to_linear_0(self):
        """0 dB = 1 linear."""
        assert snr_db_to_linear(0) == pytest.approx(1.0)

    def test_db_to_linear_10(self):
        """10 dB = 10 linear."""
        assert snr_db_to_linear(10) == pytest.approx(10.0)

    def test_db_to_linear_20(self):
        """20 dB = 100 linear."""
        assert snr_db_to_linear(20) == pytest.approx(100.0)

    def test_linear_to_db_1(self):
        """1 linear = 0 dB."""
        assert snr_linear_to_db(1) == pytest.approx(0.0)

    def test_linear_to_db_100(self):
        """100 linear = 20 dB."""
        assert snr_linear_to_db(100) == pytest.approx(20.0)

    def test_linear_to_db_zero(self):
        """0 linear = -inf dB."""
        assert snr_linear_to_db(0) == float('-inf')


class TestSNRTheta:
    """Test SNR theta calculation."""

    def test_zero_snr(self):
        """Zero SNR gives theta = 0."""
        theta = compute_snr_theta(0)
        assert theta == 0.0

    def test_target_snr(self):
        """Target SNR gives theta = 1."""
        theta = compute_snr_theta(40, 40)
        assert theta == 1.0

    def test_half_target(self):
        """Half target gives theta = 0.5."""
        theta = compute_snr_theta(20, 40)
        assert theta == 0.5


class TestCompressionTheta:
    """Test compression theta calculation."""

    def test_no_compression(self):
        """Ratio = 1 gives theta = 1."""
        theta = compute_compression_theta(1.0)
        assert theta == 1.0

    def test_lossless(self):
        """Lossless flag gives theta = 1."""
        theta = compute_compression_theta(5.0, is_lossless=True)
        assert theta == 1.0

    def test_max_compression(self):
        """Max ratio gives theta = 0."""
        theta = compute_compression_theta(10.0, 10.0)
        assert theta == 0.0


class TestRateDistortion:
    """Test rate-distortion function."""

    def test_zero_rate(self):
        """Zero rate gives full distortion."""
        D = rate_distortion(0, 1.0)
        assert D == 1.0

    def test_high_rate(self):
        """High rate gives low distortion."""
        D = rate_distortion(10, 1.0)
        assert D < 0.001


class TestDistortionTheta:
    """Test distortion theta calculation."""

    def test_zero_distortion(self):
        """Zero distortion gives theta = 1."""
        theta = compute_distortion_theta(0)
        assert theta == 1.0

    def test_target_distortion(self):
        """Target distortion gives theta = 1."""
        theta = compute_distortion_theta(0.01, 0.01)
        assert theta == 1.0

    def test_high_distortion(self):
        """High distortion gives low theta."""
        theta = compute_distortion_theta(1.0, 0.01)
        assert theta == 0.01


class TestSparsityTheta:
    """Test compressive sensing theta calculation."""

    def test_zero_sparsity(self):
        """Zero sparsity (zero signal) gives theta = 1."""
        theta = compute_sparsity_theta(0, 0.5)
        assert theta == 1.0

    def test_not_sparse(self):
        """Full sparsity gives theta = 0."""
        theta = compute_sparsity_theta(1.0, 0.5)
        assert theta == 0.0

    def test_sparse_recoverable(self):
        """Sparse with enough measurements gives high theta."""
        theta = compute_sparsity_theta(0.01, 0.5)
        assert theta > 0.9


class TestFilterTheta:
    """Test filter theta calculation."""

    def test_ideal_filter(self):
        """Near-ideal filter gives high theta."""
        theta = compute_filter_theta(0.1, 60, 0.05)
        assert theta > 0.9

    def test_poor_filter(self):
        """Poor filter gives low theta."""
        theta = compute_filter_theta(1.0, 20, 0.2)
        assert theta < 0.5


class TestQuantization:
    """Test quantization calculations."""

    def test_16bit_snr(self):
        """16-bit gives ~98 dB SNR."""
        snr = quantization_snr(16)
        assert 95 < snr < 100

    def test_8bit_snr(self):
        """8-bit gives ~50 dB SNR."""
        snr = quantization_snr(8)
        assert 48 < snr < 52


class TestQuantizationTheta:
    """Test quantization theta calculation."""

    def test_target_bits(self):
        """Target bits gives theta = 1."""
        theta = compute_quantization_theta(16, 16)
        assert theta == 1.0

    def test_low_bits(self):
        """Low bits gives low theta."""
        theta = compute_quantization_theta(8, 16)
        assert theta == 0.5


class TestUnifiedSignalTheta:
    """Test unified signal processing theta calculation."""

    def test_all_systems_valid_theta(self):
        """All systems should have theta in [0, 1]."""
        for name, system in SIGNAL_SYSTEMS.items():
            theta = compute_signal_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_cd_audio_high(self):
        """CD audio has high quality."""
        cd = SIGNAL_SYSTEMS["cd_audio"]
        theta = compute_signal_theta(cd)
        assert theta > 0.8

    def test_studio_master_highest(self):
        """Studio master has highest quality."""
        studio = SIGNAL_SYSTEMS["studio_master"]
        cd = SIGNAL_SYSTEMS["cd_audio"]
        theta_studio = compute_signal_theta(studio)
        theta_cd = compute_signal_theta(cd)
        assert theta_studio >= theta_cd


class TestClassifySignalQuality:
    """Test signal quality classification."""

    def test_corrupted(self):
        """Negative SNR -> CORRUPTED."""
        result = classify_signal_quality(-5)
        assert result == SignalQuality.CORRUPTED

    def test_noisy(self):
        """Low SNR -> NOISY."""
        result = classify_signal_quality(10)
        assert result == SignalQuality.NOISY

    def test_clean(self):
        """Medium SNR -> CLEAN."""
        result = classify_signal_quality(30)
        assert result == SignalQuality.CLEAN

    def test_pristine(self):
        """High SNR -> PRISTINE."""
        result = classify_signal_quality(50)
        assert result == SignalQuality.PRISTINE


class TestClassifyCompression:
    """Test compression classification."""

    def test_lossless(self):
        """Lossless flag -> LOSSLESS."""
        result = classify_compression(5, True)
        assert result == CompressionLevel.LOSSLESS

    def test_near_lossless(self):
        """Low ratio -> NEAR_LOSSLESS."""
        result = classify_compression(1.5, False)
        assert result == CompressionLevel.NEAR_LOSSLESS

    def test_lossy(self):
        """Medium ratio -> LOSSY."""
        result = classify_compression(5, False)
        assert result == CompressionLevel.LOSSY

    def test_extreme(self):
        """High ratio -> EXTREME."""
        result = classify_compression(20, False)
        assert result == CompressionLevel.EXTREME


class TestEnums:
    """Test enum definitions."""

    def test_signal_quality(self):
        """All signal quality levels defined."""
        assert SignalQuality.CORRUPTED.value == "corrupted"
        assert SignalQuality.PRISTINE.value == "pristine"

    def test_processing_domain(self):
        """All processing domains defined."""
        assert ProcessingDomain.TIME.value == "time"
        assert ProcessingDomain.SPARSE.value == "sparse"

    def test_filter_type(self):
        """All filter types defined."""
        assert FilterType.LOWPASS.value == "lowpass"
        assert FilterType.BANDPASS.value == "bandpass"


class TestSignalSystemDataclass:
    """Test SignalSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with name only."""
        system = SignalSystem(name="Test")
        assert system.name == "Test"
        assert system.sampling_rate == 44100.0
        assert system.snr_db == 40.0

    def test_custom_values(self):
        """Can set custom values."""
        system = SignalSystem(
            name="Custom",
            sampling_rate=96000,
            snr_db=60.0,
            compression_ratio=2.0
        )
        assert system.sampling_rate == 96000
        assert system.compression_ratio == 2.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_snr(self):
        """Negative SNR handled gracefully."""
        theta = compute_snr_theta(-10)
        assert theta == 0.0

    def test_zero_target(self):
        """Zero target gives theta = 0."""
        theta = compute_snr_theta(10, 0)
        assert theta == 0.0

    def test_zero_bits(self):
        """Zero bits gives zero SNR."""
        snr = quantization_snr(0)
        assert snr == 0.0
