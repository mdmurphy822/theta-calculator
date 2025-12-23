"""
Tests for quantum_biology module.

Tests quantum biology theta calculations for photosynthesis,
magnetoreception, enzyme catalysis, and other biological systems.
"""

import pytest

from theta_calculator.domains.quantum_biology import (
    BIOLOGICAL_SYSTEMS,
    BiologicalSystem,
    QuantumBioRegime,
    QuantumMechanism,
    compute_quantum_bio_theta,
    compute_coherence_theta,
    compute_tunneling_theta,
    classify_regime,
    thermal_time,
    M_PROTON,
    M_ELECTRON,
)


class TestBiologicalSystems:
    """Test the predefined biological systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "fmo_complex", "fmo_room_temp", "lhcii_complex",
            "cryptochrome_bird", "cryptochrome_drosophila",
            "alcohol_dehydrogenase", "soybean_lipoxygenase",
            "dna_tautomerization", "olfactory_receptor", "atp_synthase"
        ]
        for name in expected:
            assert name in BIOLOGICAL_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in BIOLOGICAL_SYSTEMS.items():
            assert isinstance(system, BiologicalSystem)
            assert system.name
            assert system.organism
            assert isinstance(system.mechanism, QuantumMechanism)
            assert system.coherence_time > 0
            assert system.thermal_time > 0
            assert system.functional_time > 0
            assert system.temperature > 0


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in BIOLOGICAL_SYSTEMS.items():
            theta = compute_quantum_bio_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_fmo_complex_high_coherence(self):
        """FMO complex at 77K should have significant quantum effects."""
        fmo = BIOLOGICAL_SYSTEMS["fmo_complex"]
        theta = compute_quantum_bio_theta(fmo)
        # Photosynthesis should be in functional/coherent regime
        assert theta > 0.1, f"FMO theta too low: {theta}"

    def test_magnetoreception_coherent(self):
        """Bird magnetoreception requires long coherence."""
        crypto = BIOLOGICAL_SYSTEMS["cryptochrome_bird"]
        theta_coh = compute_coherence_theta(crypto)
        # Cryptochrome has microsecond coherence at body temp
        assert theta_coh > 0.5, f"Cryptochrome coherence theta too low: {theta_coh}"

    def test_enzyme_tunneling_present(self):
        """Enzyme catalysis should show tunneling effects."""
        adh = BIOLOGICAL_SYSTEMS["alcohol_dehydrogenase"]
        theta = compute_quantum_bio_theta(adh)
        # Tunneling contributes but doesn't dominate
        assert 0 < theta < 1, f"Enzyme theta unexpected: {theta}"

    def test_room_temp_lower_than_cryogenic(self):
        """Room temperature should reduce quantum effects."""
        fmo_cold = BIOLOGICAL_SYSTEMS["fmo_complex"]
        fmo_warm = BIOLOGICAL_SYSTEMS["fmo_room_temp"]

        theta_cold = compute_quantum_bio_theta(fmo_cold)
        theta_warm = compute_quantum_bio_theta(fmo_warm)

        # Warmer should have lower theta (more decoherence)
        assert theta_warm <= theta_cold, "Room temp should have lower theta"


class TestCoherenceTheta:
    """Test coherence-based theta calculation."""

    def test_coherence_theta_formula(self):
        """Test theta = tau_c / (tau_c + tau_th) where tau_th = hbar/(k_B*T)."""
        # At 300K, thermal_time ≈ 2.5e-14 s
        # Set coherence_time equal to thermal_time for theta = 0.5
        tau_th = thermal_time(300)  # ~2.5e-14 s
        system = BiologicalSystem(
            name="Test",
            organism="Test",
            mechanism=QuantumMechanism.COHERENT_TRANSFER,
            coherence_time=tau_th,  # Equal to thermal time
            thermal_time=tau_th,
            functional_time=1e-12,
            temperature=300
        )
        theta = compute_coherence_theta(system)
        # When tau_c = tau_th, theta should be 0.5
        assert abs(theta - 0.5) < 0.01

    def test_high_coherence_gives_high_theta(self):
        """Long coherence time should give theta near 1."""
        system = BiologicalSystem(
            name="Test",
            organism="Test",
            mechanism=QuantumMechanism.COHERENT_TRANSFER,
            coherence_time=1e-9,   # 1 ns (long)
            thermal_time=1e-12,    # 1 ps (short)
            functional_time=1e-12,
            temperature=300
        )
        theta = compute_coherence_theta(system)
        assert theta > 0.99, f"High coherence should give theta near 1: {theta}"

    def test_low_coherence_gives_low_theta(self):
        """Short coherence time should give theta near 0."""
        system = BiologicalSystem(
            name="Test",
            organism="Test",
            mechanism=QuantumMechanism.COHERENT_TRANSFER,
            coherence_time=1e-17,  # Much shorter than thermal time
            thermal_time=1e-12,
            functional_time=1e-12,
            temperature=300
        )
        theta = compute_coherence_theta(system)
        # tau_th at 300K ≈ 2.5e-14 s, so 1e-17 << tau_th gives low theta
        assert theta < 0.01, f"Low coherence should give theta near 0: {theta}"


class TestTunnelingTheta:
    """Test quantum tunneling theta calculation."""

    def test_tunneling_electron_high(self):
        """Electrons tunnel easily through thin barriers."""
        theta = compute_tunneling_theta(
            barrier_height=0.1,  # 0.1 eV (low)
            barrier_width=0.1,  # 0.1 nm (thin)
            particle_mass=M_ELECTRON,
            temperature=300
        )
        # Electron through thin/low barrier should tunnel well
        assert theta > 0.5, f"Electron tunneling theta too low: {theta}"

    def test_tunneling_proton_moderate(self):
        """Protons tunnel less easily than electrons."""
        theta_e = compute_tunneling_theta(0.3, 0.05, M_ELECTRON, 300)
        theta_p = compute_tunneling_theta(0.3, 0.05, M_PROTON, 300)

        # Heavier particle should tunnel less
        assert theta_p < theta_e, "Proton should tunnel less than electron"

    def test_thick_barrier_reduces_tunneling(self):
        """Thicker barrier should reduce tunneling."""
        theta_thin = compute_tunneling_theta(0.3, 0.05, M_PROTON, 300)
        theta_thick = compute_tunneling_theta(0.3, 0.2, M_PROTON, 300)

        assert theta_thick < theta_thin, "Thicker barrier should reduce tunneling"


class TestRegimeClassification:
    """Test regime classification function."""

    def test_coherent_regime(self):
        """Theta > 0.6 should be COHERENT."""
        assert classify_regime(0.7) == QuantumBioRegime.COHERENT
        assert classify_regime(0.9) == QuantumBioRegime.COHERENT

    def test_functional_regime(self):
        """0.3 < theta < 0.6 should be FUNCTIONAL."""
        assert classify_regime(0.4) == QuantumBioRegime.FUNCTIONAL
        assert classify_regime(0.5) == QuantumBioRegime.FUNCTIONAL

    def test_transition_regime(self):
        """0.1 < theta < 0.3 should be TRANSITION."""
        assert classify_regime(0.15) == QuantumBioRegime.TRANSITION
        assert classify_regime(0.25) == QuantumBioRegime.TRANSITION

    def test_classical_regime(self):
        """theta < 0.1 should be CLASSICAL."""
        assert classify_regime(0.05) == QuantumBioRegime.CLASSICAL
        assert classify_regime(0.0) == QuantumBioRegime.CLASSICAL


class TestThermalTime:
    """Test thermal time calculation."""

    def test_thermal_time_room_temp(self):
        """Thermal time at 300K should be ~25 fs."""
        tau = thermal_time(300)
        # hbar / (k_B * 300) ≈ 2.5e-14 s
        assert 2e-14 < tau < 3e-14, f"Thermal time unexpected: {tau}"

    def test_thermal_time_cryogenic(self):
        """Thermal time at 77K should be ~100 fs."""
        tau = thermal_time(77)
        # hbar / (k_B * 77) ≈ 1e-13 s
        assert 0.5e-13 < tau < 1.5e-13, f"Thermal time unexpected: {tau}"

    def test_thermal_time_inversely_proportional(self):
        """Thermal time should scale as 1/T."""
        tau_300 = thermal_time(300)
        tau_150 = thermal_time(150)
        ratio = tau_150 / tau_300
        # Should be ~2
        assert 1.9 < ratio < 2.1, f"Scaling ratio unexpected: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
