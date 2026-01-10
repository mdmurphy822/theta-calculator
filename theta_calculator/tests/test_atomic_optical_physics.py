"""
Tests for Atomic and Optical Physics Domain Module

Tests cover:
- Rabi oscillations and coherence
- Cavity QED cooperativity
- Optical depth and light-matter coupling
- Squeezed light states
- Laser cooling regimes
- Theta range validation [0, 1]
"""

import pytest

from theta_calculator.domains.atomic_optical_physics import (
    AtomicOpticalSystem,
    AtomicRegime,
    OpticalPhase,
    TrapType,
    CoolingRegime,
    compute_atomic_optical_theta,
    compute_rabi_theta,
    compute_cooperativity_theta,
    compute_optical_depth_theta,
    compute_squeezing_theta,
    compute_cooling_theta,
    compute_clock_theta,
    rabi_frequency,
    cavity_cooperativity,
    optical_depth,
    resonant_cross_section,
    squeezing_to_linear,
    shot_noise_variance,
    doppler_temperature,
    recoil_temperature,
    clock_stability,
    classify_atomic_regime,
    classify_optical_phase,
    classify_cooling_regime,
    ATOMIC_OPTICAL_SYSTEMS,
    HBAR,
    K_B,
    GAMMA_RB87_D2,
    LAMBDA_RB87_D2,
    T_DOPPLER_RB87,
    T_RECOIL_RB87,
)


class TestAtomicOpticalSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """ATOMIC_OPTICAL_SYSTEMS dict should exist."""
        assert ATOMIC_OPTICAL_SYSTEMS is not None
        assert isinstance(ATOMIC_OPTICAL_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(ATOMIC_OPTICAL_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["rb87_mot", "rb87_bec", "trapped_ion_clock", "cavity_qed_cs"]
        for name in expected:
            assert name in ATOMIC_OPTICAL_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in ATOMIC_OPTICAL_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "atom_species")
            assert hasattr(system, "wavelength")


class TestRabiFrequency:
    """Test Rabi frequency calculation."""

    def test_typical_values(self):
        """Typical dipole and field give reasonable Rabi frequency."""
        dipole = 1e-29  # C*m, typical atomic dipole
        E_field = 1e4   # V/m
        Omega = rabi_frequency(dipole, E_field)
        assert Omega > 0
        assert Omega < 1e12  # Less than THz

    def test_zero_field(self):
        """Zero field gives zero Rabi frequency."""
        Omega = rabi_frequency(1e-29, 0)
        assert Omega == 0.0

    def test_proportional_to_field(self):
        """Rabi frequency scales linearly with field."""
        dipole = 1e-29
        Omega_1 = rabi_frequency(dipole, 1e3)
        Omega_2 = rabi_frequency(dipole, 2e3)
        assert Omega_2 == pytest.approx(2 * Omega_1)


class TestRabiTheta:
    """Test Rabi theta calculation."""

    def test_strong_driving(self):
        """Strong driving gives high theta."""
        theta = compute_rabi_theta(1e8, 1e-4, GAMMA_RB87_D2)
        assert theta > 0.5

    def test_weak_driving(self):
        """Weak driving gives low theta."""
        theta = compute_rabi_theta(1e4, 1e-6, GAMMA_RB87_D2)
        assert theta < 0.5

    def test_zero_rabi(self):
        """Zero Rabi frequency gives theta = 0."""
        theta = compute_rabi_theta(0, 1e-4)
        assert theta == 0.0

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for Omega in [1e3, 1e6, 1e9]:
            theta = compute_rabi_theta(Omega, 1e-4)
            assert 0 <= theta <= 1


class TestCavityCooperativity:
    """Test cavity QED cooperativity."""

    def test_strong_coupling(self):
        """Large g gives C > 1."""
        C = cavity_cooperativity(1e7, 1e6, 1e6)  # g > kappa, gamma
        assert C > 1

    def test_weak_coupling(self):
        """Small g gives C < 1."""
        C = cavity_cooperativity(1e4, 1e6, 1e6)  # g << kappa, gamma
        assert C < 1

    def test_zero_g(self):
        """Zero coupling gives C = 0."""
        C = cavity_cooperativity(0, 1e6, 1e6)
        assert C == 0.0

    def test_zero_kappa(self):
        """Zero cavity decay gives C = 0 (handled)."""
        C = cavity_cooperativity(1e6, 0, 1e6)
        assert C == 0.0


class TestCooperativityTheta:
    """Test cooperativity theta calculation."""

    def test_strong_coupling(self):
        """C >> 1 gives theta near 1."""
        theta = compute_cooperativity_theta(100)
        assert theta > 0.99

    def test_weak_coupling(self):
        """C << 1 gives theta near 0."""
        theta = compute_cooperativity_theta(0.01)
        assert theta < 0.01

    def test_critical_coupling(self):
        """C = 1 gives theta = 0.5."""
        theta = compute_cooperativity_theta(1.0)
        assert theta == pytest.approx(0.5)

    def test_zero_cooperativity(self):
        """C = 0 gives theta = 0."""
        theta = compute_cooperativity_theta(0)
        assert theta == 0.0


class TestOpticalDepth:
    """Test optical depth calculation."""

    def test_typical_values(self):
        """Typical values give positive OD."""
        OD = optical_depth(1e18, 3e-13, 1e-3)  # n, sigma, L
        assert OD > 0

    def test_zero_density(self):
        """Zero density gives OD = 0."""
        OD = optical_depth(0, 3e-13, 1e-3)
        assert OD == 0.0


class TestResonantCrossSection:
    """Test resonant cross section."""

    def test_rb87_d2(self):
        """Rb-87 D2 cross section is correct order of magnitude."""
        sigma = resonant_cross_section(LAMBDA_RB87_D2)
        # Should be ~3e-13 m^2
        assert 1e-14 < sigma < 1e-12


class TestOpticalDepthTheta:
    """Test optical depth theta calculation."""

    def test_high_od(self):
        """High OD gives theta near 1."""
        theta = compute_optical_depth_theta(500)
        assert theta > 0.99

    def test_low_od(self):
        """Low OD gives low theta."""
        theta = compute_optical_depth_theta(1)
        assert theta < 0.1

    def test_zero_od(self):
        """Zero OD gives theta = 0."""
        theta = compute_optical_depth_theta(0)
        assert theta == 0.0

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for OD in [0, 1, 10, 100, 1000]:
            theta = compute_optical_depth_theta(OD)
            assert 0 <= theta <= 1


class TestSqueezingToLinear:
    """Test dB to linear conversion for squeezing."""

    def test_no_squeezing(self):
        """0 dB = 1 linear."""
        ratio = squeezing_to_linear(0)
        assert ratio == pytest.approx(1.0)

    def test_3db_squeezing(self):
        """3 dB = 2 linear, -3 dB = 0.5 linear."""
        ratio_pos = squeezing_to_linear(3)
        ratio_neg = squeezing_to_linear(-3)
        assert ratio_pos == pytest.approx(2.0, rel=0.01)
        assert ratio_neg == pytest.approx(0.5, rel=0.01)

    def test_10db_squeezing(self):
        """10 dB = 10 linear."""
        ratio = squeezing_to_linear(10)
        assert ratio == pytest.approx(10.0)


class TestShotNoiseVariance:
    """Test shot noise calculation."""

    def test_typical_photon_number(self):
        """sqrt(N) for photon number."""
        variance = shot_noise_variance(100)
        assert variance == pytest.approx(10.0)

    def test_zero_photons(self):
        """Zero photons gives zero variance."""
        variance = shot_noise_variance(0)
        assert variance == 0.0

    def test_negative_photons(self):
        """Negative photons handled gracefully."""
        variance = shot_noise_variance(-10)
        assert variance == 0.0


class TestSqueezingTheta:
    """Test squeezing theta calculation."""

    def test_target_squeezing(self):
        """Achieving target gives theta = 1."""
        theta = compute_squeezing_theta(-15, -15)
        assert theta == pytest.approx(1.0)

    def test_no_squeezing(self):
        """No squeezing gives theta = 0."""
        theta = compute_squeezing_theta(0, -15)
        assert theta == 0.0

    def test_partial_squeezing(self):
        """Partial squeezing gives intermediate theta."""
        theta = compute_squeezing_theta(-7.5, -15)
        assert theta == pytest.approx(0.5)


class TestDopplerTemperature:
    """Test Doppler cooling limit."""

    def test_rb87(self):
        """Rb-87 Doppler limit is ~145 uK."""
        m_rb = 87 * 1.66e-27  # kg
        T_D = doppler_temperature(GAMMA_RB87_D2, m_rb)
        assert 100e-6 < T_D < 200e-6


class TestRecoilTemperature:
    """Test recoil temperature."""

    def test_rb87(self):
        """Rb-87 recoil temperature is ~360 nK."""
        m_rb = 87 * 1.66e-27
        T_r = recoil_temperature(LAMBDA_RB87_D2, m_rb)
        assert 100e-9 < T_r < 1e-6


class TestCoolingTheta:
    """Test cooling theta calculation."""

    def test_thermal(self):
        """Far above Doppler gives low theta."""
        theta = compute_cooling_theta(1e-3, 100e-6, 100e-9)
        assert theta < 0.2

    def test_doppler_limited(self):
        """At Doppler limit gives theta ~ 0.5."""
        theta = compute_cooling_theta(100e-6, 100e-6, 100e-9)
        assert 0.4 < theta < 0.6

    def test_recoil_limited(self):
        """Near recoil limit gives high theta."""
        theta = compute_cooling_theta(500e-9, 100e-6, 100e-9)
        assert theta > 0.7

    def test_bec(self):
        """Below recoil gives theta = 1."""
        theta = compute_cooling_theta(50e-9, 100e-6, 100e-9)
        assert theta == 1.0


class TestClockStability:
    """Test atomic clock stability."""

    def test_typical_clock(self):
        """Typical values give reasonable stability."""
        stability = clock_stability(1.0, 0.1, 1000, 1e15)
        assert stability > 0
        assert stability < 1e-15

    def test_more_atoms_better(self):
        """More atoms improves stability."""
        stab_few = clock_stability(1.0, 0.1, 100, 1e15)
        stab_many = clock_stability(1.0, 0.1, 10000, 1e15)
        assert stab_many < stab_few


class TestClockTheta:
    """Test clock theta calculation."""

    def test_achieved_target(self):
        """Achieving target gives theta = 1."""
        theta = compute_clock_theta(1e-18, 1e-18)
        assert theta == 1.0

    def test_worse_than_target(self):
        """Worse stability gives lower theta."""
        theta = compute_clock_theta(1e-16, 1e-18)
        assert theta < 1.0
        assert theta > 0

    def test_zero_stability(self):
        """Zero stability gives theta = 0."""
        theta = compute_clock_theta(0, 1e-18)
        assert theta == 0.0


class TestUnifiedAtomicOpticalTheta:
    """Test unified atomic/optical theta calculation."""

    def test_all_systems_valid_theta(self):
        """All systems should have theta in [0, 1]."""
        for name, system in ATOMIC_OPTICAL_SYSTEMS.items():
            theta = compute_atomic_optical_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_bec_high_theta(self):
        """BEC should have high theta (quantum degenerate)."""
        bec = ATOMIC_OPTICAL_SYSTEMS["rb87_bec"]
        theta = compute_atomic_optical_theta(bec)
        assert theta > 0.5

    def test_cavity_qed_strong(self):
        """Cavity QED with high C has high theta."""
        cavity = ATOMIC_OPTICAL_SYSTEMS["cavity_qed_cs"]
        theta = compute_atomic_optical_theta(cavity)
        assert theta > 0.5


class TestClassifyAtomicRegime:
    """Test atomic regime classification."""

    def test_incoherent(self):
        """Very low theta -> INCOHERENT."""
        assert classify_atomic_regime(0.05) == AtomicRegime.INCOHERENT

    def test_weak_coupling(self):
        """Low theta -> WEAK_COUPLING."""
        assert classify_atomic_regime(0.2) == AtomicRegime.WEAK_COUPLING

    def test_intermediate(self):
        """Medium theta -> INTERMEDIATE."""
        assert classify_atomic_regime(0.45) == AtomicRegime.INTERMEDIATE

    def test_strong_coupling(self):
        """High theta -> STRONG_COUPLING."""
        assert classify_atomic_regime(0.75) == AtomicRegime.STRONG_COUPLING

    def test_ultrastrong(self):
        """Very high theta -> ULTRASTRONG."""
        assert classify_atomic_regime(0.95) == AtomicRegime.ULTRASTRONG


class TestClassifyOpticalPhase:
    """Test optical phase classification."""

    def test_thermal(self):
        """Positive dB -> THERMAL."""
        assert classify_optical_phase(5) == OpticalPhase.THERMAL

    def test_coherent(self):
        """Near zero dB -> COHERENT."""
        assert classify_optical_phase(0) == OpticalPhase.COHERENT

    def test_squeezed(self):
        """Negative dB -> SQUEEZED."""
        assert classify_optical_phase(-10) == OpticalPhase.SQUEEZED

    def test_entangled(self):
        """With entanglement flag -> ENTANGLED."""
        assert classify_optical_phase(-10, is_entangled=True) == OpticalPhase.ENTANGLED


class TestClassifyCoolingRegime:
    """Test cooling regime classification."""

    def test_thermal(self):
        """Far above Doppler -> THERMAL."""
        # T > 10 * T_doppler for THERMAL
        regime = classify_cooling_regime(2e-3, 100e-6, 100e-9)
        assert regime == CoolingRegime.THERMAL

    def test_doppler(self):
        """Near Doppler -> DOPPLER."""
        # 2 * T_doppler < T < 10 * T_doppler for DOPPLER
        regime = classify_cooling_regime(500e-6, 100e-6, 100e-9)
        assert regime == CoolingRegime.DOPPLER

    def test_sub_doppler(self):
        """Between Doppler and recoil -> SUB_DOPPLER."""
        regime = classify_cooling_regime(10e-6, 100e-6, 100e-9)
        assert regime == CoolingRegime.SUB_DOPPLER

    def test_recoil(self):
        """Near recoil -> RECOIL."""
        regime = classify_cooling_regime(500e-9, 100e-6, 100e-9)
        assert regime == CoolingRegime.RECOIL

    def test_degenerate(self):
        """Below recoil -> DEGENERATE."""
        regime = classify_cooling_regime(50e-9, 100e-6, 100e-9)
        assert regime == CoolingRegime.DEGENERATE


class TestEnums:
    """Test enum definitions."""

    def test_atomic_regimes(self):
        """All atomic regimes defined."""
        assert AtomicRegime.INCOHERENT.value == "incoherent"
        assert AtomicRegime.WEAK_COUPLING.value == "weak_coupling"
        assert AtomicRegime.INTERMEDIATE.value == "intermediate"
        assert AtomicRegime.STRONG_COUPLING.value == "strong_coupling"
        assert AtomicRegime.ULTRASTRONG.value == "ultrastrong"

    def test_optical_phases(self):
        """All optical phases defined."""
        assert OpticalPhase.THERMAL.value == "thermal"
        assert OpticalPhase.COHERENT.value == "coherent"
        assert OpticalPhase.SQUEEZED.value == "squeezed"
        assert OpticalPhase.ENTANGLED.value == "entangled"

    def test_trap_types(self):
        """All trap types defined."""
        assert TrapType.MAGNETO_OPTICAL.value == "mot"
        assert TrapType.OPTICAL_LATTICE.value == "lattice"
        assert TrapType.ION_TRAP.value == "ion"
        assert TrapType.TWEEZER.value == "tweezer"

    def test_cooling_regimes(self):
        """All cooling regimes defined."""
        assert CoolingRegime.THERMAL.value == "thermal"
        assert CoolingRegime.DOPPLER.value == "doppler"
        assert CoolingRegime.SUB_DOPPLER.value == "sub_doppler"
        assert CoolingRegime.RECOIL.value == "recoil"
        assert CoolingRegime.DEGENERATE.value == "degenerate"


class TestAtomicOpticalSystemDataclass:
    """Test AtomicOpticalSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with required parameters."""
        system = AtomicOpticalSystem(name="Test")
        assert system.name == "Test"
        assert system.atom_species == "Rb87"  # Default

    def test_default_values(self):
        """Optional fields have defaults."""
        system = AtomicOpticalSystem(name="Test")
        assert system.temperature == 1e-6
        assert system.squeezing_db == 0.0
        assert system.trap_type == TrapType.MAGNETO_OPTICAL

    def test_custom_values(self):
        """Can set custom values."""
        system = AtomicOpticalSystem(
            name="Custom",
            atom_species="Cs133",
            cooperativity=50.0,
            squeezing_db=-10.0
        )
        assert system.atom_species == "Cs133"
        assert system.cooperativity == 50.0
        assert system.squeezing_db == -10.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_temperature(self):
        """Zero temperature handled gracefully."""
        theta = compute_cooling_theta(0, 100e-6, 100e-9)
        assert theta == 0.0

    def test_negative_squeezing_extreme(self):
        """Extreme squeezing clipped to 1."""
        theta = compute_squeezing_theta(-30, -15)
        assert theta == 1.0

    def test_very_high_cooperativity(self):
        """Very high cooperativity gives theta near 1."""
        theta = compute_cooperativity_theta(1e6)
        assert theta > 0.999

    def test_negative_coherence_time(self):
        """Negative coherence time handled."""
        theta = compute_rabi_theta(1e6, -1)
        assert 0 <= theta <= 1


class TestPhysicalConstants:
    """Test that physical constants are reasonable."""

    def test_hbar(self):
        """hbar should be ~1e-34 J*s."""
        assert 1e-35 < HBAR < 1e-33

    def test_kb(self):
        """k_B should be ~1.38e-23 J/K."""
        assert 1e-24 < K_B < 1e-22

    def test_gamma_rb87(self):
        """Rb-87 D2 linewidth should be ~38 Mrad/s."""
        assert 3e7 < GAMMA_RB87_D2 < 4e7

    def test_lambda_rb87(self):
        """Rb-87 D2 wavelength should be ~780 nm."""
        assert 779e-9 < LAMBDA_RB87_D2 < 781e-9

    def test_t_doppler_rb87(self):
        """Rb-87 Doppler temp should be ~145 uK."""
        assert 100e-6 < T_DOPPLER_RB87 < 200e-6

    def test_t_recoil_rb87(self):
        """Rb-87 recoil temp should be ~360 nK."""
        assert 100e-9 < T_RECOIL_RB87 < 1e-6
