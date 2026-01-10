"""
Tests for High Energy Physics Domain Module

Tests cover:
- QCD running coupling and asymptotic freedom
- Lattice QCD spacing and continuum limit
- Confinement and string tension
- Chiral symmetry breaking
- Collider kinematics
- Theta range validation [0, 1]
"""

import pytest

from theta_calculator.domains.high_energy_physics import (
    HEPSystem,
    ParticleRegime,
    LatticeRegime,
    ChiralRegime,
    ExperimentType,
    compute_hep_theta,
    compute_qcd_coupling_theta,
    compute_lattice_theta,
    compute_confinement_theta,
    compute_chiral_theta,
    compute_precision_theta,
    running_alpha_s,
    beta_0,
    lattice_beta,
    string_tension,
    chiral_condensate,
    cms_energy,
    classify_particle_regime,
    classify_lattice_regime,
    classify_chiral_regime,
    HEP_SYSTEMS,
    LAMBDA_QCD,
    ALPHA_S_MZ,
    M_Z,
    T_QGP,
    SIGMA_0,
)


class TestHEPSystemsExist:
    """Test that example HEP systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """HEP_SYSTEMS dict should exist."""
        assert HEP_SYSTEMS is not None
        assert isinstance(HEP_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(HEP_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key HEP systems should be defined."""
        expected = ["lhc_13tev", "lattice_physical", "qgp_rhic", "rare_b_decay"]
        for name in expected:
            assert name in HEP_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in HEP_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "energy_scale")
            assert hasattr(system, "experiment_type")
            assert system.energy_scale > 0


class TestBeta0:
    """Test QCD beta function coefficient."""

    def test_qcd_5_flavor(self):
        """5-flavor QCD beta_0 is positive (asymptotic freedom)."""
        b0 = beta_0(n_f=5)
        assert b0 > 0

    def test_qcd_6_flavor(self):
        """6-flavor QCD still has asymptotic freedom."""
        b0 = beta_0(n_f=6)
        assert b0 > 0

    def test_more_flavors_smaller_beta(self):
        """More flavors reduce beta_0."""
        b0_3 = beta_0(n_f=3)
        b0_5 = beta_0(n_f=5)
        b0_6 = beta_0(n_f=6)
        assert b0_3 > b0_5 > b0_6

    def test_asymptotic_freedom_limit(self):
        """Asymptotic freedom requires n_f < 16.5 for SU(3)."""
        # 11 * 3 - 2 * n_f > 0 => n_f < 16.5
        b0_16 = beta_0(n_f=16)
        b0_17 = beta_0(n_f=17)
        assert b0_16 > 0
        assert b0_17 < 0  # Infrared freedom


class TestRunningAlphaS:
    """Test QCD running coupling."""

    def test_at_z_mass(self):
        """alpha_s at M_Z should match input."""
        alpha = running_alpha_s(M_Z)
        assert alpha == pytest.approx(ALPHA_S_MZ, rel=0.01)

    def test_asymptotic_freedom(self):
        """Higher energy gives smaller coupling."""
        alpha_low = running_alpha_s(10)
        alpha_high = running_alpha_s(1000)
        assert alpha_high < alpha_low

    def test_low_energy_strong_coupling(self):
        """Low energy approaches strong coupling."""
        alpha = running_alpha_s(LAMBDA_QCD * 2)
        assert alpha > ALPHA_S_MZ

    def test_very_high_energy(self):
        """Very high energy coupling is small."""
        alpha = running_alpha_s(10000)
        assert alpha < 0.1

    def test_zero_energy(self):
        """Zero energy returns maximum coupling."""
        alpha = running_alpha_s(0)
        assert alpha == 1.0

    def test_negative_energy(self):
        """Negative energy returns maximum coupling."""
        alpha = running_alpha_s(-10)
        assert alpha == 1.0


class TestQCDCouplingTheta:
    """Test QCD coupling theta calculation."""

    def test_high_energy_low_theta(self):
        """High energy (asymptotic freedom) gives low theta."""
        theta = compute_qcd_coupling_theta(10000)
        assert theta < 0.5

    def test_low_energy_high_theta(self):
        """Low energy (strong coupling) gives high theta."""
        theta = compute_qcd_coupling_theta(1)
        assert theta > 0.5

    def test_theta_in_range(self):
        """Theta should always be in [0, 1]."""
        for Q in [0.1, 1, 10, 100, 1000, 10000]:
            theta = compute_qcd_coupling_theta(Q)
            assert 0 <= theta <= 1


class TestLatticeBeta:
    """Test lattice QCD beta parameter."""

    def test_fine_lattice(self):
        """Fine lattice (small a) gives higher beta."""
        beta_fine = lattice_beta(0.05)
        beta_coarse = lattice_beta(0.15)
        assert beta_fine > beta_coarse

    def test_zero_spacing(self):
        """Zero spacing gives infinite beta (continuum)."""
        beta = lattice_beta(0)
        assert beta == float('inf')

    def test_typical_range(self):
        """Typical lattice beta is in reasonable range."""
        beta = lattice_beta(0.1)
        assert 5 <= beta <= 7


class TestLatticeTheta:
    """Test lattice spacing theta calculation."""

    def test_continuum_limit(self):
        """Very small spacing gives theta near 0."""
        theta = compute_lattice_theta(0.01)
        assert theta < 0.2

    def test_coarse_lattice(self):
        """Coarse lattice gives theta near 1."""
        theta = compute_lattice_theta(0.2)
        assert theta >= 0.99

    def test_typical_lattice(self):
        """Typical lattice gives intermediate theta."""
        theta = compute_lattice_theta(0.09)
        assert 0.3 < theta < 0.8

    def test_zero_spacing(self):
        """Zero spacing gives theta = 0."""
        theta = compute_lattice_theta(0)
        assert theta == 0.0

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for a in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
            theta = compute_lattice_theta(a)
            assert 0 <= theta <= 1


class TestStringTension:
    """Test temperature-dependent string tension."""

    def test_zero_temperature(self):
        """Zero temperature gives full string tension."""
        sigma = string_tension(SIGMA_0, 0)
        assert sigma == SIGMA_0

    def test_above_transition(self):
        """Above QGP transition, string tension vanishes."""
        sigma = string_tension(SIGMA_0, T_QGP * 1.5)
        assert sigma == 0.0

    def test_at_transition(self):
        """At transition, string tension vanishes."""
        sigma = string_tension(SIGMA_0, T_QGP)
        assert sigma == 0.0

    def test_below_transition(self):
        """Below transition, string tension is reduced."""
        sigma = string_tension(SIGMA_0, T_QGP * 0.5)
        assert 0 < sigma < SIGMA_0


class TestConfinementTheta:
    """Test confinement theta calculation."""

    def test_zero_temperature(self):
        """Zero temperature gives full confinement (theta = 1)."""
        theta = compute_confinement_theta(0)
        assert theta == 1.0

    def test_above_transition(self):
        """Above QGP transition, deconfined (theta = 0)."""
        theta = compute_confinement_theta(T_QGP * 1.5)
        assert theta == 0.0

    def test_intermediate_temperature(self):
        """Intermediate temperature gives intermediate theta."""
        theta = compute_confinement_theta(T_QGP * 0.5)
        assert 0 < theta < 1

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for T in [0, 0.05, 0.1, 0.15, 0.2]:
            theta = compute_confinement_theta(T)
            assert 0 <= theta <= 1


class TestChiralCondensate:
    """Test chiral condensate calculation."""

    def test_zero_temperature(self):
        """Zero temperature gives full condensate."""
        condensate = chiral_condensate(0, 1.0)
        assert condensate == 1.0

    def test_high_temperature(self):
        """High temperature restores chiral symmetry."""
        condensate = chiral_condensate(T_QGP * 1.5, 1.0)
        assert condensate == 0.0


class TestChiralTheta:
    """Test chiral symmetry theta calculation."""

    def test_physical_masses(self):
        """Physical quark mass ratio gives theta = 1."""
        theta = compute_chiral_theta(27.3, 27.3)
        assert theta == 1.0

    def test_chiral_limit(self):
        """Zero quark masses give theta = 0."""
        theta = compute_chiral_theta(0)
        assert theta == 0.0

    def test_intermediate(self):
        """Intermediate mass ratio gives intermediate theta."""
        theta = compute_chiral_theta(13.65, 27.3)
        assert theta == pytest.approx(0.5)


class TestCMSEnergy:
    """Test center of mass energy calculation."""

    def test_collider(self):
        """Collider sqrt(s) = 2 * E_beam."""
        sqrt_s = cms_energy(6500, is_collider=True)
        assert sqrt_s == 13000

    def test_fixed_target(self):
        """Fixed target gives lower sqrt(s)."""
        sqrt_s = cms_energy(100, is_collider=False)
        # sqrt(2 * 0.938 * 100) ~ 13.7 GeV
        assert sqrt_s < 20
        assert sqrt_s > 10


class TestPrecisionTheta:
    """Test precision measurement theta."""

    def test_achieved_target(self):
        """Achieving target precision gives theta = 1."""
        theta = compute_precision_theta(1e-6, 1e-6)
        assert theta == 1.0

    def test_worse_than_target(self):
        """Worse than target gives theta < 1."""
        theta = compute_precision_theta(1e-4, 1e-6)
        assert theta < 1.0

    def test_better_than_target(self):
        """Better than target clips to 1."""
        theta = compute_precision_theta(1e-8, 1e-6)
        assert theta == 1.0

    def test_zero_precision(self):
        """Zero precision gives theta = 0."""
        theta = compute_precision_theta(0, 1e-6)
        assert theta == 0.0


class TestUnifiedHEPTheta:
    """Test unified HEP theta calculation."""

    def test_all_systems_valid_theta(self):
        """All HEP systems should have theta in [0, 1]."""
        for name, system in HEP_SYSTEMS.items():
            theta = compute_hep_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_lhc_low_coupling(self):
        """LHC at high energy should have lower theta (asymptotic freedom)."""
        lhc = HEP_SYSTEMS["lhc_13tev"]
        charm = HEP_SYSTEMS["charm_physics"]
        theta_lhc = compute_hep_theta(lhc)
        theta_charm = compute_hep_theta(charm)
        assert theta_lhc < theta_charm

    def test_qgp_deconfined(self):
        """QGP above T_c has lower theta."""
        qgp_lhc = HEP_SYSTEMS["qgp_lhc"]
        theta = compute_hep_theta(qgp_lhc)
        # Above T_QGP, so confinement theta is low
        assert theta < 0.5


class TestClassifyParticleRegime:
    """Test particle physics regime classification."""

    def test_asymptotic(self):
        """Very low theta -> ASYMPTOTIC."""
        assert classify_particle_regime(0.05) == ParticleRegime.ASYMPTOTIC

    def test_perturbative(self):
        """Low theta -> PERTURBATIVE."""
        assert classify_particle_regime(0.2) == ParticleRegime.PERTURBATIVE

    def test_transition(self):
        """Medium theta -> TRANSITION."""
        assert classify_particle_regime(0.4) == ParticleRegime.TRANSITION

    def test_confined(self):
        """High theta -> CONFINED."""
        assert classify_particle_regime(0.7) == ParticleRegime.CONFINED

    def test_strongly_coupled(self):
        """Very high theta -> STRONGLY_COUPLED."""
        assert classify_particle_regime(0.95) == ParticleRegime.STRONGLY_COUPLED


class TestClassifyLatticeRegime:
    """Test lattice QCD regime classification."""

    def test_continuum(self):
        """Low theta -> CONTINUUM."""
        assert classify_lattice_regime(0.1) == LatticeRegime.CONTINUUM

    def test_fine(self):
        """Moderate low theta -> FINE."""
        assert classify_lattice_regime(0.3) == LatticeRegime.FINE

    def test_moderate(self):
        """Medium theta -> MODERATE."""
        assert classify_lattice_regime(0.5) == LatticeRegime.MODERATE

    def test_coarse(self):
        """High theta -> COARSE."""
        assert classify_lattice_regime(0.7) == LatticeRegime.COARSE

    def test_strong_coupling(self):
        """Very high theta -> STRONG_COUPLING."""
        assert classify_lattice_regime(0.9) == LatticeRegime.STRONG_COUPLING


class TestClassifyChiralRegime:
    """Test chiral regime classification."""

    def test_restored(self):
        """Low theta -> RESTORED."""
        assert classify_chiral_regime(0.1) == ChiralRegime.RESTORED

    def test_partial(self):
        """Medium-low theta -> PARTIAL."""
        assert classify_chiral_regime(0.35) == ChiralRegime.PARTIAL

    def test_broken(self):
        """Medium-high theta -> BROKEN."""
        assert classify_chiral_regime(0.65) == ChiralRegime.BROKEN

    def test_deeply_broken(self):
        """High theta -> DEEPLY_BROKEN."""
        assert classify_chiral_regime(0.9) == ChiralRegime.DEEPLY_BROKEN


class TestEnums:
    """Test enum definitions."""

    def test_particle_regimes(self):
        """All particle regimes should be defined."""
        assert ParticleRegime.ASYMPTOTIC.value == "asymptotic"
        assert ParticleRegime.PERTURBATIVE.value == "perturbative"
        assert ParticleRegime.TRANSITION.value == "transition"
        assert ParticleRegime.CONFINED.value == "confined"
        assert ParticleRegime.STRONGLY_COUPLED.value == "strongly_coupled"

    def test_lattice_regimes(self):
        """All lattice regimes should be defined."""
        assert LatticeRegime.CONTINUUM.value == "continuum"
        assert LatticeRegime.FINE.value == "fine"
        assert LatticeRegime.MODERATE.value == "moderate"
        assert LatticeRegime.COARSE.value == "coarse"
        assert LatticeRegime.STRONG_COUPLING.value == "strong_coupling"

    def test_experiment_types(self):
        """All experiment types should be defined."""
        assert ExperimentType.COLLIDER.value == "collider"
        assert ExperimentType.FIXED_TARGET.value == "fixed_target"
        assert ExperimentType.RARE_DECAY.value == "rare_decay"
        assert ExperimentType.PRECISION.value == "precision"
        assert ExperimentType.HEAVY_ION.value == "heavy_ion"


class TestHEPSystemDataclass:
    """Test HEPSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with required parameters."""
        system = HEPSystem(
            name="Test",
            energy_scale=100.0
        )
        assert system.name == "Test"
        assert system.energy_scale == 100.0

    def test_default_values(self):
        """Optional fields have defaults."""
        system = HEPSystem(
            name="Test",
            energy_scale=100.0
        )
        assert system.coupling_alpha_s is None
        assert system.lattice_spacing is None
        assert system.temperature is None
        assert system.n_colors == 3
        assert system.n_flavors == 6

    def test_custom_values(self):
        """Can set custom values."""
        system = HEPSystem(
            name="Custom",
            energy_scale=50.0,
            coupling_alpha_s=0.3,
            lattice_spacing=0.1,
            n_colors=3,
            n_flavors=4
        )
        assert system.coupling_alpha_s == 0.3
        assert system.lattice_spacing == 0.1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_energy(self):
        """Very high energy stays in valid range."""
        theta = compute_qcd_coupling_theta(1e6)
        assert 0 <= theta <= 1

    def test_very_low_energy(self):
        """Very low energy stays in valid range."""
        theta = compute_qcd_coupling_theta(0.1)
        assert 0 <= theta <= 1

    def test_negative_temperature(self):
        """Negative temperature treated as zero."""
        sigma = string_tension(SIGMA_0, -1)
        assert sigma == SIGMA_0

    def test_zero_sigma_0(self):
        """Zero sigma_0 gives theta = 0."""
        theta = compute_confinement_theta(0.1, sigma_0=0)
        assert theta == 0.0

    def test_negative_lattice_spacing(self):
        """Negative lattice spacing gives theta = 0."""
        theta = compute_lattice_theta(-0.1)
        assert theta == 0.0


class TestPhysicalConstants:
    """Test that physical constants are reasonable."""

    def test_lambda_qcd(self):
        """Lambda_QCD should be ~200 MeV."""
        assert 0.1 < LAMBDA_QCD < 0.5  # GeV

    def test_alpha_s_mz(self):
        """alpha_s at M_Z should be ~0.12."""
        assert 0.10 < ALPHA_S_MZ < 0.15

    def test_mz(self):
        """Z mass should be ~91 GeV."""
        assert 90 < M_Z < 92

    def test_t_qgp(self):
        """QGP temperature should be ~150 MeV."""
        assert 0.1 < T_QGP < 0.2  # GeV

    def test_sigma_0(self):
        """String tension should be ~0.4 GeV^2."""
        assert 0.3 < SIGMA_0 < 0.6
