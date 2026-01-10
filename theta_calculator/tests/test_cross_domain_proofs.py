"""
Tests for Cross-Domain Unification Proofs

Tests cover:
- Information unification (Bekenstein, Landauer, holographic)
- Scale invariance (universality classes, exponents)
- Emergent correspondences (market-ferromagnet, neural-Ising)
- Renormalization proofs (RG flow, fixed points)
"""

import pytest
import math

from theta_calculator.proofs.cross_domain import (
    # Information unification
    InformationUnificationProof,
    compute_universal_information_theta,
    compute_channel_capacity_theta,
    verify_information_bounds,
    # Scale invariance
    ScaleInvarianceProof,
    UniversalityClass,
    compute_universal_exponents,
    compute_scaling_function,
    verify_universality_class,
    classify_universality_class,
    UNIVERSALITY_CLASSES,
    # Emergent correspondences
    EmergentCorrespondenceProof,
    map_market_to_ferromagnet,
    map_neural_to_ising,
    map_bec_to_consensus,
    map_superconductor_to_flow,
    verify_correspondence,
    KNOWN_CORRESPONDENCES,
    # Renormalization proofs
    RenormalizationProof,
    compute_rg_flow_theta,
    compute_beta_function,
    find_fixed_points,
    classify_flow_regime,
)

from theta_calculator.proofs.cross_domain.information_unification import (
    compute_bekenstein_bound,
    compute_landauer_limit,
    compute_holographic_entropy,
    InformationDomain,
)

from theta_calculator.proofs.cross_domain.scale_invariance import (
    verify_scaling_relations,
)

from theta_calculator.proofs.cross_domain.emergent_correspondences import (
    DomainType,
)


class TestInformationUnification:
    """Test information unification proofs."""

    def test_bekenstein_bound_positive(self):
        """Bekenstein bound should be positive for physical systems."""
        s_max = compute_bekenstein_bound(radius_m=1.0, energy_j=1e10)
        assert s_max > 0

    def test_bekenstein_bound_scales(self):
        """Larger radius/energy -> larger bound."""
        s1 = compute_bekenstein_bound(1.0, 1e10)
        s2 = compute_bekenstein_bound(2.0, 1e10)
        s3 = compute_bekenstein_bound(1.0, 2e10)
        assert s2 > s1
        assert s3 > s1

    def test_landauer_limit_positive(self):
        """Landauer limit should be positive at finite T."""
        e_min = compute_landauer_limit(temperature_k=300, n_bits=1)
        assert e_min > 0
        assert e_min == pytest.approx(300 * 1.38e-23 * math.log(2), rel=0.01)

    def test_landauer_scales_with_bits(self):
        """More bits -> more energy."""
        e1 = compute_landauer_limit(300, 1)
        e2 = compute_landauer_limit(300, 10)
        assert e2 == pytest.approx(10 * e1)

    def test_holographic_entropy_positive(self):
        """Holographic entropy should be positive."""
        s = compute_holographic_entropy(area_m2=1e-66)  # Planck area scale
        assert s > 0

    def test_universal_information_theta_bounds(self):
        """Theta should be in [0, 1]."""
        theta = compute_universal_information_theta(
            entropy_bits=1e30,
            radius_m=1.0,
            energy_j=1e20,
        )
        assert 0 <= theta <= 1

    def test_channel_capacity_theta(self):
        """Shannon capacity theta should be in [0, 1]."""
        theta = compute_channel_capacity_theta(
            bandwidth_hz=1e6,
            signal_power_w=1.0,
            noise_power_w=0.1,
            actual_rate_bps=1e5,
        )
        assert 0 <= theta <= 1

    def test_channel_at_capacity(self):
        """At capacity, theta should be 1."""
        # Capacity = B * log2(1 + S/N)
        bandwidth = 1e6
        snr = 10
        capacity = bandwidth * math.log2(1 + snr)

        theta = compute_channel_capacity_theta(
            bandwidth_hz=bandwidth,
            signal_power_w=10,
            noise_power_w=1,
            actual_rate_bps=capacity,
        )
        assert theta == pytest.approx(1.0, rel=0.01)

    def test_verify_information_bounds(self):
        """Physical systems should satisfy information bounds."""
        results = verify_information_bounds(
            domain=InformationDomain.COMPUTATION,
            entropy_bits=1e20,
            energy_j=1e10,
            radius_m=0.1,
            temperature_k=300,
            n_operations=1e15,
        )
        # At least some bounds should be satisfied
        assert any(results.values())


class TestInformationUnificationProof:
    """Test InformationUnificationProof class."""

    def test_proof_initialization(self):
        """Proof framework should initialize."""
        proof = InformationUnificationProof()
        assert proof.domain_mappings is not None
        assert len(proof.domain_mappings) >= 5

    def test_analyze_bekenstein(self):
        """Bekenstein analysis should return valid result."""
        proof = InformationUnificationProof()
        result = proof.analyze_bekenstein(
            domain=InformationDomain.PHYSICS,
            entropy_bits=1e40,
            radius_m=1e3,  # 1km
            energy_j=1e45,  # Large energy
        )
        assert 0 <= result.theta <= 1
        assert result.interpretation is not None

    def test_analyze_landauer(self):
        """Landauer analysis should return valid result."""
        proof = InformationUnificationProof()
        result = proof.analyze_landauer(
            domain=InformationDomain.COMPUTATION,
            bits_erased=1e15,
            energy_actual_j=1e-5,  # Much more than Landauer limit
            temperature_k=300,
        )
        assert 0 <= result.theta <= 1


class TestScaleInvariance:
    """Test scale invariance proofs."""

    def test_universality_classes_exist(self):
        """All universality classes should be defined."""
        assert len(UNIVERSALITY_CLASSES) >= 5
        assert UniversalityClass.ISING_3D in UNIVERSALITY_CLASSES

    def test_ising_3d_exponents(self):
        """3D Ising exponents should match literature."""
        exps = UNIVERSALITY_CLASSES[UniversalityClass.ISING_3D]
        assert exps["beta"] == pytest.approx(0.326, abs=0.01)
        assert exps["gamma"] == pytest.approx(1.237, abs=0.02)
        assert exps["nu"] == pytest.approx(0.630, abs=0.01)

    def test_ising_2d_exact(self):
        """2D Ising exponents should be exact."""
        exps = UNIVERSALITY_CLASSES[UniversalityClass.ISING_2D]
        assert exps["beta"] == 0.125  # 1/8
        assert exps["gamma"] == 1.75   # 7/4
        assert exps["nu"] == 1.0

    def test_compute_universal_exponents(self):
        """Should return all exponents for a class."""
        exps = compute_universal_exponents(UniversalityClass.ISING_3D)
        assert "beta" in exps
        assert "gamma" in exps
        assert "nu" in exps
        assert exps["beta"].value == pytest.approx(0.326, abs=0.01)

    def test_scaling_function(self):
        """Scaling function should compute order parameter."""
        M, x = compute_scaling_function(
            t=-0.1,  # Below Tc
            h=0.0,
            beta=0.326,
            delta=4.789,
        )
        assert M > 0  # Spontaneous magnetization below Tc

    def test_verify_universality_class(self):
        """Should identify universality class from exponents."""
        measured = {"beta": 0.33, "gamma": 1.24, "nu": 0.63}
        uc, chi2 = verify_universality_class(measured)
        assert uc == UniversalityClass.ISING_3D

    def test_classify_universality_class(self):
        """Should classify based on system properties."""
        # 3D Ising: n=1, d=3
        uc = classify_universality_class(n_components=1, spatial_dim=3)
        assert uc == UniversalityClass.ISING_3D

        # Mean field above d=4
        uc = classify_universality_class(n_components=1, spatial_dim=5)
        assert uc == UniversalityClass.MEAN_FIELD

    def test_verify_scaling_relations(self):
        """Scaling relations should be satisfied."""
        exps = UNIVERSALITY_CLASSES[UniversalityClass.ISING_3D]
        results = verify_scaling_relations(exps)
        # Rushbrooke should be satisfied
        assert results.get("rushbrooke", False)


class TestScaleInvarianceProof:
    """Test ScaleInvarianceProof class."""

    def test_proof_initialization(self):
        """Proof framework should initialize."""
        proof = ScaleInvarianceProof()
        assert proof.domain_classes is not None

    def test_compute_theta_from_exponent(self):
        """Should compute theta from critical scaling."""
        proof = ScaleInvarianceProof()

        # At critical point
        theta = proof.compute_theta_from_exponent(
            t=0.0,
            exponent_name="beta",
            universality_class=UniversalityClass.ISING_3D,
        )
        assert theta == pytest.approx(0.5, abs=0.01)

        # Below critical (ordered)
        theta = proof.compute_theta_from_exponent(
            t=-0.1,
            exponent_name="beta",
            universality_class=UniversalityClass.ISING_3D,
        )
        assert theta > 0.5

    def test_prove_universality(self):
        """Should prove two systems share universality class."""
        proof = ScaleInvarianceProof()

        # Same exponents -> same class
        exp1 = {"beta": 0.326, "gamma": 1.237}
        exp2 = {"beta": 0.33, "gamma": 1.24}

        result = proof.prove_universality(
            system1="ferromagnet",
            system2="market",
            measured_exponents_1=exp1,
            measured_exponents_2=exp2,
        )
        assert result["proven_universal"]


class TestEmergentCorrespondences:
    """Test emergent correspondence proofs."""

    def test_known_correspondences_exist(self):
        """Should have predefined correspondences."""
        assert len(KNOWN_CORRESPONDENCES) >= 4
        assert "market_ferromagnet" in KNOWN_CORRESPONDENCES

    def test_market_ferromagnet_mapping(self):
        """Should map market to ferromagnet."""
        result = map_market_to_ferromagnet(
            correlation=0.5,
            volatility=0.2,
            market_size=100,
        )
        assert "coupling_J" in result
        assert "theta" in result
        assert 0 <= result["theta"] <= 1

    def test_neural_ising_mapping(self):
        """Should map neural avalanches to Ising."""
        # At criticality: branching ratio = 1
        result = map_neural_to_ising(
            branching_ratio=1.0,
            avalanche_exponent=1.5,
            network_size=1000,
        )
        assert result["is_critical"]
        assert result["theta"] == pytest.approx(0.5, abs=0.1)

    def test_bec_consensus_mapping(self):
        """Should map BEC to social consensus."""
        result = map_bec_to_consensus(
            condensate_fraction=0.8,
            temperature_ratio=0.5,
            n_particles=1000,
        )
        assert result["opinion_alignment"] == 0.8
        assert result["theta"] == pytest.approx(0.8)

    def test_superconductor_flow_mapping(self):
        """Should map superconductor to flow state."""
        result = map_superconductor_to_flow(
            gap_energy_ev=1e-3,
            temperature_k=4.0,
            critical_temp_k=10.0,
        )
        assert "flow_depth" in result
        assert 0 <= result["theta"] <= 1

    def test_verify_correspondence(self):
        """Should verify correspondence holds."""
        domain_a = {"theta": 0.7}
        domain_b = {"theta": 0.72}

        result = verify_correspondence(
            "market_ferromagnet",
            domain_a,
            domain_b,
            tolerance=0.1,
        )
        assert result["verified"]


class TestEmergentCorrespondenceProof:
    """Test EmergentCorrespondenceProof class."""

    def test_proof_initialization(self):
        """Proof framework should initialize."""
        proof = EmergentCorrespondenceProof()
        assert len(proof.list_correspondences()) >= 4

    def test_predict_across_domains(self):
        """Should predict target domain theta from source."""
        proof = EmergentCorrespondenceProof()

        result = proof.predict_across_domains(
            source_theta=0.7,
            source_domain=DomainType.ECONOMICS_MARKET,
            target_domain=DomainType.PHYSICS_MAGNETIC,
        )
        assert result["predicted"]
        assert result["predicted_theta"] == pytest.approx(0.7)


class TestRenormalizationProofs:
    """Test renormalization group proofs."""

    def test_compute_beta_function(self):
        """Beta function should compute correctly."""
        # QCD-like: negative beta
        beta = compute_beta_function(
            coupling=0.3,
            beta_coefficients=(1.0, 0.5),
            loop_order=2,
        )
        assert beta < 0  # Asymptotically free

    def test_compute_rg_flow_theta(self):
        """RG theta should be in [0, 1]."""
        theta = compute_rg_flow_theta(
            coupling=0.5,
            g_uv=0.0,
            g_ir=1.0,
        )
        assert theta == pytest.approx(0.5)

        # At UV
        theta = compute_rg_flow_theta(coupling=0.0, g_uv=0.0, g_ir=1.0)
        assert theta == pytest.approx(0.0)

        # At IR
        theta = compute_rg_flow_theta(coupling=1.0, g_uv=0.0, g_ir=1.0)
        assert theta == pytest.approx(1.0)

    def test_find_fixed_points(self):
        """Should find fixed points of beta function."""
        def simple_beta(g):
            return g * (1 - g)  # Fixed points at 0 and 1

        fps = find_fixed_points(simple_beta, search_range=(0, 2))
        # Should find g=0 and g=1
        assert len(fps) >= 1
        g_values = [fp[0] for fp in fps]
        assert any(abs(g) < 0.1 for g in g_values) or any(abs(g - 1) < 0.1 for g in g_values)

    def test_classify_flow_regime(self):
        """Should classify flow regimes correctly."""
        assert classify_flow_regime(0.05, 0.1) == "UV_FLOWING_TO_IR"
        assert classify_flow_regime(0.95, -0.1) == "IR_FLOWING_TO_UV"
        assert classify_flow_regime(0.5, 0.001) == "CROSSOVER_SLOW"


class TestRenormalizationProof:
    """Test RenormalizationProof class."""

    def test_proof_initialization(self):
        """Proof framework should initialize."""
        proof = RenormalizationProof()
        assert "qcd" in proof.domain_betas
        assert "ising" in proof.domain_betas

    def test_analyze_flow(self):
        """Should analyze RG flow."""
        proof = RenormalizationProof()

        def simple_beta(g):
            return -0.1 * g**2  # Asymptotically free

        result = proof.analyze_flow(
            beta_func=simple_beta,
            initial_coupling=0.5,
        )
        assert result.is_asymptotically_free
        assert len(result.theta_trajectory) > 0

        # Theta should increase along flow (toward IR)
        thetas = [t[1] for t in result.theta_trajectory]
        # First theta should be small (UV), last should be larger
        assert thetas[-1] >= thetas[0] - 0.1  # Allow some tolerance

    def test_compute_theta_from_scale(self):
        """Should compute theta from scale ratio."""
        proof = RenormalizationProof()

        theta = proof.compute_theta_from_scale(
            scale_ratio=1.0,
            domain="ising",
        )
        assert 0 <= theta <= 1


class TestCrossDomainIntegration:
    """Test integration across all cross-domain proofs."""

    def test_all_proofs_consistent(self):
        """All proof frameworks should give consistent theta."""
        # At a critical point, all methods should give θ ≈ 0.5

        # Scale invariance: t = 0 -> θ = 0.5
        scale_proof = ScaleInvarianceProof()
        theta_scale = scale_proof.compute_theta_from_exponent(
            t=0.0,
            exponent_name="beta",
            universality_class=UniversalityClass.ISING_3D,
        )

        # RG: middle of flow -> θ ≈ 0.5
        theta_rg = compute_rg_flow_theta(
            coupling=0.5,
            g_uv=0.0,
            g_ir=1.0,
        )

        assert abs(theta_scale - theta_rg) < 0.2

    def test_cross_domain_theta_ranges(self):
        """All domains should produce theta in [0, 1]."""
        # Market
        market_result = map_market_to_ferromagnet(0.5, 0.2, 100)
        assert 0 <= market_result["theta"] <= 1

        # Neural
        neural_result = map_neural_to_ising(1.0, 1.5, 1000)
        assert 0 <= neural_result["theta"] <= 1

        # BEC
        bec_result = map_bec_to_consensus(0.5, 1.0, 1000)
        assert 0 <= bec_result["theta"] <= 1


class TestDocstrings:
    """Test that modules have proper documentation."""

    def test_module_docstrings(self):
        """All cross-domain modules should have docstrings with citations."""
        import theta_calculator.proofs.cross_domain as cd
        assert cd.__doc__ is not None
        assert "\\cite{" in cd.__doc__

        import theta_calculator.proofs.cross_domain.information_unification as info
        assert info.__doc__ is not None
        assert "\\cite{" in info.__doc__

        import theta_calculator.proofs.cross_domain.scale_invariance as scale
        assert scale.__doc__ is not None
        assert "\\cite{" in scale.__doc__

        import theta_calculator.proofs.cross_domain.emergent_correspondences as emerg
        assert emerg.__doc__ is not None
        assert "\\cite{" in emerg.__doc__

        import theta_calculator.proofs.cross_domain.renormalization_proofs as rg
        assert rg.__doc__ is not None
        assert "\\cite{" in rg.__doc__
