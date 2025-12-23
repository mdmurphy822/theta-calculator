"""
Tests for work-life balance domain module.

Tests verify theta mappings for burnout, effort-reward imbalance,
work-family conflict, cognitive load, and job demands-resources models.

References:
    Maslach1981 - Maslach Burnout Inventory
    Siegrist1996 - Effort-Reward Imbalance model
    Greenhaus1985 - Work-family conflict theory
    Sweller1988 - Cognitive Load Theory
    Demerouti2001 - Job Demands-Resources model
    Karasek1979 - Job strain model
"""

import pytest

from theta_calculator.domains import (
    # Core classes
    WorkLifeSystem,
    WellbeingPhase,
    BurnoutDimension,
    ConflictDirection,
    # Burnout functions
    compute_burnout_theta,
    classify_burnout,
    # Effort-Reward functions
    compute_effort_reward_theta,
    # Work-Family functions
    compute_work_family_conflict_theta,
    # Cognitive Load functions
    compute_cognitive_load_theta,
    # JD-R functions
    compute_jdr_theta,
    compute_recovery_theta,
    # Job Strain functions
    compute_job_strain_theta,
    classify_job_strain,
    # Composite functions
    compute_work_life_theta,
    classify_wellbeing_phase,
    # Example systems
    WORK_LIFE_SYSTEMS,
)


class TestWorkLifeSystemsExist:
    """Tests for example system definitions."""

    def test_example_systems_exist(self):
        """Verify example work-life systems are defined."""
        assert len(WORK_LIFE_SYSTEMS) >= 5
        assert "balanced_professional" in WORK_LIFE_SYSTEMS
        assert "burnout_case" in WORK_LIFE_SYSTEMS
        assert "overworked_parent" in WORK_LIFE_SYSTEMS

    def test_system_attributes(self):
        """Verify WorkLifeSystem has expected attributes."""
        system = WORK_LIFE_SYSTEMS["balanced_professional"]
        assert hasattr(system, "name")
        assert hasattr(system, "exhaustion")
        assert hasattr(system, "cynicism")
        assert hasattr(system, "efficacy")
        assert hasattr(system, "effort")
        assert hasattr(system, "reward")

    def test_enums_defined(self):
        """Verify enums are properly defined."""
        assert WellbeingPhase.THRIVING.value == "thriving"
        assert WellbeingPhase.BURNOUT.value == "burnout"
        assert BurnoutDimension.EXHAUSTION.value == "exhaustion"
        assert ConflictDirection.WORK_TO_FAMILY.value == "work_to_family"


class TestBurnoutTheta:
    """Tests for Maslach Burnout Inventory theta mapping."""

    def test_burnout_zero_scores(self):
        """Zero burnout scores should give theta ~ 0."""
        theta = compute_burnout_theta(0, 0, 6.0)  # Perfect efficacy
        assert theta == pytest.approx(0.0, abs=0.01)

    def test_burnout_max_scores(self):
        """Maximum burnout scores should give theta ~ 1."""
        theta = compute_burnout_theta(6, 6, 0)  # Max exhaustion, cynicism, zero efficacy
        assert theta == pytest.approx(1.0, abs=0.01)

    def test_burnout_moderate(self):
        """Moderate scores should give theta ~ 0.5."""
        theta = compute_burnout_theta(3, 3, 3)  # Middle scores
        assert 0.4 <= theta <= 0.6

    def test_burnout_theta_range(self):
        """Burnout theta should always be in [0, 1]."""
        for exhaustion in [0, 2, 4, 6]:
            for cynicism in [0, 2, 4, 6]:
                for efficacy in [0, 2, 4, 6]:
                    theta = compute_burnout_theta(exhaustion, cynicism, efficacy)
                    assert 0 <= theta <= 1

    def test_burnout_classification(self):
        """Test burnout classification levels."""
        assert classify_burnout(0.1) == "low_burnout"
        assert classify_burnout(0.4) == "moderate_burnout"
        assert classify_burnout(0.6) == "high_burnout"
        assert classify_burnout(0.8) == "severe_burnout"


class TestEffortRewardTheta:
    """Tests for Siegrist Effort-Reward Imbalance model."""

    def test_balanced_effort_reward(self):
        """Equal effort and reward should give moderate theta."""
        theta = compute_effort_reward_theta(50, 50)
        assert 0.3 <= theta <= 0.7

    def test_high_effort_low_reward(self):
        """High effort, low reward should give high theta."""
        theta = compute_effort_reward_theta(90, 30)
        assert theta > 0.5  # Higher than balanced (0.5)

    def test_low_effort_high_reward(self):
        """Low effort, high reward should give low theta."""
        theta = compute_effort_reward_theta(30, 90)
        assert theta < 0.3

    def test_zero_reward(self):
        """Zero reward should give theta = 1."""
        theta = compute_effort_reward_theta(50, 0)
        assert theta == 1.0

    def test_effort_reward_range(self):
        """Effort-reward theta should be in [0, 1]."""
        for effort in [0, 25, 50, 75, 100]:
            for reward in [10, 25, 50, 75, 100]:  # Avoid zero
                theta = compute_effort_reward_theta(effort, reward)
                assert 0 <= theta <= 1


class TestWorkFamilyConflictTheta:
    """Tests for work-family conflict theta mapping."""

    def test_no_conflict(self):
        """No conflict should give theta = 0."""
        theta = compute_work_family_conflict_theta(0, 0)
        assert theta == pytest.approx(0.0, abs=0.01)

    def test_max_conflict(self):
        """Maximum conflict should give theta = 1."""
        theta = compute_work_family_conflict_theta(5, 5)
        assert theta == pytest.approx(1.0, abs=0.01)

    def test_one_direction_conflict(self):
        """Conflict in one direction should give theta = 0.5."""
        theta_wif = compute_work_family_conflict_theta(5, 0)
        theta_fiw = compute_work_family_conflict_theta(0, 5)
        assert theta_wif == pytest.approx(0.5, abs=0.01)
        assert theta_fiw == pytest.approx(0.5, abs=0.01)

    def test_conflict_symmetric(self):
        """WIF and FIW should contribute equally."""
        theta1 = compute_work_family_conflict_theta(3, 2)
        theta2 = compute_work_family_conflict_theta(2, 3)
        assert theta1 == pytest.approx(theta2, abs=0.01)


class TestCognitiveLoadTheta:
    """Tests for Sweller's Cognitive Load Theory."""

    def test_no_load(self):
        """No cognitive load should give theta = 0."""
        theta = compute_cognitive_load_theta(0, 0)
        assert theta == pytest.approx(0.0, abs=0.01)

    def test_full_capacity(self):
        """Load at capacity should give theta = 1."""
        theta = compute_cognitive_load_theta(50, 50, capacity=100)
        assert theta == pytest.approx(1.0, abs=0.01)

    def test_overload_capped(self):
        """Overload should be capped at theta = 1."""
        theta = compute_cognitive_load_theta(80, 80, capacity=100)
        assert theta == 1.0

    def test_intrinsic_only(self):
        """Intrinsic load only should work."""
        theta = compute_cognitive_load_theta(30, 0)
        assert theta == pytest.approx(0.3, abs=0.01)

    def test_load_classification(self):
        """Test cognitive load classification."""
        from theta_calculator.domains.work_life_balance import classify_cognitive_load
        assert classify_cognitive_load(0.2) == "low_load"
        assert classify_cognitive_load(0.5) == "optimal_load"
        assert classify_cognitive_load(0.8) == "high_load"
        assert classify_cognitive_load(0.95) == "overload"


class TestJDRModel:
    """Tests for Job Demands-Resources model."""

    def test_balanced_jdr(self):
        """Equal demands and resources should give theta = 0.5."""
        theta = compute_jdr_theta(50, 50)
        assert theta == pytest.approx(0.5, abs=0.01)

    def test_high_demands_low_resources(self):
        """High demands, low resources = high theta."""
        theta = compute_jdr_theta(80, 20)
        assert theta > 0.7

    def test_low_demands_high_resources(self):
        """Low demands, high resources = low theta."""
        theta = compute_jdr_theta(20, 80)
        assert theta < 0.3

    def test_zero_total(self):
        """Zero demands and resources should give theta = 0.5."""
        theta = compute_jdr_theta(0, 0)
        assert theta == 0.5

    def test_jdr_range(self):
        """JDR theta should be in [0, 1]."""
        for demands in [0, 25, 50, 75, 100]:
            for resources in [0, 25, 50, 75, 100]:
                theta = compute_jdr_theta(demands, resources)
                assert 0 <= theta <= 1


class TestRecoveryTheta:
    """Tests for recovery-demand ratio."""

    def test_full_recovery(self):
        """Full recovery (ratio >= 1) should give theta = 0."""
        theta = compute_recovery_theta(10, 8)  # More recovery than demands
        assert theta == pytest.approx(0.0, abs=0.01)

    def test_no_recovery(self):
        """No recovery should give theta = 1."""
        theta = compute_recovery_theta(0, 8)
        assert theta == pytest.approx(1.0, abs=0.01)

    def test_half_recovery(self):
        """Half recovery should give theta = 0.5."""
        theta = compute_recovery_theta(4, 8)
        assert theta == pytest.approx(0.5, abs=0.01)

    def test_zero_demands(self):
        """Zero demands should give theta = 0."""
        theta = compute_recovery_theta(8, 0)
        assert theta == 0.0


class TestJobStrainTheta:
    """Tests for Karasek's Job Strain model."""

    def test_low_strain(self):
        """Low demands, high control = low strain."""
        theta = compute_job_strain_theta(20, 80)
        assert theta < 0.4

    def test_high_strain(self):
        """High demands, low control = high strain."""
        theta = compute_job_strain_theta(80, 20)
        assert theta > 0.6

    def test_active_job(self):
        """High demands, high control = active (moderate theta)."""
        theta = compute_job_strain_theta(80, 80)
        assert 0.3 <= theta <= 0.6

    def test_support_buffers(self):
        """Social support should buffer strain."""
        theta_no_support = compute_job_strain_theta(70, 30, support=0)
        theta_support = compute_job_strain_theta(70, 30, support=50)
        assert theta_support < theta_no_support

    def test_strain_classification(self):
        """Test job strain classification."""
        assert classify_job_strain(0.2) == "low_strain"
        assert classify_job_strain(0.4) == "active"
        assert classify_job_strain(0.6) == "passive"
        assert classify_job_strain(0.8) == "high_strain"


class TestCompositeWorkLifeTheta:
    """Tests for composite work-life theta computation."""

    def test_composite_returns_dict(self):
        """Composite function should return dictionary."""
        system = WORK_LIFE_SYSTEMS["balanced_professional"]
        result = compute_work_life_theta(system)
        assert isinstance(result, dict)
        assert "burnout" in result
        assert "effort_reward" in result
        assert "work_family_conflict" in result
        assert "cognitive_load" in result
        assert "recovery_deficit" in result
        assert "composite" in result
        assert "phase" in result

    def test_composite_range(self):
        """All composite values should be in [0, 1]."""
        for name, system in WORK_LIFE_SYSTEMS.items():
            result = compute_work_life_theta(system)
            assert 0 <= result["burnout"] <= 1
            assert 0 <= result["effort_reward"] <= 1
            assert 0 <= result["work_family_conflict"] <= 1
            assert 0 <= result["cognitive_load"] <= 1
            assert 0 <= result["recovery_deficit"] <= 1
            assert 0 <= result["composite"] <= 1

    def test_balanced_professional_low_theta(self):
        """Balanced professional should have low composite theta."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["balanced_professional"])
        assert result["composite"] < 0.4
        assert result["phase"] in [WellbeingPhase.THRIVING, WellbeingPhase.BALANCED]

    def test_burnout_case_high_theta(self):
        """Burnout case should have high composite theta."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["burnout_case"])
        assert result["composite"] > 0.6
        assert result["phase"] in [WellbeingPhase.AT_RISK, WellbeingPhase.BURNOUT]

    def test_theta_ordering(self):
        """Burnout case should have higher theta than balanced professional."""
        balanced = compute_work_life_theta(WORK_LIFE_SYSTEMS["balanced_professional"])
        burnout = compute_work_life_theta(WORK_LIFE_SYSTEMS["burnout_case"])
        assert burnout["composite"] > balanced["composite"]


class TestWellbeingPhaseClassification:
    """Tests for wellbeing phase classification."""

    def test_thriving_phase(self):
        """Low theta should classify as thriving."""
        phase = classify_wellbeing_phase(0.1)
        assert phase == WellbeingPhase.THRIVING

    def test_balanced_phase(self):
        """Moderate low theta should classify as balanced."""
        phase = classify_wellbeing_phase(0.3)
        assert phase == WellbeingPhase.BALANCED

    def test_strained_phase(self):
        """Moderate theta should classify as strained."""
        phase = classify_wellbeing_phase(0.5)
        assert phase == WellbeingPhase.STRAINED

    def test_at_risk_phase(self):
        """High theta should classify as at_risk."""
        phase = classify_wellbeing_phase(0.7)
        assert phase == WellbeingPhase.AT_RISK

    def test_burnout_phase(self):
        """Very high theta should classify as burnout."""
        phase = classify_wellbeing_phase(0.9)
        assert phase == WellbeingPhase.BURNOUT

    def test_phase_boundaries(self):
        """Test exact phase boundaries."""
        assert classify_wellbeing_phase(0.19) == WellbeingPhase.THRIVING
        assert classify_wellbeing_phase(0.20) == WellbeingPhase.BALANCED
        assert classify_wellbeing_phase(0.39) == WellbeingPhase.BALANCED
        assert classify_wellbeing_phase(0.40) == WellbeingPhase.STRAINED
        assert classify_wellbeing_phase(0.59) == WellbeingPhase.STRAINED
        assert classify_wellbeing_phase(0.60) == WellbeingPhase.AT_RISK
        assert classify_wellbeing_phase(0.79) == WellbeingPhase.AT_RISK
        assert classify_wellbeing_phase(0.80) == WellbeingPhase.BURNOUT


class TestExampleSystemsTheta:
    """Tests for expected theta values of example systems."""

    def test_engaged_employee_thriving(self):
        """Engaged employee should be thriving or balanced."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["engaged_employee"])
        assert result["phase"] in [WellbeingPhase.THRIVING, WellbeingPhase.BALANCED]

    def test_healthcare_worker_strained(self):
        """Healthcare worker should be strained or at risk."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["healthcare_worker"])
        assert result["phase"] in [WellbeingPhase.STRAINED, WellbeingPhase.AT_RISK, WellbeingPhase.BURNOUT]

    def test_overworked_parent_high_conflict(self):
        """Overworked parent should have high work-family conflict."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["overworked_parent"])
        assert result["work_family_conflict"] > 0.5

    def test_remote_worker_moderate(self):
        """Remote worker should have moderate theta."""
        result = compute_work_life_theta(WORK_LIFE_SYSTEMS["remote_worker"])
        assert 0.2 <= result["composite"] <= 0.6


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zeros_system(self):
        """System with all zeros should not crash."""
        system = WorkLifeSystem(
            name="all_zeros",
            exhaustion=0,
            cynicism=0,
            efficacy=6,
            effort=0,
            reward=1,  # Avoid division by zero
            work_to_family_conflict=0,
            family_to_work_conflict=0,
            cognitive_load=0,
            recovery_time=0,
            work_demands=1  # Avoid division by zero
        )
        result = compute_work_life_theta(system)
        assert 0 <= result["composite"] <= 1

    def test_extreme_values(self):
        """Extreme values should be handled gracefully."""
        system = WorkLifeSystem(
            name="extreme",
            exhaustion=6,
            cynicism=6,
            efficacy=0,
            effort=100,
            reward=1,
            work_to_family_conflict=5,
            family_to_work_conflict=5,
            cognitive_load=100,
            recovery_time=0,
            work_demands=24
        )
        result = compute_work_life_theta(system)
        assert result["composite"] <= 1.0
        assert result["phase"] == WellbeingPhase.BURNOUT

    def test_negative_values_clamped(self):
        """Negative input should be handled (clamped to 0)."""
        # Burnout with negative (invalid) inputs
        theta = compute_burnout_theta(-1, -1, 7)  # Should clamp
        assert theta >= 0


class TestThetaInterpretationConsistency:
    """Tests for consistent theta interpretation across all measures."""

    def test_higher_stress_higher_theta(self):
        """All measures should give higher theta for more stress."""
        # Burnout: more exhaustion = higher theta
        theta_low_exhaustion = compute_burnout_theta(1, 1, 5)
        theta_high_exhaustion = compute_burnout_theta(5, 5, 1)
        assert theta_high_exhaustion > theta_low_exhaustion

        # Effort-Reward: more imbalance = higher theta
        theta_balanced = compute_effort_reward_theta(50, 50)
        theta_imbalanced = compute_effort_reward_theta(90, 30)
        assert theta_imbalanced > theta_balanced

        # Work-Family: more conflict = higher theta
        theta_no_conflict = compute_work_family_conflict_theta(0, 0)
        theta_conflict = compute_work_family_conflict_theta(4, 4)
        assert theta_conflict > theta_no_conflict

        # Cognitive Load: more load = higher theta
        theta_low_load = compute_cognitive_load_theta(20, 10)
        theta_high_load = compute_cognitive_load_theta(60, 30)
        assert theta_high_load > theta_low_load

        # JDR: more demands = higher theta
        theta_low_demands = compute_jdr_theta(20, 80)
        theta_high_demands = compute_jdr_theta(80, 20)
        assert theta_high_demands > theta_low_demands
