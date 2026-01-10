"""
Tests for Mechanical Systems Domain Module

Tests cover:
- Carnot, Otto, and Diesel efficiency
- Electric motor efficiency
- Battery electrochemistry
- Damped oscillators
- Theta range validation [0, 1]
"""

import pytest
import numpy as np

from theta_calculator.domains.mechanical_systems import (
    MechanicalSystem,
    SystemType,
    EfficiencyRegime,
    compute_mechanical_theta,
    compute_engine_theta,
    compute_motor_theta,
    compute_battery_theta,
    compute_damping_theta,
    carnot_efficiency,
    otto_efficiency,
    diesel_efficiency,
    critical_damping,
    nernst_potential,
    classify_efficiency,
    MECHANICAL_SYSTEMS,
)


class TestMechanicalSystemsExist:
    """Test that example mechanical systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """MECHANICAL_SYSTEMS dict should exist."""
        assert MECHANICAL_SYSTEMS is not None
        assert isinstance(MECHANICAL_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(MECHANICAL_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["car_engine", "ev_motor", "lithium_battery", "power_plant"]
        for name in expected:
            assert name in MECHANICAL_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in MECHANICAL_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "system_type")
            assert hasattr(system, "efficiency")
            assert hasattr(system, "theoretical_max")


class TestCarnotEfficiency:
    """Test Carnot efficiency calculation."""

    def test_typical_values(self):
        """Test typical temperature range."""
        eta = carnot_efficiency(600, 300)  # T_hot=600K, T_cold=300K
        assert eta == pytest.approx(0.5)

    def test_large_temperature_difference(self):
        """Large temperature difference gives high efficiency."""
        eta = carnot_efficiency(1000, 300)
        assert eta > 0.6

    def test_small_temperature_difference(self):
        """Small temperature difference gives low efficiency."""
        eta = carnot_efficiency(350, 300)
        assert eta < 0.2

    def test_efficiency_range(self):
        """Efficiency should be in (0, 1)."""
        eta = carnot_efficiency(500, 300)
        assert 0 < eta < 1

    def test_t_hot_less_than_t_cold_raises(self):
        """T_hot <= T_cold should raise ValueError."""
        with pytest.raises(ValueError):
            carnot_efficiency(300, 400)

    def test_equal_temperatures_raises(self):
        """Equal temperatures should raise ValueError."""
        with pytest.raises(ValueError):
            carnot_efficiency(300, 300)

    def test_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError):
            carnot_efficiency(-100, 50)


class TestOttoEfficiency:
    """Test Otto cycle efficiency."""

    def test_typical_compression_ratio(self):
        """Test typical gasoline engine compression ratio."""
        eta = otto_efficiency(10)  # r = 10 typical
        assert 0.5 < eta < 0.65

    def test_higher_ratio_higher_efficiency(self):
        """Higher compression ratio gives higher efficiency."""
        eta_low = otto_efficiency(8)
        eta_high = otto_efficiency(12)
        assert eta_high > eta_low

    def test_compression_ratio_one_raises(self):
        """Compression ratio <= 1 should raise ValueError."""
        with pytest.raises(ValueError):
            otto_efficiency(1)

    def test_gamma_effect(self):
        """Different gamma values affect efficiency."""
        eta_air = otto_efficiency(10, gamma=1.4)
        eta_mono = otto_efficiency(10, gamma=1.67)  # Monatomic
        assert eta_mono > eta_air


class TestDieselEfficiency:
    """Test Diesel cycle efficiency."""

    def test_typical_values(self):
        """Test typical diesel engine parameters."""
        eta = diesel_efficiency(compression_ratio=20, cutoff_ratio=2)
        assert 0.5 < eta < 0.7

    def test_higher_compression_higher_efficiency(self):
        """Higher compression ratio gives higher efficiency."""
        eta_low = diesel_efficiency(15, 2)
        eta_high = diesel_efficiency(25, 2)
        assert eta_high > eta_low

    def test_cutoff_ratio_effect(self):
        """Higher cutoff ratio reduces efficiency."""
        eta_low_cut = diesel_efficiency(20, 1.5)
        eta_high_cut = diesel_efficiency(20, 3.0)
        assert eta_low_cut > eta_high_cut

    def test_invalid_ratios_raise(self):
        """Ratios <= 1 should raise ValueError."""
        with pytest.raises(ValueError):
            diesel_efficiency(1, 2)
        with pytest.raises(ValueError):
            diesel_efficiency(20, 1)


class TestEngineTheta:
    """Test engine theta calculation."""

    def test_ideal_engine(self):
        """Perfect Carnot engine has theta = 1."""
        eta_carnot = carnot_efficiency(600, 300)
        theta = compute_engine_theta(eta_carnot, 600, 300)
        assert theta == pytest.approx(1.0)

    def test_half_carnot(self):
        """50% of Carnot gives theta = 0.5."""
        eta_carnot = carnot_efficiency(600, 300)
        theta = compute_engine_theta(eta_carnot * 0.5, 600, 300)
        assert theta == pytest.approx(0.5)

    def test_zero_efficiency(self):
        """Zero efficiency gives theta = 0."""
        theta = compute_engine_theta(0.0, 600, 300)
        assert theta == 0.0

    def test_typical_car_engine(self):
        """Real car engine has theta < 1."""
        car = MECHANICAL_SYSTEMS["car_engine"]
        theta = compute_engine_theta(
            car.efficiency,
            car.temperature_hot,
            car.temperature_cold
        )
        assert 0 < theta < 1


class TestMotorTheta:
    """Test electric motor theta calculation."""

    def test_perfect_motor(self):
        """98% efficiency motor at 98% max has theta = 1."""
        theta = compute_motor_theta(100, 98, max_efficiency=0.98)
        assert theta == 1.0

    def test_typical_motor(self):
        """95% efficiency motor has high theta."""
        theta = compute_motor_theta(100, 95, max_efficiency=0.98)
        assert theta > 0.9

    def test_poor_motor(self):
        """Low efficiency motor has low theta."""
        theta = compute_motor_theta(100, 50)
        assert theta < 0.6

    def test_zero_input_power(self):
        """Zero input power gives theta = 0."""
        theta = compute_motor_theta(0, 0)
        assert theta == 0.0


class TestBatteryTheta:
    """Test battery theta calculation."""

    def test_at_open_circuit(self):
        """At OCV (no load), theta = 1."""
        theta = compute_battery_theta(4.2, 4.2)
        assert theta == 1.0

    def test_under_load(self):
        """Under load, voltage drops, theta < 1."""
        theta = compute_battery_theta(3.8, 4.2)
        assert theta < 1.0
        assert theta > 0.9

    def test_with_internal_resistance(self):
        """Internal resistance reduces theta."""
        theta = compute_battery_theta(
            4.2, 4.2, internal_resistance=0.1, current=2
        )
        # V = 4.2 - 0.1*2 = 4.0
        assert theta == pytest.approx(4.0 / 4.2)

    def test_zero_ocv(self):
        """Zero OCV gives theta = 0."""
        theta = compute_battery_theta(3.0, 0.0)
        assert theta == 0.0


class TestCriticalDamping:
    """Test critical damping coefficient."""

    def test_formula(self):
        """Verify c_c = 2*sqrt(k*m)."""
        m = 1.0
        k = 100.0
        c_c = critical_damping(m, k)
        assert c_c == pytest.approx(2 * np.sqrt(k * m))

    def test_car_suspension(self):
        """Typical car suspension values."""
        m = 500  # kg (quarter car mass)
        k = 20000  # N/m
        c_c = critical_damping(m, k)
        assert c_c > 0


class TestDampingTheta:
    """Test damping ratio theta calculation."""

    def test_critically_damped(self):
        """Critical damping gives theta = 1."""
        m, k = 1.0, 100.0
        c_c = critical_damping(m, k)
        theta = compute_damping_theta(c_c, m, k)
        assert theta == pytest.approx(1.0)

    def test_underdamped(self):
        """Underdamped gives theta < 1."""
        m, k = 1.0, 100.0
        c_c = critical_damping(m, k)
        theta = compute_damping_theta(c_c * 0.5, m, k)
        assert theta == pytest.approx(0.5)

    def test_overdamped(self):
        """Overdamped gives theta > 1."""
        m, k = 1.0, 100.0
        c_c = critical_damping(m, k)
        theta = compute_damping_theta(c_c * 2, m, k)
        assert theta == pytest.approx(2.0)


class TestNernstPotential:
    """Test Nernst equation."""

    def test_at_standard_conditions(self):
        """At unit activity ratio, E = E0."""
        E = nernst_potential(1.0, 298, 2, 1.0)
        assert E == pytest.approx(1.0)

    def test_activity_effect(self):
        """Higher product activity reduces potential."""
        E_low = nernst_potential(1.0, 298, 2, 0.1)
        E_high = nernst_potential(1.0, 298, 2, 10.0)
        assert E_low > E_high


class TestUnifiedMechanicalTheta:
    """Test unified mechanical theta calculation."""

    def test_all_systems_valid_theta(self):
        """All mechanical systems should have theta in [0, 1]."""
        for name, system in MECHANICAL_SYSTEMS.items():
            theta = compute_mechanical_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_ev_motor_high_theta(self):
        """EV motor should have high theta (efficient)."""
        ev = MECHANICAL_SYSTEMS["ev_motor"]
        theta = compute_mechanical_theta(ev)
        assert theta > 0.9

    def test_car_engine_moderate_theta(self):
        """Car engine has moderate theta."""
        car = MECHANICAL_SYSTEMS["car_engine"]
        theta = compute_mechanical_theta(car)
        assert 0.3 < theta < 0.7

    def test_zero_theoretical_max(self):
        """Zero theoretical max gives theta = 0."""
        system = MechanicalSystem(
            name="Test",
            system_type=SystemType.HEAT_ENGINE,
            input_power=100,
            output_power=50,
            efficiency=0.5,
            theoretical_max=0.0
        )
        theta = compute_mechanical_theta(system)
        assert theta == 0.0


class TestClassifyEfficiency:
    """Test efficiency regime classification."""

    def test_wasteful(self):
        """Low theta -> WASTEFUL."""
        assert classify_efficiency(0.1) == EfficiencyRegime.WASTEFUL
        assert classify_efficiency(0.25) == EfficiencyRegime.WASTEFUL

    def test_typical(self):
        """Medium theta -> TYPICAL."""
        assert classify_efficiency(0.4) == EfficiencyRegime.TYPICAL
        assert classify_efficiency(0.55) == EfficiencyRegime.TYPICAL

    def test_efficient(self):
        """High theta -> EFFICIENT."""
        assert classify_efficiency(0.7) == EfficiencyRegime.EFFICIENT
        assert classify_efficiency(0.85) == EfficiencyRegime.EFFICIENT

    def test_near_ideal(self):
        """Very high theta -> NEAR_IDEAL."""
        assert classify_efficiency(0.92) == EfficiencyRegime.NEAR_IDEAL
        assert classify_efficiency(0.99) == EfficiencyRegime.NEAR_IDEAL


class TestEnums:
    """Test enum definitions."""

    def test_system_types(self):
        """All system types should be defined."""
        assert SystemType.HEAT_ENGINE.value == "heat_engine"
        assert SystemType.ELECTRIC_MOTOR.value == "electric_motor"
        assert SystemType.BATTERY.value == "battery"
        assert SystemType.OSCILLATOR.value == "oscillator"

    def test_efficiency_regimes(self):
        """All efficiency regimes should be defined."""
        assert EfficiencyRegime.WASTEFUL.value == "wasteful"
        assert EfficiencyRegime.TYPICAL.value == "typical"
        assert EfficiencyRegime.EFFICIENT.value == "efficient"
        assert EfficiencyRegime.NEAR_IDEAL.value == "near_ideal"


class TestMechanicalSystemDataclass:
    """Test MechanicalSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with required parameters."""
        system = MechanicalSystem(
            name="Test",
            system_type=SystemType.HEAT_ENGINE,
            input_power=1000,
            output_power=400,
            efficiency=0.4,
            theoretical_max=0.6
        )
        assert system.name == "Test"
        assert system.efficiency == 0.4

    def test_optional_temperatures(self):
        """Temperature fields are optional."""
        system = MechanicalSystem(
            name="Test",
            system_type=SystemType.HEAT_ENGINE,
            input_power=1000,
            output_power=400,
            efficiency=0.4,
            theoretical_max=0.6
        )
        assert system.temperature_hot is None
        assert system.temperature_cold is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_compression_otto(self):
        """Very high compression ratio approaches limit."""
        eta = otto_efficiency(50)
        assert eta < 1.0
        assert eta > 0.7

    def test_very_large_temperature_ratio(self):
        """Very large temperature ratio approaches 1."""
        eta = carnot_efficiency(10000, 300)
        assert eta > 0.95
        assert eta < 1.0

    def test_negative_efficiency(self):
        """Negative efficiency should be clipped."""
        system = MechanicalSystem(
            name="Invalid",
            system_type=SystemType.HEAT_ENGINE,
            input_power=100,
            output_power=-50,
            efficiency=-0.5,
            theoretical_max=0.5
        )
        theta = compute_mechanical_theta(system)
        assert theta == 0.0
