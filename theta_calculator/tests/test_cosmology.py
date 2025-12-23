"""
Tests for cosmology module.

Tests cosmic theta evolution from Planck era to heat death,
including phase transitions and temperature scaling.
"""

import pytest

from theta_calculator.domains.cosmology import (
    COSMIC_TIMELINE,
    PHASE_TRANSITIONS,
    CosmicEpoch,
    CosmicEra,
    compute_cosmic_theta,
    compute_thermal_theta,
    temperature_to_energy_ev,
    energy_to_temperature,
    theta_evolution,
    orders_of_magnitude_summary,
    T_PLANCK,
    E_PLANCK_EV,
)


class TestCosmicTimeline:
    """Test the cosmic timeline definition."""

    def test_timeline_not_empty(self):
        """Timeline should have entries."""
        assert len(COSMIC_TIMELINE) >= 10, "Timeline should have many epochs"

    def test_key_epochs_exist(self):
        """Key cosmic epochs should be defined."""
        expected = [
            "planck_era", "electroweak_era", "nucleosynthesis",
            "recombination", "present_day"
        ]
        for name in expected:
            assert name in COSMIC_TIMELINE, f"Missing epoch: {name}"

    def test_epochs_have_required_attributes(self):
        """All epochs should have required attributes."""
        for name, epoch in COSMIC_TIMELINE.items():
            assert isinstance(epoch, CosmicEpoch)
            assert epoch.name
            assert isinstance(epoch.era, CosmicEra)
            assert epoch.time > 0
            assert epoch.temperature > 0
            assert epoch.energy > 0
            assert epoch.description

    def test_timeline_chronological(self):
        """Epochs should be in chronological order when sorted by time."""
        times = [(name, epoch.time) for name, epoch in COSMIC_TIMELINE.items()]
        times_sorted = sorted(times, key=lambda x: x[1])
        # First should be Planck era
        assert "planck" in times_sorted[0][0].lower()


class TestThetaComputation:
    """Test theta computation for cosmic epochs."""

    def test_all_epochs_theta_in_range(self):
        """All epochs should produce theta in [0, 1]."""
        for name, epoch in COSMIC_TIMELINE.items():
            theta = compute_cosmic_theta(epoch)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_planck_era_theta_high(self):
        """Planck era should have theta near 1."""
        planck = COSMIC_TIMELINE["planck_era"]
        theta = compute_cosmic_theta(planck)
        assert theta > 0.9, f"Planck era theta should be near 1: {theta}"

    def test_present_day_theta_low(self):
        """Present day should have theta near 0."""
        present = COSMIC_TIMELINE["present_day"]
        theta = compute_cosmic_theta(present)
        assert theta < 1e-20, f"Present day theta should be tiny: {theta}"

    def test_theta_decreases_with_time(self):
        """Theta should generally decrease as universe ages."""
        evolution = theta_evolution()
        # Check overall trend: first theta >> last theta
        first_theta = evolution[0][1]
        last_theta = evolution[-1][1]
        assert first_theta > last_theta * 1e30, \
            "Theta should decrease from Planck era to heat death"

    def test_theta_spans_many_orders_of_magnitude(self):
        """Theta should span at least 30 orders of magnitude."""
        summary = orders_of_magnitude_summary()
        assert summary["log_range"] > 25, \
            f"Theta range too small: {summary['log_range']}"


class TestThermalTheta:
    """Test thermal theta calculation."""

    def test_planck_temperature_gives_theta_one(self):
        """At Planck temperature, theta should be 1."""
        theta = compute_thermal_theta(T_PLANCK)
        assert abs(theta - 1.0) < 0.01, f"Planck temp theta should be 1: {theta}"

    def test_cmb_temperature_gives_low_theta(self):
        """At CMB temperature (2.7K), theta should be tiny."""
        theta = compute_thermal_theta(2.725)
        assert theta < 1e-30, f"CMB temp theta should be tiny: {theta}"

    def test_thermal_theta_scales_linearly(self):
        """Thermal theta should scale linearly with temperature."""
        theta1 = compute_thermal_theta(1e15)
        theta2 = compute_thermal_theta(1e16)
        ratio = theta2 / theta1
        assert 9 < ratio < 11, f"Scaling ratio should be ~10: {ratio}"


class TestPhaseTransitions:
    """Test cosmological phase transitions."""

    def test_transitions_exist(self):
        """Phase transitions should be defined."""
        assert len(PHASE_TRANSITIONS) >= 3

    def test_electroweak_temperature(self):
        """Electroweak transition at ~10^12 K."""
        ew = PHASE_TRANSITIONS["electroweak_transition"]
        assert 1e11 < ew.temperature < 1e13, \
            f"Electroweak temp unexpected: {ew.temperature}"

    def test_qcd_temperature(self):
        """QCD transition at ~10^12 K."""
        qcd = PHASE_TRANSITIONS["qcd_transition"]
        assert 1e11 < qcd.temperature < 1e13, \
            f"QCD temp unexpected: {qcd.temperature}"


class TestEnergyTemperatureConversion:
    """Test energy-temperature conversion functions."""

    def test_room_temperature_energy(self):
        """Room temperature ~300K should be ~0.026 eV."""
        E = temperature_to_energy_ev(300)
        assert 0.02 < E < 0.03, f"Room temp energy unexpected: {E}"

    def test_planck_energy_conversion(self):
        """Planck temperature should give Planck energy."""
        E = temperature_to_energy_ev(T_PLANCK)
        ratio = E / E_PLANCK_EV
        assert 0.9 < ratio < 1.1, f"Planck energy ratio: {ratio}"

    def test_round_trip_conversion(self):
        """Energy -> Temp -> Energy should be identity."""
        E_original = 1e6  # 1 MeV
        T = energy_to_temperature(E_original)
        E_back = temperature_to_energy_ev(T)
        ratio = E_back / E_original
        assert 0.99 < ratio < 1.01, f"Round trip failed: {ratio}"


class TestCosmicEras:
    """Test cosmic era classification."""

    def test_planck_era_correct(self):
        """Planck era should be classified correctly."""
        planck = COSMIC_TIMELINE["planck_era"]
        assert planck.era == CosmicEra.PLANCK

    def test_nucleosynthesis_era_correct(self):
        """Nucleosynthesis should be in nucleosynthesis era."""
        bbn = COSMIC_TIMELINE["nucleosynthesis"]
        assert bbn.era == CosmicEra.NUCLEOSYNTHESIS

    def test_present_day_era(self):
        """Present day should be in dark energy era."""
        present = COSMIC_TIMELINE["present_day"]
        assert present.era == CosmicEra.DARK_ENERGY

    def test_heat_death_era(self):
        """Heat death should be in far future era."""
        heat_death = COSMIC_TIMELINE["heat_death"]
        assert heat_death.era == CosmicEra.FAR_FUTURE


class TestThetaEvolution:
    """Test the theta evolution function."""

    def test_evolution_returns_list(self):
        """theta_evolution should return list of tuples."""
        evolution = theta_evolution()
        assert isinstance(evolution, list)
        assert len(evolution) > 0
        assert len(evolution[0]) == 3  # (time, theta, name)

    def test_evolution_sorted_by_time(self):
        """Evolution should be sorted by time."""
        evolution = theta_evolution()
        times = [e[0] for e in evolution]
        assert times == sorted(times), "Evolution not sorted by time"

    def test_evolution_starts_at_planck(self):
        """Evolution should start at Planck time."""
        evolution = theta_evolution()
        first_time = evolution[0][0]
        assert first_time < 1e-40, f"First time too late: {first_time}"

    def test_evolution_covers_present(self):
        """Evolution should include present day."""
        evolution = theta_evolution()
        names = [e[2] for e in evolution]
        assert any("present" in n.lower() for n in names), \
            "Present day not in evolution"


class TestOrdersOfMagnitude:
    """Test the summary statistics."""

    def test_summary_has_required_keys(self):
        """Summary should have expected keys."""
        summary = orders_of_magnitude_summary()
        assert "max_theta" in summary
        assert "min_theta" in summary
        assert "log_range" in summary
        assert "time_range_orders" in summary

    def test_max_theta_near_one(self):
        """Maximum theta should be near 1 (Planck era)."""
        summary = orders_of_magnitude_summary()
        assert summary["max_theta"] > 0.9

    def test_time_range_large(self):
        """Time range should span many orders of magnitude."""
        summary = orders_of_magnitude_summary()
        assert summary["time_range_orders"] > 100, \
            f"Time range too small: {summary['time_range_orders']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
