"""Tests for fundamental constants and Planck units."""

import numpy as np
import pytest

from theta_calculator.constants.values import (
    FundamentalConstants,
    verify_relationships,
    c, h, h_bar, G, k_B, alpha, e, epsilon_0, mu_0
)
from theta_calculator.constants.planck_units import PlanckUnits


class TestFundamentalConstants:
    """Test fundamental constant values and relationships."""

    def test_speed_of_light_exact(self):
        """c should be exactly 299792458 m/s (by definition)."""
        assert FundamentalConstants.c.value == 299792458.0
        assert FundamentalConstants.c.uncertainty == 0.0

    def test_planck_constant_exact(self):
        """h should be exactly 6.62607015e-34 J·s (by 2019 SI)."""
        assert FundamentalConstants.h.value == 6.62607015e-34
        assert FundamentalConstants.h.uncertainty == 0.0

    def test_h_bar_relationship(self):
        """ℏ should equal h/(2π)."""
        h_bar_computed = h / (2 * np.pi)
        assert np.isclose(h_bar, h_bar_computed, rtol=1e-10)

    def test_boltzmann_constant_exact(self):
        """k should be exactly 1.380649e-23 J/K (by 2019 SI)."""
        assert FundamentalConstants.k.value == 1.380649e-23
        assert FundamentalConstants.k.uncertainty == 0.0

    def test_fine_structure_constant_value(self):
        """α should be approximately 1/137.036."""
        assert np.isclose(alpha, 1/137.036, rtol=1e-4)

    def test_alpha_from_other_constants(self):
        """α = e²/(4πε₀ℏc) should be consistent."""
        alpha_computed = e**2 / (4 * np.pi * epsilon_0 * h_bar * c)
        assert np.isclose(alpha, alpha_computed, rtol=1e-6)

    def test_c_from_electromagnetic(self):
        """c = 1/√(ε₀μ₀) should be consistent."""
        c_computed = 1.0 / np.sqrt(epsilon_0 * mu_0)
        assert np.isclose(c, c_computed, rtol=1e-9)

    def test_verify_relationships(self):
        """All internal relationships should verify."""
        results = verify_relationships()
        for name, passed in results.items():
            assert passed, f"Relationship {name} failed"

    def test_get_all_returns_dict(self):
        """get_all() should return a dictionary of constants."""
        all_constants = FundamentalConstants.get_all()
        assert isinstance(all_constants, dict)
        assert 'c' in all_constants
        assert 'h_bar' in all_constants
        assert 'G' in all_constants

    def test_get_exact_only_exact(self):
        """get_exact() should only return constants with zero uncertainty."""
        exact = FundamentalConstants.get_exact()
        for name, const in exact.items():
            assert const.uncertainty == 0.0, f"{name} has non-zero uncertainty"


class TestPlanckUnits:
    """Test Planck unit calculations."""

    def test_planck_length_value(self):
        """l_P should be approximately 1.616e-35 m."""
        l_P = PlanckUnits.planck_length()
        assert np.isclose(l_P, 1.616e-35, rtol=1e-2)

    def test_planck_time_value(self):
        """t_P should be approximately 5.391e-44 s."""
        t_P = PlanckUnits.planck_time()
        assert np.isclose(t_P, 5.391e-44, rtol=1e-2)

    def test_planck_mass_value(self):
        """m_P should be approximately 2.176e-8 kg."""
        m_P = PlanckUnits.planck_mass()
        assert np.isclose(m_P, 2.176e-8, rtol=1e-2)

    def test_planck_energy_value(self):
        """E_P should be approximately 1.956e9 J."""
        E_P = PlanckUnits.planck_energy()
        assert np.isclose(E_P, 1.956e9, rtol=1e-2)

    def test_planck_temperature_value(self):
        """T_P should be approximately 1.417e32 K."""
        T_P = PlanckUnits.planck_temperature()
        assert np.isclose(T_P, 1.417e32, rtol=1e-2)

    def test_planck_length_time_relationship(self):
        """l_P / t_P should equal c."""
        l_P = PlanckUnits.planck_length()
        t_P = PlanckUnits.planck_time()
        assert np.isclose(l_P / t_P, c, rtol=1e-10)

    def test_planck_energy_mass_relationship(self):
        """E_P should equal m_P × c²."""
        m_P = PlanckUnits.planck_mass()
        E_P = PlanckUnits.planck_energy()
        assert np.isclose(E_P, m_P * c**2, rtol=1e-10)

    def test_planck_temperature_energy_relationship(self):
        """T_P should equal E_P / k."""
        E_P = PlanckUnits.planck_energy()
        T_P = PlanckUnits.planck_temperature()
        assert np.isclose(T_P, E_P / k_B, rtol=1e-10)

    def test_verify_planck_relationships(self):
        """All Planck unit relationships should verify."""
        results = PlanckUnits.verify_relationships()
        for name, passed in results.items():
            assert passed, f"Relationship {name} failed"

    def test_in_planck_units(self):
        """Converting to Planck units should work correctly."""
        # 1 meter in Planck lengths
        l_P = PlanckUnits.planck_length()
        one_meter_in_planck = PlanckUnits.in_planck_units(1.0, 'length')
        assert np.isclose(one_meter_in_planck, 1.0 / l_P, rtol=1e-10)

    def test_from_planck_units(self):
        """Converting from Planck units should work correctly."""
        # 1 Planck length in meters
        l_P = PlanckUnits.planck_length()
        one_planck_in_meters = PlanckUnits.from_planck_units(1.0, 'length')
        assert np.isclose(one_planck_in_meters, l_P, rtol=1e-10)

    def test_get_all_returns_dict(self):
        """get_all() should return all Planck units."""
        all_units = PlanckUnits.get_all()
        assert isinstance(all_units, dict)
        assert 'l_P' in all_units
        assert 't_P' in all_units
        assert 'm_P' in all_units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
