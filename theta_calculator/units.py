"""
Unit Handling and Input Validation for Theta Calculator.

This module provides:
1. Unit registry via pint for dimensional analysis
2. Validation functions for physical inputs
3. Warning system for impossible combinations

Usage:
    from theta_calculator.units import Q_, validate_physical_inputs, parse_with_units

    # Using pint quantities
    mass = Q_(9.1e-31, 'kg')
    length = Q_(2.8e-15, 'm')

    # Parsing user input with units
    mass = parse_with_units('1 kg', 'mass')
    length = parse_with_units('1e-10 m', 'length')

    # Validating inputs
    is_valid = validate_physical_inputs(mass=1.0, length=1e-10, temperature=300)
"""

import warnings
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import pint
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    ureg = None
    Q_ = None


class ValidationWarning(UserWarning):
    """Warning for unphysical input values."""
    pass


class ValidationError(ValueError):
    """Error for invalid input values."""
    pass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    warnings: list
    errors: list

    def __bool__(self):
        return self.is_valid


# Physical limits (SI units)
PLANCK_MASS = 2.176434e-8  # kg
PLANCK_LENGTH = 1.616255e-35  # m
PLANCK_TIME = 5.391247e-44  # s
PLANCK_TEMPERATURE = 1.416784e32  # K
SPEED_OF_LIGHT = 299792458  # m/s


def validate_physical_inputs(
    mass: Optional[float] = None,
    length: Optional[float] = None,
    energy: Optional[float] = None,
    temperature: Optional[float] = None,
    time: Optional[float] = None,
    strict: bool = False
) -> ValidationResult:
    """
    Validate physical inputs and emit warnings for impossible combinations.

    Args:
        mass: Mass in kg
        length: Length in m
        energy: Energy in J
        temperature: Temperature in K
        time: Time in s
        strict: If True, raise errors instead of warnings

    Returns:
        ValidationResult with is_valid flag and list of warnings/errors

    Examples:
        >>> result = validate_physical_inputs(mass=-1.0)
        >>> result.is_valid
        False
        >>> 'Negative mass' in result.warnings[0]
        True
    """
    warnings_list = []
    errors_list = []

    # Mass validation
    if mass is not None:
        if mass < 0:
            msg = f"Negative mass ({mass} kg) is unphysical"
            errors_list.append(msg)
        elif mass == 0:
            msg = "Zero mass: treating as massless particle"
            warnings_list.append(msg)
        elif mass > 1e53:  # Observable universe mass ~1e53 kg
            msg = f"Mass ({mass:.2e} kg) exceeds observable universe mass"
            warnings_list.append(msg)

    # Length validation
    if length is not None:
        if length <= 0:
            msg = f"Length ({length} m) must be positive"
            errors_list.append(msg)
        elif length < PLANCK_LENGTH:
            msg = f"Length ({length:.2e} m) below Planck length ({PLANCK_LENGTH:.2e} m)"
            warnings_list.append(msg)
        elif length > 8.8e26:  # Observable universe radius ~4.4e26 m
            msg = f"Length ({length:.2e} m) exceeds observable universe scale"
            warnings_list.append(msg)

    # Temperature validation
    if temperature is not None:
        if temperature < 0:
            msg = f"Temperature ({temperature} K) below absolute zero is unphysical"
            errors_list.append(msg)
        elif temperature == 0:
            msg = "Temperature at absolute zero: quantum ground state"
            warnings_list.append(msg)
        elif temperature > PLANCK_TEMPERATURE:
            msg = f"Temperature ({temperature:.2e} K) exceeds Planck temperature ({PLANCK_TEMPERATURE:.2e} K)"
            warnings_list.append(msg)

    # Energy validation
    if energy is not None:
        if energy < 0:
            msg = f"Negative energy ({energy} J): ensure this is intentional (binding energy?)"
            warnings_list.append(msg)

    # Time validation
    if time is not None:
        if time < 0:
            msg = f"Negative time ({time} s) may indicate time-reversal context"
            warnings_list.append(msg)
        elif time > 0 and time < PLANCK_TIME:
            msg = f"Time ({time:.2e} s) below Planck time ({PLANCK_TIME:.2e} s)"
            warnings_list.append(msg)

    # Cross-parameter consistency checks
    if mass is not None and length is not None and mass > 0 and length > 0:
        # Schwarzschild radius check
        G = 6.67430e-11  # m^3/(kg*s^2)
        schwarzschild_radius = 2 * G * mass / (SPEED_OF_LIGHT ** 2)
        if length < schwarzschild_radius:
            msg = (f"Length ({length:.2e} m) smaller than Schwarzschild radius "
                   f"({schwarzschild_radius:.2e} m): system would be a black hole")
            warnings_list.append(msg)

    # Emit warnings
    for w in warnings_list:
        warnings.warn(w, ValidationWarning)

    # Handle errors
    is_valid = len(errors_list) == 0
    if strict and errors_list:
        raise ValidationError("; ".join(errors_list))

    return ValidationResult(
        is_valid=is_valid,
        warnings=warnings_list,
        errors=errors_list
    )


def parse_with_units(
    value_str: str,
    expected_dimension: str = None
) -> Tuple[float, str]:
    """
    Parse a string value with optional units using pint.

    Args:
        value_str: String like "1.5 kg" or "300 K" or "1e-10"
        expected_dimension: Expected dimension ('mass', 'length', 'temperature', 'energy', 'time')

    Returns:
        Tuple of (magnitude in SI units, unit string)

    Examples:
        >>> parse_with_units("1.5 kg", "mass")
        (1.5, 'kg')
        >>> parse_with_units("300", "temperature")
        (300.0, 'K')  # Assumes K for temperature
        >>> parse_with_units("1 MeV", "energy")
        (1.602176634e-13, 'J')  # Converted to Joules
    """
    if not PINT_AVAILABLE:
        # Fallback: try to parse as float
        try:
            parts = value_str.strip().split()
            if len(parts) == 1:
                return float(parts[0]), _default_unit(expected_dimension)
            else:
                return float(parts[0]), parts[1]
        except ValueError:
            raise ValueError(f"Could not parse '{value_str}' (pint not available)")

    try:
        # Try to parse with pint
        quantity = ureg.parse_expression(value_str)

        # If it's dimensionless, add default units based on expected dimension
        if not hasattr(quantity, 'units') or quantity.dimensionless:
            default_unit = _default_unit(expected_dimension)
            if default_unit:
                quantity = float(quantity) * ureg(default_unit)
            else:
                return float(quantity), ''

        # Convert to SI base units
        si_quantity = quantity.to_base_units()

        return float(si_quantity.magnitude), str(si_quantity.units)

    except Exception as e:
        # Fallback: try to parse as plain float
        try:
            value = float(value_str.strip())
            return value, _default_unit(expected_dimension)
        except ValueError:
            raise ValueError(f"Could not parse '{value_str}': {e}")


def _default_unit(dimension: Optional[str]) -> str:
    """Return default SI unit for a dimension."""
    defaults = {
        'mass': 'kg',
        'length': 'm',
        'temperature': 'K',
        'energy': 'J',
        'time': 's',
        'velocity': 'm/s',
        'momentum': 'kg*m/s',
        'action': 'J*s',
    }
    return defaults.get(dimension, '')


def convert_to_si(value: float, from_unit: str, dimension: str) -> float:
    """
    Convert a value from arbitrary units to SI.

    Args:
        value: Numeric value
        from_unit: Source unit string (e.g., 'eV', 'nm', 'MeV')
        dimension: Dimension for context ('energy', 'length', etc.)

    Returns:
        Value in SI units

    Examples:
        >>> convert_to_si(1.0, 'eV', 'energy')
        1.602176634e-19
        >>> convert_to_si(1.0, 'nm', 'length')
        1e-09
    """
    if not PINT_AVAILABLE:
        # Manual conversion factors for common units
        conversions = {
            ('eV', 'energy'): 1.602176634e-19,
            ('MeV', 'energy'): 1.602176634e-13,
            ('GeV', 'energy'): 1.602176634e-10,
            ('nm', 'length'): 1e-9,
            ('um', 'length'): 1e-6,
            ('mm', 'length'): 1e-3,
            ('cm', 'length'): 1e-2,
            ('fm', 'length'): 1e-15,
            ('C', 'temperature'): None,  # Special case
            ('F', 'temperature'): None,  # Special case
        }
        key = (from_unit, dimension)
        if key in conversions:
            factor = conversions[key]
            if factor is not None:
                return value * factor
            # Temperature conversions
            if from_unit == 'C':
                return value + 273.15
            if from_unit == 'F':
                return (value - 32) * 5/9 + 273.15
        return value  # Assume already SI

    try:
        quantity = value * ureg(from_unit)
        si_quantity = quantity.to_base_units()
        return float(si_quantity.magnitude)
    except Exception:
        return value  # Return unchanged if conversion fails


# Export for convenience
__all__ = [
    'Q_',
    'ureg',
    'PINT_AVAILABLE',
    'validate_physical_inputs',
    'parse_with_units',
    'convert_to_si',
    'ValidationResult',
    'ValidationWarning',
    'ValidationError',
    'PLANCK_MASS',
    'PLANCK_LENGTH',
    'PLANCK_TIME',
    'PLANCK_TEMPERATURE',
]
