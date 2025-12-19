"""
Theta Landscape Visualization: Map theta across parameter space.

This module generates visualizations showing how theta varies across
physical parameters like mass, length, temperature. These landscapes
reveal:
- Where the quantum-classical boundary lies
- How different systems compare
- Phase transition regions where theta changes rapidly
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Callable

from ..core.theta_state import PhysicalSystem
from ..core.interpolation import ThetaCalculator, estimate_theta_quick
from ..constants.values import FundamentalConstants as FC
from ..constants.planck_units import PlanckUnits


@dataclass
class LandscapeConfig:
    """Configuration for theta landscape plots."""
    resolution: int = 100
    colormap: str = "viridis"
    title: str = "Theta Landscape"
    xlabel: str = "log10(Parameter 1)"
    ylabel: str = "log10(Parameter 2)"
    show_colorbar: bool = True
    show_contours: bool = True
    contour_levels: int = 10
    figsize: Tuple[int, int] = (12, 8)


class ThetaLandscapePlotter:
    """
    Generates 2D and 3D visualizations of theta across parameter space.

    These plots demonstrate:
    1. Where theta = 0 (classical regime, shown as dark)
    2. Where theta = 1 (quantum regime, shown as bright)
    3. The gradient/transition zone between them
    4. Phase boundaries and notable physical systems

    Use this to visualize the quantum-classical boundary and
    understand where different systems lie on the theta spectrum.
    """

    def __init__(self, config: Optional[LandscapeConfig] = None):
        self.config = config or LandscapeConfig()
        self.calculator = ThetaCalculator()

    def compute_theta_grid(
        self,
        theta_function: Callable[[float, float], float],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute theta values on a 2D grid.

        Args:
            theta_function: Function(x, y) -> theta
            x_range: (min, max) for x parameter
            y_range: (min, max) for y parameter
            resolution: Grid resolution (default from config)

        Returns:
            (X, Y, Theta) meshgrid arrays
        """
        res = resolution or self.config.resolution

        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)

        Theta = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                try:
                    Theta[i, j] = theta_function(X[i, j], Y[i, j])
                except Exception:
                    Theta[i, j] = np.nan

        return X, Y, Theta

    def mass_length_landscape(
        self,
        log_mass_range: Tuple[float, float] = (-35, 35),
        log_length_range: Tuple[float, float] = (-40, 15),
        temperature: float = 300.0
    ) -> Dict:
        """
        Generate theta landscape for mass vs length scale.

        This is the primary visualization showing the quantum-classical
        boundary across all scales.

        Args:
            log_mass_range: (min, max) log10(mass/kg)
            log_length_range: (min, max) log10(length/m)
            temperature: Fixed temperature in Kelvin

        Returns:
            Dict with X, Y, Theta arrays and metadata
        """
        def theta_at_point(log_m: float, log_L: float) -> float:
            mass = 10**log_m
            length = 10**log_L
            return estimate_theta_quick(mass, length, temperature)

        X, Y, Theta = self.compute_theta_grid(
            theta_at_point,
            log_mass_range,
            log_length_range
        )

        return {
            "X": X,
            "Y": Y,
            "Theta": Theta,
            "xlabel": "log10(mass / kg)",
            "ylabel": "log10(length / m)",
            "title": f"Theta Landscape (T = {temperature} K)",
            "metadata": {
                "temperature": temperature,
                "x_param": "mass",
                "y_param": "length",
            }
        }

    def temperature_mass_landscape(
        self,
        log_mass_range: Tuple[float, float] = (-35, 5),
        log_temp_range: Tuple[float, float] = (-3, 10),
        length: float = 1e-10
    ) -> Dict:
        """
        Generate theta landscape for temperature vs mass.

        Shows how thermal effects influence the quantum-classical boundary.

        Args:
            log_mass_range: (min, max) log10(mass/kg)
            log_temp_range: (min, max) log10(temperature/K)
            length: Fixed length scale in meters

        Returns:
            Dict with landscape data
        """
        def theta_at_point(log_m: float, log_T: float) -> float:
            mass = 10**log_m
            temp = 10**log_T
            return estimate_theta_quick(mass, length, temp)

        X, Y, Theta = self.compute_theta_grid(
            theta_at_point,
            log_mass_range,
            log_temp_range
        )

        return {
            "X": X,
            "Y": Y,
            "Theta": Theta,
            "xlabel": "log10(mass / kg)",
            "ylabel": "log10(temperature / K)",
            "title": f"Theta Landscape (L = {length:.2e} m)",
            "metadata": {
                "length": length,
                "x_param": "mass",
                "y_param": "temperature",
            }
        }

    def notable_systems(self) -> List[Dict]:
        """
        Return coordinates of notable physical systems for annotation.

        These can be plotted on landscapes to show where familiar
        objects lie on the theta spectrum.
        """
        l_P = PlanckUnits.planck_length()
        m_P = PlanckUnits.planck_mass()

        return [
            {"name": "Planck scale", "log_mass": np.log10(m_P), "log_length": np.log10(l_P), "theta": 1.0},
            {"name": "Electron", "log_mass": np.log10(9.109e-31), "log_length": np.log10(2.8e-15), "theta": 0.99},
            {"name": "Proton", "log_mass": np.log10(1.673e-27), "log_length": np.log10(8.8e-16), "theta": 0.98},
            {"name": "Hydrogen atom", "log_mass": np.log10(1.67e-27), "log_length": np.log10(5.3e-11), "theta": 0.7},
            {"name": "DNA molecule", "log_mass": np.log10(1e-20), "log_length": np.log10(2e-9), "theta": 0.3},
            {"name": "Virus", "log_mass": np.log10(1e-18), "log_length": np.log10(1e-7), "theta": 0.05},
            {"name": "Bacterium", "log_mass": np.log10(1e-12), "log_length": np.log10(1e-6), "theta": 0.001},
            {"name": "Human cell", "log_mass": np.log10(1e-9), "log_length": np.log10(1e-5), "theta": 1e-6},
            {"name": "Grain of sand", "log_mass": np.log10(1e-6), "log_length": np.log10(1e-4), "theta": 1e-15},
            {"name": "Baseball", "log_mass": np.log10(0.145), "log_length": np.log10(0.074), "theta": 1e-35},
            {"name": "Human", "log_mass": np.log10(70), "log_length": np.log10(1.7), "theta": 1e-40},
            {"name": "Earth", "log_mass": np.log10(5.97e24), "log_length": np.log10(6.37e6), "theta": 0},
            {"name": "Sun", "log_mass": np.log10(1.99e30), "log_length": np.log10(6.96e8), "theta": 0},
        ]

    def find_phase_boundary(
        self,
        landscape: Dict,
        theta_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Find the phase boundary where theta crosses a threshold.

        This locates the quantum-classical transition line.

        Args:
            landscape: Landscape dict from mass_length_landscape etc.
            theta_threshold: Theta value defining the boundary

        Returns:
            Boolean array marking boundary points
        """
        Theta = landscape["Theta"]
        boundary = np.abs(Theta - theta_threshold) < 0.05
        return boundary

    def to_matplotlib_data(self, landscape: Dict) -> str:
        """
        Generate matplotlib code to plot the landscape.

        Returns Python code as a string that can be executed.
        """
        code = f'''
import numpy as np
import matplotlib.pyplot as plt

# Landscape data (copy these arrays)
X = np.array({landscape["X"].tolist()})
Y = np.array({landscape["Y"].tolist()})
Theta = np.array({landscape["Theta"].tolist()})

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot theta landscape
im = ax.pcolormesh(X, Y, Theta, cmap='viridis', shading='auto')

# Add contours
contours = ax.contour(X, Y, Theta, levels=[0.01, 0.1, 0.5, 0.9, 0.99],
                       colors='white', alpha=0.5)
ax.clabel(contours, inline=True, fontsize=8, fmt='θ=%.2f')

# Labels
ax.set_xlabel('{landscape["xlabel"]}')
ax.set_ylabel('{landscape["ylabel"]}')
ax.set_title('{landscape["title"]}')

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='θ (theta)')

plt.tight_layout()
plt.savefig('theta_landscape.png', dpi=150)
plt.show()
'''
        return code

    def landscape_summary(self, landscape: Dict) -> str:
        """
        Generate text summary of a landscape.
        """
        Theta = landscape["Theta"]
        quantum_fraction = np.sum(Theta > 0.9) / Theta.size
        classical_fraction = np.sum(Theta < 0.1) / Theta.size
        transition_fraction = 1 - quantum_fraction - classical_fraction

        return f"""
THETA LANDSCAPE SUMMARY
=======================
{landscape['title']}

Region statistics:
  Quantum (θ > 0.9):     {quantum_fraction*100:.1f}%
  Classical (θ < 0.1):   {classical_fraction*100:.1f}%
  Transition:            {transition_fraction*100:.1f}%

Theta range: [{np.nanmin(Theta):.4f}, {np.nanmax(Theta):.4f}]
Mean theta: {np.nanmean(Theta):.4f}

The landscape shows where quantum mechanics (bright) and
classical physics (dark) are appropriate descriptions.
"""


def quick_landscape() -> Dict:
    """Generate a quick mass-length theta landscape at room temperature."""
    plotter = ThetaLandscapePlotter()
    return plotter.mass_length_landscape()
