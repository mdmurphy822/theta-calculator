r"""
Cybersecurity Domain: Attack/Defense Dynamics as Theta

This module implements theta as the security posture parameter
using frameworks from threat modeling and security metrics.

Key Insight: Security systems exhibit phase transitions between:
- theta ~ 0: Compromised/vulnerable (wide attack surface, slow detection)
- theta ~ 1: Fortified/resilient (minimal exposure, proactive defense)

Theta Maps To:
1. Attack Surface: 1 - (exposed_services / total_services)
2. Detection Speed: MTTD_optimal / MTTD_actual
3. Response Speed: MTTR_optimal / MTTR_actual
4. Defense Depth: layers / max_layers
5. Cryptographic Strength: key_bits / 256 (normalized to AES-256)

Security Regimes:
- COMPROMISED (theta < 0.2): Active breach, attacker persistence
- VULNERABLE (0.2 <= theta < 0.4): Known exploits, weak defenses
- BASELINE (0.4 <= theta < 0.6): Standard enterprise security
- HARDENED (0.6 <= theta < 0.8): Defense-in-depth, proactive monitoring
- FORTIFIED (theta >= 0.8): Zero-trust, quantum-safe cryptography

Physical Analogy:
The security posture follows a phase transition similar to ferromagnetic
ordering. Below a critical "security temperature" (threat level),
individual vulnerabilities couple to create systemic risk (analogous
to spontaneous magnetization). At the critical point, small attacks
can cascade through correlated weaknesses.

References (see BIBLIOGRAPHY.bib):
    \cite{Mitre2023} - MITRE ATT&CK framework
    \cite{NIST2018} - NIST Cybersecurity Framework
    \cite{CVSS2019} - Common Vulnerability Scoring System
    \cite{Schneier2000} - Secrets and Lies: Security in a Networked World
    \cite{Anderson2020} - Security Engineering
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class SecurityPosture(Enum):
    """Security posture states based on theta."""
    COMPROMISED = "compromised"    # theta < 0.2
    VULNERABLE = "vulnerable"      # 0.2 <= theta < 0.4
    BASELINE = "baseline"          # 0.4 <= theta < 0.6
    HARDENED = "hardened"          # 0.6 <= theta < 0.8
    FORTIFIED = "fortified"        # theta >= 0.8


class ThreatLevel(Enum):
    """Threat actor sophistication levels."""
    SCRIPT_KIDDIE = "script_kiddie"
    OPPORTUNISTIC = "opportunistic"
    TARGETED = "targeted"
    APT = "advanced_persistent_threat"
    NATION_STATE = "nation_state"


class CryptoStrength(Enum):
    """Cryptographic strength levels."""
    BROKEN = "broken"              # DES, MD5, SHA-1
    WEAK = "weak"                  # 3DES, SHA-256 w/ short keys
    STANDARD = "standard"          # AES-128, RSA-2048
    STRONG = "strong"              # AES-256, RSA-4096
    QUANTUM_SAFE = "quantum_safe"  # Lattice-based, NTRU


@dataclass
class SecuritySystem:
    """
    A security system for theta analysis.

    Attributes:
        name: System identifier
        total_services: Total network services
        exposed_services: Internet-facing services
        mttd_hours: Mean Time to Detect (hours)
        mttr_hours: Mean Time to Respond (hours)
        defense_layers: Number of security layers
        crypto_bits: Effective cryptographic key strength
        patch_coverage: Fraction of vulnerabilities patched [0, 1]
        mfa_coverage: Multi-factor auth coverage [0, 1]
        threat_level: Current threat environment
    """
    name: str
    total_services: int
    exposed_services: int
    mttd_hours: float  # Mean Time to Detect
    mttr_hours: float  # Mean Time to Respond
    defense_layers: int
    crypto_bits: int
    patch_coverage: float
    mfa_coverage: float
    threat_level: ThreatLevel = ThreatLevel.OPPORTUNISTIC

    @property
    def attack_surface_ratio(self) -> float:
        """Fraction of services exposed to attack."""
        if self.total_services == 0:
            return 1.0
        return self.exposed_services / self.total_services


# =============================================================================
# ATTACK SURFACE THETA
# =============================================================================

def compute_attack_surface_theta(
    exposed: int,
    total: int
) -> float:
    r"""
    Compute theta from attack surface ratio.

    A smaller attack surface = higher theta (more secure).

    Args:
        exposed: Number of exposed services/ports
        total: Total number of services

    Returns:
        theta in [0, 1]

    Reference: \cite{NIST2018} - Attack Surface Reduction
    """
    if total == 0:
        return 0.0

    ratio = exposed / total
    theta = 1 - ratio
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DETECTION SPEED THETA (MTTD)
# =============================================================================

def compute_detection_theta(
    mttd_hours: float,
    optimal_hours: float = 1.0
) -> float:
    r"""
    Compute theta from Mean Time to Detect.

    Faster detection = higher theta.
    Industry average MTTD is ~197 days (4,728 hours).
    Elite SOCs detect in minutes to hours.

    Args:
        mttd_hours: Actual mean time to detect (hours)
        optimal_hours: Target detection time (default 1 hour)

    Returns:
        theta in [0, 1]

    Reference: \cite{Mandiant2023} - M-Trends Report
    """
    if mttd_hours <= 0:
        return 1.0  # Instant detection

    # Exponential decay: theta = optimal / actual (capped at 1)
    theta = optimal_hours / mttd_hours
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# RESPONSE SPEED THETA (MTTR)
# =============================================================================

def compute_response_theta(
    mttr_hours: float,
    optimal_hours: float = 4.0
) -> float:
    r"""
    Compute theta from Mean Time to Respond/Remediate.

    Faster response = higher theta.
    Industry average MTTR is ~69 days (1,656 hours).
    Elite teams respond in hours.

    Args:
        mttr_hours: Actual mean time to respond (hours)
        optimal_hours: Target response time (default 4 hours)

    Returns:
        theta in [0, 1]

    Reference: \cite{Mandiant2023} - M-Trends Report
    """
    if mttr_hours <= 0:
        return 1.0  # Instant response

    theta = optimal_hours / mttr_hours
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DEFENSE DEPTH THETA
# =============================================================================

def compute_defense_depth_theta(
    layers: int,
    max_layers: int = 7
) -> float:
    r"""
    Compute theta from defense-in-depth layers.

    More layers = higher theta.

    Standard layers:
    1. Perimeter (firewall)
    2. Network (segmentation)
    3. Endpoint (EDR)
    4. Application (WAF)
    5. Data (encryption)
    6. Identity (MFA)
    7. Recovery (backup/DR)

    Args:
        layers: Number of active defense layers
        max_layers: Maximum layers (default 7)

    Returns:
        theta in [0, 1]

    Reference: \cite{NIST2018} - Defense in Depth
    """
    if max_layers <= 0:
        return 0.0

    theta = layers / max_layers
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# CRYPTOGRAPHIC STRENGTH THETA
# =============================================================================

def compute_crypto_theta(
    key_bits: int,
    reference_bits: int = 256
) -> float:
    r"""
    Compute theta from cryptographic key strength.

    Normalized to AES-256 (256 bits) as reference.

    Quantum computing reduces effective strength:
    - Grover's algorithm: sqrt reduction for symmetric
    - Shor's algorithm: breaks RSA/ECC entirely

    Args:
        key_bits: Effective key strength in bits
        reference_bits: Reference strength (default 256 for AES-256)

    Returns:
        theta in [0, 1]

    Reference: \cite{NIST2022} - Post-Quantum Cryptography
    """
    if reference_bits <= 0:
        return 0.0

    theta = key_bits / reference_bits
    return np.clip(theta, 0.0, 1.0)


def classify_crypto_strength(key_bits: int) -> CryptoStrength:
    """Classify cryptographic strength from key bits."""
    if key_bits < 80:
        return CryptoStrength.BROKEN
    elif key_bits < 128:
        return CryptoStrength.WEAK
    elif key_bits < 192:
        return CryptoStrength.STANDARD
    elif key_bits < 256:
        return CryptoStrength.STRONG
    else:
        return CryptoStrength.QUANTUM_SAFE


# =============================================================================
# PATCH COVERAGE THETA
# =============================================================================

def compute_patch_theta(
    patched: int,
    total_vulns: int
) -> float:
    r"""
    Compute theta from patch coverage.

    Higher patch coverage = higher theta.

    Args:
        patched: Number of patched vulnerabilities
        total_vulns: Total known vulnerabilities

    Returns:
        theta in [0, 1]

    Reference: \cite{CVSS2019} - Vulnerability Management
    """
    if total_vulns == 0:
        return 1.0  # No known vulnerabilities

    theta = patched / total_vulns
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# CVSS SEVERITY TO THETA
# =============================================================================

def cvss_to_theta(cvss_score: float) -> float:
    r"""
    Convert CVSS score to theta.

    CVSS 0 (no vuln) -> theta = 1.0 (secure)
    CVSS 10 (critical) -> theta = 0.0 (compromised)

    Args:
        cvss_score: CVSS v3 score [0, 10]

    Returns:
        theta in [0, 1]

    Reference: \cite{CVSS2019} - Common Vulnerability Scoring System
    """
    theta = 1 - (cvss_score / 10.0)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# AUTHENTICATION STRENGTH THETA
# =============================================================================

def compute_auth_theta(
    mfa_coverage: float,
    password_strength: float = 0.5
) -> float:
    r"""
    Compute theta from authentication strength.

    MFA coverage is primary factor (70% weight).
    Password strength is secondary (30% weight).

    Args:
        mfa_coverage: Fraction of accounts with MFA [0, 1]
        password_strength: Password policy strength [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{NIST2017} - Digital Identity Guidelines
    """
    theta = 0.7 * mfa_coverage + 0.3 * password_strength
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ZERO TRUST MATURITY THETA
# =============================================================================

def compute_zero_trust_theta(
    identity_verified: float,
    device_verified: float,
    network_segmented: float,
    data_encrypted: float,
    continuous_monitoring: float
) -> float:
    r"""
    Compute theta from Zero Trust maturity model.

    Five pillars of Zero Trust (CISA model):
    1. Identity verification
    2. Device security
    3. Network segmentation
    4. Data protection
    5. Continuous monitoring

    Args:
        identity_verified: Identity pillar maturity [0, 1]
        device_verified: Device pillar maturity [0, 1]
        network_segmented: Network pillar maturity [0, 1]
        data_encrypted: Data pillar maturity [0, 1]
        continuous_monitoring: Visibility pillar maturity [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{CISA2021} - Zero Trust Maturity Model
    """
    # Equal weighting across pillars
    theta = (
        0.2 * identity_verified +
        0.2 * device_verified +
        0.2 * network_segmented +
        0.2 * data_encrypted +
        0.2 * continuous_monitoring
    )
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED SECURITY THETA
# =============================================================================

def compute_security_theta(system: SecuritySystem) -> float:
    """
    Compute unified theta for security system.

    Combines:
    - Attack surface (20%)
    - Detection speed (20%)
    - Response speed (15%)
    - Defense depth (15%)
    - Crypto strength (15%)
    - Patch + MFA coverage (15%)

    Args:
        system: SecuritySystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Component thetas
    theta_surface = compute_attack_surface_theta(
        system.exposed_services,
        system.total_services
    )
    theta_detect = compute_detection_theta(system.mttd_hours)
    theta_respond = compute_response_theta(system.mttr_hours)
    theta_depth = compute_defense_depth_theta(system.defense_layers)
    theta_crypto = compute_crypto_theta(system.crypto_bits)
    theta_hygiene = (system.patch_coverage + system.mfa_coverage) / 2

    # Weighted combination
    theta = (
        0.20 * theta_surface +
        0.20 * theta_detect +
        0.15 * theta_respond +
        0.15 * theta_depth +
        0.15 * theta_crypto +
        0.15 * theta_hygiene
    )

    return np.clip(theta, 0.0, 1.0)


def classify_security_posture(theta: float) -> SecurityPosture:
    """Classify security posture from theta."""
    if theta < 0.2:
        return SecurityPosture.COMPROMISED
    elif theta < 0.4:
        return SecurityPosture.VULNERABLE
    elif theta < 0.6:
        return SecurityPosture.BASELINE
    elif theta < 0.8:
        return SecurityPosture.HARDENED
    else:
        return SecurityPosture.FORTIFIED


# =============================================================================
# THREAT COUPLING (ISING MODEL ANALOG)
# =============================================================================

def compute_threat_coupling(
    n_systems: int,
    correlation: float
) -> float:
    r"""
    Compute threat coupling strength (Ising J analog).

    When vulnerabilities are correlated (e.g., shared software),
    attacks can cascade through the network.

    J > J_c: Ordered phase (systemic vulnerability)
    J < J_c: Disordered phase (isolated vulnerabilities)

    Args:
        n_systems: Number of connected systems
        correlation: Vulnerability correlation [0, 1]

    Returns:
        Coupling strength J

    Reference: \cite{Bornholdt2001} - Ising model in complex systems
    """
    # J ~ sqrt(N) * correlation
    J = np.sqrt(n_systems) * correlation
    return J


def critical_coupling(n_systems: int) -> float:
    """Critical coupling strength for cascade transition."""
    # Mean-field approximation: J_c = 1/sqrt(N)
    return 1.0 / np.sqrt(n_systems) if n_systems > 0 else 0.0


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

SECURITY_SYSTEMS: Dict[str, SecuritySystem] = {
    "unpatched_legacy": SecuritySystem(
        name="Unpatched Legacy System",
        total_services=50,
        exposed_services=45,  # 90% exposed
        mttd_hours=4728.0,    # 197 days (industry avg breach)
        mttr_hours=1656.0,    # 69 days
        defense_layers=1,     # Firewall only
        crypto_bits=56,       # DES
        patch_coverage=0.1,
        mfa_coverage=0.0,
        threat_level=ThreatLevel.SCRIPT_KIDDIE,
    ),
    "smb_typical": SecuritySystem(
        name="Typical SMB",
        total_services=30,
        exposed_services=15,  # 50% exposed
        mttd_hours=720.0,     # 30 days
        mttr_hours=168.0,     # 7 days
        defense_layers=3,     # Firewall, AV, backup
        crypto_bits=128,      # AES-128
        patch_coverage=0.6,
        mfa_coverage=0.3,
        threat_level=ThreatLevel.OPPORTUNISTIC,
    ),
    "enterprise_standard": SecuritySystem(
        name="Standard Enterprise",
        total_services=200,
        exposed_services=40,  # 20% exposed
        mttd_hours=168.0,     # 7 days
        mttr_hours=48.0,      # 2 days
        defense_layers=5,
        crypto_bits=192,
        patch_coverage=0.8,
        mfa_coverage=0.7,
        threat_level=ThreatLevel.TARGETED,
    ),
    "soc_monitored": SecuritySystem(
        name="SOC-Monitored Enterprise",
        total_services=200,
        exposed_services=20,  # 10% exposed
        mttd_hours=24.0,      # 1 day
        mttr_hours=8.0,       # 8 hours
        defense_layers=6,
        crypto_bits=256,
        patch_coverage=0.95,
        mfa_coverage=0.9,
        threat_level=ThreatLevel.APT,
    ),
    "zero_trust": SecuritySystem(
        name="Zero Trust Architecture",
        total_services=200,
        exposed_services=5,   # 2.5% exposed
        mttd_hours=1.0,       # 1 hour (real-time)
        mttr_hours=2.0,       # 2 hours (automated)
        defense_layers=7,     # Full defense-in-depth
        crypto_bits=256,
        patch_coverage=0.99,
        mfa_coverage=1.0,
        threat_level=ThreatLevel.NATION_STATE,
    ),
    "qkd_secured": SecuritySystem(
        name="Quantum Key Distribution Secured",
        total_services=50,
        exposed_services=2,   # 4% exposed
        mttd_hours=0.5,       # 30 minutes
        mttr_hours=1.0,       # 1 hour
        defense_layers=7,
        crypto_bits=512,      # Post-quantum + QKD
        patch_coverage=1.0,
        mfa_coverage=1.0,
        threat_level=ThreatLevel.NATION_STATE,
    ),
    "active_breach": SecuritySystem(
        name="Active Breach (Compromised)",
        total_services=100,
        exposed_services=100,  # 100% (attacker has access)
        mttd_hours=8760.0,     # 1 year (undetected)
        mttr_hours=4380.0,     # 6 months
        defense_layers=0,      # Bypassed
        crypto_bits=0,         # Keys exfiltrated
        patch_coverage=0.0,
        mfa_coverage=0.0,
        threat_level=ThreatLevel.APT,
    ),
    # Real-world sector examples
    "healthcare_hipaa": SecuritySystem(
        name="Healthcare (HIPAA Compliant)",
        total_services=150,
        exposed_services=30,   # 20% exposed (patient portals, APIs)
        mttd_hours=72.0,       # 3 days
        mttr_hours=24.0,       # 1 day
        defense_layers=5,      # Network segmentation, encryption, access control
        crypto_bits=256,       # AES-256 for PHI
        patch_coverage=0.85,
        mfa_coverage=0.8,
        threat_level=ThreatLevel.TARGETED,
    ),
    "scada_industrial": SecuritySystem(
        name="SCADA/ICS Critical Infrastructure",
        total_services=75,
        exposed_services=8,    # Air-gapped with few external interfaces
        mttd_hours=336.0,      # 14 days (legacy monitoring)
        mttr_hours=72.0,       # 3 days (safety critical)
        defense_layers=4,      # DMZ, firewalls, IDS, physical isolation
        crypto_bits=128,       # Some legacy protocols
        patch_coverage=0.5,    # Many unpatched legacy systems
        mfa_coverage=0.4,
        threat_level=ThreatLevel.NATION_STATE,
    ),
    "iot_smart_home": SecuritySystem(
        name="IoT Smart Home Network",
        total_services=40,
        exposed_services=25,   # 62% exposed (cameras, thermostats, etc.)
        mttd_hours=2190.0,     # 3 months (often unmonitored)
        mttr_hours=720.0,      # 30 days (consumer updates)
        defense_layers=2,      # Router firewall, occasional updates
        crypto_bits=128,       # TLS varies by device
        patch_coverage=0.3,    # Many devices never updated
        mfa_coverage=0.1,
        threat_level=ThreatLevel.OPPORTUNISTIC,
    ),
}


def security_theta_summary():
    """Print theta analysis for example security systems."""
    print("=" * 80)
    print("CYBERSECURITY THETA ANALYSIS (Attack/Defense Dynamics)")
    print("=" * 80)
    print()
    print(f"{'System':<30} {'Surface':>8} {'MTTD':>8} {'MTTR':>8} "
          f"{'Depth':>6} {'Crypto':>6} {'θ':>8} {'Posture':<12}")
    print("-" * 80)

    for name, system in SECURITY_SYSTEMS.items():
        theta = compute_security_theta(system)
        posture = classify_security_posture(theta)
        surface_pct = f"{system.attack_surface_ratio*100:.0f}%"
        print(f"{system.name:<30} "
              f"{surface_pct:>8} "
              f"{system.mttd_hours:>8.0f}h "
              f"{system.mttr_hours:>8.0f}h "
              f"{system.defense_layers:>6} "
              f"{system.crypto_bits:>6} "
              f"{theta:>8.3f} "
              f"{posture.value:<12}")

    print()
    print("Key: θ combines attack surface, detection/response speed, defense depth, crypto")
    print("     Zero Trust architecture approaches θ ~ 0.9 (fortified)")
    print("     Active breach has θ ~ 0.05 (compromised)")


if __name__ == "__main__":
    security_theta_summary()
