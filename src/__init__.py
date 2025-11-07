"""
Neurotronic Phase Caster
THz Coherence Wearable for Active Consciousness Engineering

Modules:
- yhwh_soliton_field_physics: Five-substrate consciousness field framework
- yhwh_abcr_integration: YHWH-ABCR unified coherence system
- thz_coherence_wearable_spec: THz wearable hardware controller
- QABCr: Adaptive Bi-Coupled Coherence Recovery system
"""

__version__ = "0.1.0-alpha"
__author__ = "Chris Sweigard"
__license__ = "MIT"

# Core exports
try:
    from .yhwh_soliton_field_physics import (
        SpacetimePoint,
        SubstrateLayer,
        YHWHSolitonField,
        compute_unity_coherence,
    )
except ImportError:
    pass

try:
    from .yhwh_abcr_integration import (
        YHWHABCRIntegrationEngine,
        CapsulePattern,
        compute_substrate_from_eeg,
        compute_eeg_from_substrate,
    )
except ImportError:
    pass

try:
    from .thz_coherence_wearable_spec import (
        THzCoherenceWearable,
        SafetySystem,
        EmitterArray,
    )
except ImportError:
    pass

try:
    from .QABCr import (
        ABCRSystem,
        extract_band_powers,
        compute_coherence_matrix,
    )
except ImportError:
    pass

__all__ = [
    # YHWH Physics
    "SpacetimePoint",
    "SubstrateLayer",
    "YHWHSolitonField",
    "compute_unity_coherence",
    # YHWH-ABCR Integration
    "YHWHABCRIntegrationEngine",
    "CapsulePattern",
    "compute_substrate_from_eeg",
    "compute_eeg_from_substrate",
    # THz Wearable
    "THzCoherenceWearable",
    "SafetySystem",
    "EmitterArray",
    # ABCR System
    "ABCRSystem",
    "extract_band_powers",
    "compute_coherence_matrix",
]
