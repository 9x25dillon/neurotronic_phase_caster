"""
Neurotronic Phase Caster
THz Coherence Wearable for Active Consciousness Engineering

Modules:
- yhwh_soliton_field_physics: Five-substrate consciousness field framework
- yhwh_abcr_integration: YHWH-ABCR unified coherence system
- thz_coherence_wearable_spec: THz wearable hardware controller
- QABCr: Adaptive Bi-Coupled Coherence Recovery system
- photonic_thz_pulse_engine: all-optical THz-rate IR pulse synthesis and photomixer backend
- water_structure_assay: preregistered, blinded persistence and symbol-channel test in liquid water
- order_records: operational estimators for absence, order fields, geometric clocks and records
- crystal_writer: writes a symbol stream into a growing crystal as dopant striations and reads it back
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

try:
    from .photonic_thz_pulse_engine import (
        CombState,
        make_comb,
        pulse_metrics,
        HeterodynePair,
        MicroringResonator,
        FabryPerotEtalon,
        TalbotMultiplier,
        KerrGate,
        SaturableAbsorberGate,
        Photomixer,
        OpticalRectifier,
        tissue_transmission,
        assess_skin_exposure,
        PhotonicTHzEmitterArray,
        PhotonicDrivePlan,
    )
except ImportError:
    pass

try:
    from .water_structure_assay import (
        SampleCell,
        RingIllumination,
        SymbolCodebook,
        TextStream,
        EEGSubstrateStream,
        RandomStream,
        Instrument,
        SimulatedInstrument,
        Preregistration,
        run_assay,
        analyze,
        unblind,
        simulate_assay,
    )
except ImportError:
    pass

try:
    from .order_records import (
        systematic_absences,
        ice_ih_absence_rule,
        tetrahedral_order,
        hexatic_defects,
        order_deficit,
        simulate_phase_ordering,
        CoarseningClock,
        box_counting_dimension,
        record_fidelity,
        record_lifetime,
    )
except ImportError:
    pass

try:
    from .crystal_writer import (
        GrowthSystem,
        GROWTH_SYSTEMS,
        WriterCodebook,
        make_actuator,
        write_crystal,
        read_crystal,
        run_writer,
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
    # Photonic THz pulse engine
    "CombState",
    "make_comb",
    "pulse_metrics",
    "HeterodynePair",
    "MicroringResonator",
    "FabryPerotEtalon",
    "TalbotMultiplier",
    "KerrGate",
    "SaturableAbsorberGate",
    "Photomixer",
    "OpticalRectifier",
    "tissue_transmission",
    "assess_skin_exposure",
    "PhotonicTHzEmitterArray",
    "PhotonicDrivePlan",
    # Water structure assay
    "SampleCell",
    "RingIllumination",
    "SymbolCodebook",
    "TextStream",
    "EEGSubstrateStream",
    "RandomStream",
    "Instrument",
    "SimulatedInstrument",
    "Preregistration",
    "run_assay",
    "analyze",
    "unblind",
    "simulate_assay",
    # Order, absence and records
    "systematic_absences",
    "ice_ih_absence_rule",
    "tetrahedral_order",
    "hexatic_defects",
    "order_deficit",
    "simulate_phase_ordering",
    "CoarseningClock",
    "box_counting_dimension",
    "record_fidelity",
    "record_lifetime",
    # Crystal writer
    "GrowthSystem",
    "GROWTH_SYSTEMS",
    "WriterCodebook",
    "make_actuator",
    "write_crystal",
    "read_crystal",
    "run_writer",
]
