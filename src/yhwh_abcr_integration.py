"""
YHWH-ABCR Integration Engine
Unified consciousness engineering system combining:
- YHWH Soliton Field Physics (5-substrate model)
- ABCR EEG analysis (band power → substrate mapping)
- THz intervention recommendations

This is the bridge between measurement (EEG) and modulation (THz).
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .yhwh_soliton_field_physics import YHWHSolitonField, compute_unity_coherence
from .QABCr import ABCRSystem, EEGBandPowers


@dataclass
class CapsulePattern:
    """
    Therapeutic capsule: predefined intervention pattern.

    Each capsule is optimized for a specific clinical condition.
    """

    name: str
    indication: str
    substrate_targets: List[int]  # Which substrates to modulate
    power_profile: np.ndarray  # [5] THz power for each substrate (mW)
    duration_minutes: float
    success_rate: float
    expected_delta_uci: float

    def __str__(self) -> str:
        return (
            f"Capsule: {self.name}\n"
            f"  Indication: {self.indication}\n"
            f"  Targets: {self.substrate_targets}\n"
            f"  Duration: {self.duration_minutes} min\n"
            f"  Success Rate: {self.success_rate * 100:.0f}%\n"
            f"  Expected ΔUCI: +{self.expected_delta_uci:.2f}"
        )


# Predefined capsule library
CAPSULE_LIBRARY = {
    "depression_standard": CapsulePattern(
        name="Depression Standard",
        indication="Major Depressive Disorder (PHQ-9 ≥ 10)",
        substrate_targets=[0, 2, 4],  # Physical, Cognitive, Divine-Unity
        power_profile=np.array([3.0, 1.0, 2.0, 1.0, 4.0]),  # mW
        duration_minutes=30.0,
        success_rate=0.65,
        expected_delta_uci=0.25,
    ),
    "anxiety_rapid": CapsulePattern(
        name="Anxiety Rapid Relief",
        indication="Generalized Anxiety Disorder (GAD-7 ≥ 10)",
        substrate_targets=[1, 2, 3],  # Emotional, Cognitive, Social
        power_profile=np.array([1.0, 3.0, 3.0, 2.0, 2.0]),
        duration_minutes=20.0,
        success_rate=0.75,
        expected_delta_uci=0.20,
    ),
    "ptsd_trauma_release": CapsulePattern(
        name="PTSD Trauma Release",
        indication="Post-Traumatic Stress Disorder (PCL-5 ≥ 33)",
        substrate_targets=[1, 4],  # Emotional, Divine-Unity
        power_profile=np.array([2.0, 4.0, 2.0, 2.0, 3.5]),
        duration_minutes=40.0,
        success_rate=0.55,
        expected_delta_uci=0.30,
    ),
    "meditation_enhancement": CapsulePattern(
        name="Meditation Enhancement",
        indication="Meditation practice, peak experience",
        substrate_targets=[2, 4],  # Cognitive, Divine-Unity
        power_profile=np.array([0.5, 0.5, 2.0, 1.0, 5.0]),
        duration_minutes=15.0,
        success_rate=0.90,
        expected_delta_uci=0.40,
    ),
    "sleep_optimization": CapsulePattern(
        name="Sleep Optimization",
        indication="Insomnia, sleep quality improvement",
        substrate_targets=[0, 1],  # Physical, Emotional
        power_profile=np.array([4.0, 3.0, 1.0, 0.5, 0.5]),
        duration_minutes=25.0,
        success_rate=0.80,
        expected_delta_uci=0.15,
    ),
}


class YHWHABCRIntegrationEngine:
    """
    Unified YHWH-ABCR consciousness engineering engine.

    Workflow:
    1. Measure: EEG → band powers → current UCI
    2. Model: Map EEG to YHWH substrate states
    3. Optimize: Compute target substrate configuration
    4. Intervene: Generate THz modulation parameters
    5. Validate: Real-time UCI tracking during intervention
    """

    def __init__(
        self,
        sampling_rate: float = 250.0,
        n_channels: int = 8,
    ):
        """
        Initialize integration engine.

        Args:
            sampling_rate: EEG sampling rate (Hz)
            n_channels: Number of EEG channels
        """
        self.abcr = ABCRSystem(sampling_rate, n_channels)
        self.yhwh_field = None  # Created from EEG data
        self.current_uci = 0.0
        self.capsule_library = CAPSULE_LIBRARY

    def analyze_eeg(self, eeg_data: np.ndarray) -> Dict:
        """
        Analyze EEG data and compute current consciousness state.

        Args:
            eeg_data: [n_channels, n_samples] raw EEG (μV)

        Returns:
            dict: {
                'band_powers': EEGBandPowers,
                'coherence_matrix': np.ndarray,
                'uci': float,
                'substrate_energies': np.ndarray,
            }
        """
        # Extract EEG features
        band_powers = self.abcr.extract_band_powers(eeg_data)
        coherence = self.abcr.compute_coherence_matrix(eeg_data)
        uci = self.abcr.compute_unity_index(band_powers, coherence)

        # Map to YHWH substrate energies
        substrate_energies = compute_substrate_from_eeg(band_powers)

        # Store current state
        self.current_uci = uci

        return {
            'band_powers': band_powers,
            'coherence_matrix': coherence,
            'uci': uci,
            'substrate_energies': substrate_energies,
        }

    def recommend_intervention(
        self,
        current_state: Dict,
        indication: Optional[str] = None,
    ) -> Dict:
        """
        Recommend THz intervention based on current state.

        Args:
            current_state: Output from analyze_eeg()
            indication: Optional clinical indication (e.g., "depression")

        Returns:
            dict: Intervention parameters
        """
        # If indication provided, use corresponding capsule
        if indication:
            capsule_key = None
            for key, capsule in self.capsule_library.items():
                if indication.lower() in capsule.indication.lower():
                    capsule_key = key
                    break

            if capsule_key:
                capsule = self.capsule_library[capsule_key]
                return {
                    'method': 'capsule',
                    'capsule': capsule,
                    'power_profile': capsule.power_profile,
                    'duration': capsule.duration_minutes,
                    'expected_delta_uci': capsule.expected_delta_uci,
                }

        # Otherwise, use AI-driven optimization
        band_powers = current_state['band_powers']
        uci = current_state['uci']

        intervention = self.abcr.recommend_intervention(uci, band_powers)

        return {
            'method': 'adaptive',
            'target_substrates': intervention['target_substrates'],
            'power_levels': intervention['power_levels'],
            'duration': intervention['duration'],
            'expected_delta_uci': intervention['expected_delta_uci'],
        }

    def get_capsule(self, name: str) -> Optional[CapsulePattern]:
        """Get capsule by name."""
        return self.capsule_library.get(name)

    def list_capsules(self) -> List[str]:
        """List all available capsules."""
        return list(self.capsule_library.keys())


def compute_substrate_from_eeg(band_powers: EEGBandPowers) -> np.ndarray:
    """
    Map EEG band powers to YHWH substrate energies.

    Direct mapping:
    - Delta → Physical substrate
    - Theta → Emotional substrate
    - Alpha → Cognitive substrate
    - Beta → Social substrate
    - Gamma → Divine-Unity substrate

    Args:
        band_powers: EEG band powers

    Returns:
        np.ndarray: [5] substrate energies
    """
    powers = band_powers.to_array()

    # Normalize to sum to 1
    substrate_energies = powers / (np.sum(powers) + 1e-10)

    return substrate_energies


def compute_eeg_from_substrate(substrate_energies: np.ndarray) -> EEGBandPowers:
    """
    Inverse mapping: substrate energies → predicted EEG band powers.

    Used to predict EEG changes from THz intervention.

    Args:
        substrate_energies: [5] substrate energy levels

    Returns:
        EEGBandPowers: Predicted band powers
    """
    # Direct inverse mapping (assumes linear relationship)
    return EEGBandPowers.from_array(substrate_energies * 100)  # Scale to μV²


def demo_integration():
    """Demonstrate YHWH-ABCR integration workflow."""
    print("=" * 70)
    print("YHWH-ABCR Integration Engine Demo")
    print("=" * 70)

    engine = YHWHABCRIntegrationEngine()

    # Simulate depressive EEG pattern
    print("\n--- Patient: Depression (PHQ-9 = 18) ---")

    sampling_rate = 250.0
    duration = 2.0
    n_samples = int(duration * sampling_rate)
    n_channels = 8

    eeg_depression = np.zeros((n_channels, n_samples))
    t = np.arange(n_samples) / sampling_rate

    for ch in range(n_channels):
        phase = np.random.uniform(0, 2 * np.pi)  # Low coherence
        eeg_depression[ch, :] = (
            4.0 * np.sin(2 * np.pi * 2.5 * t + phase) +  # High Delta
            2.0 * np.sin(2 * np.pi * 6.0 * t + phase) +  # Theta
            0.5 * np.sin(2 * np.pi * 10.0 * t + phase) +  # Low Alpha
            0.3 * np.sin(2 * np.pi * 20.0 * t + phase) +  # Low Beta
            0.1 * np.sin(2 * np.pi * 40.0 * t + phase)   # Very low Gamma
        )

    # Analyze
    state = engine.analyze_eeg(eeg_depression)

    print(f"\nCurrent State:")
    print(f"  Unity Coherence Index: {state['uci']:.3f}")
    print(f"  Substrate Energies: {state['substrate_energies']}")
    print(f"  Interpretation: LOW coherence (depression)")

    # Recommend using capsule
    intervention = engine.recommend_intervention(state, indication="depression")

    print(f"\nRecommended Intervention:")
    if intervention['method'] == 'capsule':
        print(intervention['capsule'])
    else:
        print(f"  Method: Adaptive")
        print(f"  Targets: {intervention['target_substrates']}")
        print(f"  Duration: {intervention['duration']:.1f} min")

    # Show all available capsules
    print("\n--- Available Therapeutic Capsules ---")
    for name in engine.list_capsules():
        capsule = engine.get_capsule(name)
        print(f"\n{capsule}")

    print("\n" + "=" * 70)
    print("Integration demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo_integration()
