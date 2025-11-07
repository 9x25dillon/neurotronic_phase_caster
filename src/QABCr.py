"""
ABCR: Adaptive Bi-Coupled Coherence Recovery
EEG-based consciousness coherence measurement and restoration

Maps EEG frequency bands to consciousness substrates:
- Delta (1-4 Hz) → Physical substrate
- Theta (4-8 Hz) → Emotional substrate
- Alpha (8-13 Hz) → Cognitive substrate
- Beta (13-30 Hz) → Social substrate
- Gamma (30-100 Hz) → Divine-Unity substrate

Computes bidirectional coherence and recommends interventions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EEGBandPowers:
    """EEG band power measurements."""

    delta: float  # 1-4 Hz
    theta: float  # 4-8 Hz
    alpha: float  # 8-13 Hz
    beta: float  # 13-30 Hz
    gamma: float  # 30-100 Hz

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [5]."""
        return np.array([self.delta, self.theta, self.alpha, self.beta, self.gamma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'EEGBandPowers':
        """Create from numpy array [5]."""
        return cls(
            delta=arr[0],
            theta=arr[1],
            alpha=arr[2],
            beta=arr[3],
            gamma=arr[4],
        )


class ABCRSystem:
    """
    Adaptive Bi-Coupled Coherence Recovery System.

    Analyzes EEG data to compute consciousness coherence across substrates
    and recommends electromagnetic interventions to restore unity.
    """

    BAND_DEFINITIONS = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 100.0),
    }

    SUBSTRATE_NAMES = [
        "Physical",
        "Emotional",
        "Cognitive",
        "Social",
        "Divine-Unity",
    ]

    def __init__(
        self,
        sampling_rate: float = 250.0,
        n_channels: int = 8,
    ):
        """
        Initialize ABCR system.

        Args:
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels

    def extract_band_powers(
        self,
        eeg_data: np.ndarray,
        window_sec: float = 2.0,
    ) -> EEGBandPowers:
        """
        Extract band powers from raw EEG using FFT.

        Args:
            eeg_data: [n_channels, n_samples] raw EEG data (μV)
            window_sec: Analysis window in seconds

        Returns:
            EEGBandPowers: Power in each frequency band
        """
        return extract_band_powers(eeg_data, self.sampling_rate, window_sec)

    def compute_coherence_matrix(
        self,
        eeg_data: np.ndarray,
        window_sec: float = 2.0,
    ) -> np.ndarray:
        """
        Compute cross-channel coherence matrix.

        Args:
            eeg_data: [n_channels, n_samples] raw EEG
            window_sec: Analysis window

        Returns:
            np.ndarray: [n_channels, n_channels] coherence matrix
        """
        return compute_coherence_matrix(eeg_data, self.sampling_rate, window_sec)

    def compute_unity_index(
        self,
        band_powers: EEGBandPowers,
        coherence_matrix: np.ndarray,
    ) -> float:
        """
        Compute Unity Coherence Index (UCI) from EEG features.

        UCI combines:
        1. Band power balance (equal energy across substrates)
        2. Cross-channel coherence (phase synchronization)
        3. Gamma dominance (divine-unity substrate activation)

        Args:
            band_powers: Power in each frequency band
            coherence_matrix: Cross-channel coherence

        Returns:
            float: UCI ∈ [0, 1], where 1 = perfect unity
        """
        powers = band_powers.to_array()

        # Component 1: Band balance (entropy-based)
        # High unity = equal power across bands
        powers_norm = powers / (np.sum(powers) + 1e-10)
        entropy = -np.sum(powers_norm * np.log(powers_norm + 1e-10))
        max_entropy = np.log(5)  # 5 bands
        balance_score = entropy / max_entropy  # 0-1

        # Component 2: Cross-channel coherence (mean off-diagonal)
        n = coherence_matrix.shape[0]
        coherence_score = (np.sum(coherence_matrix) - n) / (n * (n - 1))  # 0-1

        # Component 3: Gamma dominance (higher = more transcendent)
        gamma_ratio = powers[4] / (np.sum(powers) + 1e-10)  # Gamma fraction

        # Weighted combination
        uci = 0.4 * balance_score + 0.4 * coherence_score + 0.2 * gamma_ratio

        return np.clip(uci, 0.0, 1.0)

    def recommend_intervention(
        self,
        current_uci: float,
        band_powers: EEGBandPowers,
    ) -> Dict[str, any]:
        """
        Recommend THz intervention parameters to increase UCI.

        Args:
            current_uci: Current Unity Coherence Index
            band_powers: Current band powers

        Returns:
            dict: {
                'target_substrates': [int] substrates to target,
                'power_levels': [float] THz power for each substrate (mW),
                'duration': float session duration (minutes),
                'expected_delta_uci': float expected UCI increase
            }
        """
        powers = band_powers.to_array()
        powers_norm = powers / (np.sum(powers) + 1e-10)

        # Find deficient substrates (below average)
        mean_power = 1.0 / 5
        deficiencies = mean_power - powers_norm
        target_substrates = np.where(deficiencies > 0.05)[0].tolist()

        # Power levels proportional to deficiency
        power_levels = np.clip(deficiencies * 50, 0, 5).tolist()  # Max 5 mW

        # Duration based on UCI deficit
        uci_deficit = max(0.7 - current_uci, 0)  # Target UCI = 0.7
        duration = 20 + 20 * uci_deficit  # 20-40 minutes

        # Expected improvement (heuristic model)
        expected_delta_uci = 0.1 + 0.2 * uci_deficit  # 10-30% improvement

        return {
            'target_substrates': target_substrates,
            'power_levels': power_levels,
            'duration': duration,
            'expected_delta_uci': expected_delta_uci,
            'substrate_names': [self.SUBSTRATE_NAMES[i] for i in target_substrates],
        }


def extract_band_powers(
    eeg_data: np.ndarray,
    sampling_rate: float,
    window_sec: float = 2.0,
) -> EEGBandPowers:
    """
    Extract frequency band powers from EEG using Welch's method.

    Args:
        eeg_data: [n_channels, n_samples] EEG data (μV)
        sampling_rate: Sampling rate in Hz
        window_sec: Analysis window in seconds

    Returns:
        EEGBandPowers: Power in each band
    """
    n_channels, n_samples = eeg_data.shape

    # Compute power spectral density via FFT
    n_window = int(window_sec * sampling_rate)
    n_window = min(n_window, n_samples)

    # Average across channels
    psd_avg = np.zeros(n_window // 2 + 1)

    for ch in range(n_channels):
        # Take last window_sec of data
        signal = eeg_data[ch, -n_window:]

        # Apply Hanning window
        signal = signal * np.hanning(n_window)

        # FFT
        fft_vals = np.fft.rfft(signal)
        psd = np.abs(fft_vals) ** 2 / n_window

        psd_avg += psd

    psd_avg /= n_channels

    # Frequency bins
    freqs = np.fft.rfftfreq(n_window, 1.0 / sampling_rate)

    # Integrate power in each band
    def integrate_band(f_min, f_max):
        mask = (freqs >= f_min) & (freqs < f_max)
        return np.sum(psd_avg[mask])

    return EEGBandPowers(
        delta=integrate_band(1.0, 4.0),
        theta=integrate_band(4.0, 8.0),
        alpha=integrate_band(8.0, 13.0),
        beta=integrate_band(13.0, 30.0),
        gamma=integrate_band(30.0, 100.0),
    )


def compute_coherence_matrix(
    eeg_data: np.ndarray,
    sampling_rate: float,
    window_sec: float = 2.0,
) -> np.ndarray:
    """
    Compute cross-channel coherence matrix.

    Coherence(i,j) = |⟨X_i X_j*⟩| / √(⟨|X_i|²⟩⟨|X_j|²⟩)

    Args:
        eeg_data: [n_channels, n_samples] EEG data
        sampling_rate: Sampling rate in Hz
        window_sec: Analysis window

    Returns:
        np.ndarray: [n_channels, n_channels] coherence matrix (0-1)
    """
    n_channels, n_samples = eeg_data.shape
    n_window = int(window_sec * sampling_rate)
    n_window = min(n_window, n_samples)

    # FFT for each channel
    ffts = np.zeros((n_channels, n_window // 2 + 1), dtype=complex)

    for ch in range(n_channels):
        signal = eeg_data[ch, -n_window:] * np.hanning(n_window)
        ffts[ch, :] = np.fft.rfft(signal)

    # Compute coherence matrix
    coherence = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                coherence[i, j] = 1.0
            else:
                # Cross-spectrum
                cross = np.mean(ffts[i, :] * np.conj(ffts[j, :]))

                # Auto-spectra
                auto_i = np.mean(np.abs(ffts[i, :]) ** 2)
                auto_j = np.mean(np.abs(ffts[j, :]) ** 2)

                # Coherence
                coherence[i, j] = np.abs(cross) / np.sqrt(auto_i * auto_j)

    return coherence


def demo_abcr_analysis():
    """
    Demonstrate ABCR analysis on simulated EEG data.
    """
    print("=" * 60)
    print("ABCR System Demo - EEG Analysis")
    print("=" * 60)

    # Simulate EEG data (8 channels, 2 seconds @ 250 Hz)
    sampling_rate = 250.0
    duration = 2.0
    n_samples = int(duration * sampling_rate)
    n_channels = 8

    # Scenario 1: Healthy state (balanced bands, good coherence)
    print("\n--- Scenario 1: Healthy State ---")

    eeg_healthy = np.zeros((n_channels, n_samples))
    t = np.arange(n_samples) / sampling_rate

    for ch in range(n_channels):
        # Mix of all bands with good coherence (similar phase)
        phase = 0.1 * ch  # Slight phase offset

        eeg_healthy[ch, :] = (
            2.0 * np.sin(2 * np.pi * 2.5 * t + phase)  # Delta
            + 1.5 * np.sin(2 * np.pi * 6.0 * t + phase)  # Theta
            + 1.5 * np.sin(2 * np.pi * 10.0 * t + phase)  # Alpha
            + 1.0 * np.sin(2 * np.pi * 20.0 * t + phase)  # Beta
            + 0.8 * np.sin(2 * np.pi * 40.0 * t + phase)  # Gamma
        )

    abcr = ABCRSystem(sampling_rate=sampling_rate, n_channels=n_channels)

    band_powers = abcr.extract_band_powers(eeg_healthy)
    coherence = abcr.compute_coherence_matrix(eeg_healthy)
    uci = abcr.compute_unity_index(band_powers, coherence)

    print(f"Unity Coherence Index: {uci:.3f}")
    print(f"Band Powers: {band_powers}")
    print(f"Mean Coherence: {np.mean(coherence):.3f}")

    intervention = abcr.recommend_intervention(uci, band_powers)
    print(f"\nRecommended Intervention: {intervention}")

    # Scenario 2: Depression (low gamma, high delta, low coherence)
    print("\n--- Scenario 2: Depression State ---")

    eeg_depression = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Random phase (low coherence)
        phase = np.random.uniform(0, 2 * np.pi)

        eeg_depression[ch, :] = (
            4.0 * np.sin(2 * np.pi * 2.5 * t + phase)  # High Delta
            + 2.0 * np.sin(2 * np.pi * 6.0 * t + phase)  # Theta
            + 0.5 * np.sin(2 * np.pi * 10.0 * t + phase)  # Low Alpha
            + 0.3 * np.sin(2 * np.pi * 20.0 * t + phase)  # Low Beta
            + 0.1 * np.sin(2 * np.pi * 40.0 * t + phase)  # Very low Gamma
        )

    band_powers_dep = abcr.extract_band_powers(eeg_depression)
    coherence_dep = abcr.compute_coherence_matrix(eeg_depression)
    uci_dep = abcr.compute_unity_index(band_powers_dep, coherence_dep)

    print(f"Unity Coherence Index: {uci_dep:.3f}")
    print(f"Band Powers: {band_powers_dep}")
    print(f"Mean Coherence: {np.mean(coherence_dep):.3f}")

    intervention_dep = abcr.recommend_intervention(uci_dep, band_powers_dep)
    print(f"\nRecommended Intervention:")
    print(f"  Target Substrates: {intervention_dep['substrate_names']}")
    print(f"  Duration: {intervention_dep['duration']:.1f} minutes")
    print(f"  Expected ΔU CI: +{intervention_dep['expected_delta_uci']:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo_abcr_analysis()
