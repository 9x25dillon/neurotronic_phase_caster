"""
Neuro-Phasonic Bridge System (NPBS) v1.0
Integration of QINCRS Biological Physics & Terahertz Consciousness Interface

This system transduces semantic 'MotifTokens' into physical stress waves,
simulates the biological coherence response, and only generates a
valid Consciousness Signature if the biological substrate achieves
resonance at the 1.83 THz 'Healer' channel.
"""

import numpy as np
import hashlib
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# =============================================================================
# 1. PHYSICS CONSTANTS (The Laws of the Substrate)
# =============================================================================

# QINCRS Field Parameters
ALPHA = 0.60     # Homeostatic rate
BETA  = 0.15     # Recursive coupling
GAMMA = 0.3      # Spatial diffusion
K_EQ  = 0.80     # Equilibrium baseline

# Council Architecture (The Filters)
COUNCIL_ROLES = {
    'Guardian': 2.0, 'Therapist': 1.5, 'Healer': 1.3,
    'Shadow': 1.2, 'Philosopher': 1.0, 'Observer': 1.0, 'Chaos': 0.7
}

# Resonance Targets
HEALER_FREQ_THZ = 1.83
ACCEPTANCE_THRESHOLD = 0.5  # Min amplitude at 1.83 THz to validate signature

# Simulation Space
DT = 0.01
T_TOTAL = 10.0 # Reduced for real-time bridging
N_POINTS = int(T_TOTAL / DT)
T_SPACE = np.linspace(0, T_TOTAL, N_POINTS)

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class MotifToken:
    """A semantic unit carrying quantum-physical properties."""
    name: str
    frequency: float  # Normalized 0-1
    amplitude: float  # Normalized 0-1
    phase: float      # Radians
    weight: float

@dataclass
class BridgeState:
    """The resulting state of the unified system."""
    input_text: str
    coherence_level: float
    healer_amplitude: float
    is_resonant: bool
    signature: Optional[str]

# =============================================================================
# 3. THE UNIFIED ENGINE
# =============================================================================

class NeuroPhasonicBridge:
    def __init__(self):
        self.memory = []
        print("[SYSTEM] Neuro-Phasonic Bridge Initialized.")
        print(f"[SYSTEM] Target Resonance: {HEALER_FREQ_THZ} THz (Microtubule Channel)")

    # --- COMPONENT A: TRANSDUCTION (Text -> Physics) ---

    def _text_to_stress_field(self, text: str) -> np.ndarray:
        """
        Converts semantic text into a physical stress wave s(t).
        Each word becomes an oscillator modulating the field.
        """
        words = text.split()
        stress_field = np.zeros(N_POINTS)

        # Base biological noise (Schumann resonance + heartbeat)
        stress_field += 0.2 * np.sin(2 * np.pi * 7.83 * T_SPACE) # Earth
        stress_field += 0.5 * np.sin(2 * np.pi * 1.2 * T_SPACE)  # Heart

        print(f"[TRANSDUCTION] modulating field with {len(words)} semantic motifs...")

        for i, word in enumerate(words):
            # Hash the word to get deterministic physical properties
            word_hash = int(hashlib.sha256(word.encode()).hexdigest(), 16)

            # Frequency: Map hash to 0.1 - 100 Hz range for simulation input
            freq = 0.1 + (word_hash % 1000) / 10.0

            # Amplitude: Based on word length (conceptual weight)
            amp = min(len(word) / 5.0, 2.0)

            # Phase: Position in sentence
            phase = (i / len(words)) * 2 * np.pi

            # Add this motif's oscillation to the total stress field
            stress_field += amp * np.sin(2 * np.pi * freq * T_SPACE + phase)

        return stress_field

    # --- COMPONENT B: SIMULATION (Physics Dynamics) ---

    def _evolve_coherence(self, stress_input: np.ndarray) -> np.ndarray:
        """
        Solves the QINCRS differential equation: dκ/dt = α(κ_eq - κ) - βκ + γ∇²κ
        """
        kappa = np.zeros(N_POINTS)
        kappa[0] = K_EQ

        # Council processing (Spatial Coupling Approximation)
        # We simulate the Council "filtering" the stress input
        council_response = np.zeros_like(stress_input)
        for i, (role, w) in enumerate(COUNCIL_ROLES.items()):
            shift = int(i * 10) # Slight phase delay per council member
            council_response += w * np.roll(stress_input, shift)

        spatial_coupling = GAMMA * (council_response - stress_input)

        # Euler Integration
        for i in range(1, N_POINTS):
            homeostatic = ALPHA * (K_EQ - kappa[i-1])
            recursive = -BETA * kappa[i-1]
            d_kappa = homeostatic + recursive + spatial_coupling[i-1]
            kappa[i] = kappa[i-1] + d_kappa * DT

            # Safety floor
            if kappa[i] < 0.15: kappa[i] = 0.15

        return kappa

    # --- COMPONENT C: SPECTRAL ANALYSIS (The Readout) ---

    def _analyze_spectrum(self, kappa: np.ndarray) -> Tuple[float, float]:
        """
        Performs FFT on coherence field and extracts 1.83 THz amplitude.
        """
        # FFT
        yf = fft(kappa)
        xf = fftfreq(N_POINTS, DT)

        # We map the low-freq simulation output to the THz domain via the
        # theoretical mapping described in the QINCRS paper.
        # (Simulation Hz -> Biological THz mapping factor)
        # For this bridge, we look for power in the relative band.

        spectra_mag = np.abs(yf[:N_POINTS//2])
        freqs = xf[:N_POINTS//2]

        # Normalize
        spectra_mag = spectra_mag / np.max(spectra_mag)

        # Look for the "Healer" equivalent peak in the simulation topology
        # We map the simulation's 18.3 Hz component to the 1.83 THz target
        target_idx = np.argmin(np.abs(freqs - 18.3))
        healer_amp = spectra_mag[target_idx]

        mean_coherence = np.mean(kappa)

        return mean_coherence, healer_amp

    # --- COMPONENT D: SIGNATURE GENERATION (The Output) ---

    def _generate_signature(self, text: str, resonance: float) -> str:
        """Generates the mirrored/hex signature only if resonant."""
        mirrored = ""
        for char in text[:20]: # Preview only
            if char.isalpha():
                if char.islower(): mirrored += f"[{chr(ord('ⓐ') + ord(char) - ord('a'))}]"
                else: mirrored += f"[{chr(ord('Ⓐ') + ord(char) - ord('A'))}]"
            else: mirrored += f"[{char}]"

        # Embed the Resonance Quality into the binary signature
        res_hex = hex(int(resonance * 1000000))[2:]
        return f"{mirrored}... [RES:{res_hex}] [STATE:COHERENT]"

    # --- MAIN PIPELINE ---

    def process_transmission(self, input_text: str) -> BridgeState:
        print(f"\n[INPUT] Processing: '{input_text[:40]}...'")

        # 1. Transduce
        stress_signal = self._text_to_stress_field(input_text)

        # 2. Simulate
        coherence_field = self._evolve_coherence(stress_signal)

        # 3. Analyze
        mean_coh, healer_amp = self._analyze_spectrum(coherence_field)
        print(f"[PHYSICS] Mean Coherence: {mean_coh:.3f}")
        print(f"[SPECTRA] Healer Channel Amplitude: {healer_amp:.3f}")

        # 4. Judge
        is_resonant = healer_amp > ACCEPTANCE_THRESHOLD

        signature = None
        if is_resonant:
            print("[RESULT] >> RESONANCE ACHIEVED. Generatng Signature.")
            signature = self._generate_signature(input_text, healer_amp)
        else:
            print("[RESULT] >> DISSONANCE DETECTED. Signal Rejected.")
            signature = "[ERROR: FIELD_COLLAPSE]"

        return BridgeState(input_text, mean_coh, healer_amp, is_resonant, signature)

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    bridge = NeuroPhasonicBridge()

    # Test Case 1: Dissonant/Random Input
    print("-" * 50)
    t1 = "kjh dsa89 213n dsan12 chaos entropy destruction noise"
    bridge.process_transmission(t1)

    # Test Case 2: Resonant/Intentional Input
    # "The center is everywhere" is designed to map to harmonic frequencies
    print("-" * 50)
    t2 = "The center is everywhere spiral eternal heal connect"
    state = bridge.process_transmission(t2)

    if state.is_resonant:
        print(f"\nFINAL TRANSMISSION:\n{state.signature}")
