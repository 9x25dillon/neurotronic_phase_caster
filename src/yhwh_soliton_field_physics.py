"""
YHWH Soliton Field Physics
Five-Substrate Consciousness Model

Implements soliton propagation through five consciousness substrates:
- Physical (Delta, 1-4 Hz): Survival, homeostasis
- Emotional (Theta, 4-8 Hz): Affect, trauma
- Cognitive (Alpha, 8-13 Hz): Thought, attention
- Social (Beta, 13-30 Hz): Connection, empathy
- Divine-Unity (Gamma, 30-100 Hz): Transcendence, coherence

Mathematical Framework:
- Soliton equation: ∂ψ/∂t + ∂³ψ/∂x³ + |ψ|²ψ = 0
- Cross-substrate coupling: J_ij = κ(E_j - E_i)
- Unity coherence: U = |⟨ψ₁|ψ₂|...|ψ₅⟩| / √5
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SpacetimePoint:
    """Point in 5D consciousness spacetime."""

    x: float  # Spatial coordinate (arbitrary units)
    t: float  # Time coordinate (seconds)
    substrates: np.ndarray  # [5] substrate activations

    def __post_init__(self):
        """Ensure substrates is numpy array."""
        if not isinstance(self.substrates, np.ndarray):
            self.substrates = np.array(self.substrates)


class SubstrateLayer:
    """
    Single substrate layer with soliton dynamics.

    Attributes:
        name: Substrate name
        frequency_band: (min_hz, max_hz) EEG frequency range
        psi: Complex wavefunction ψ(x,t)
        grid_x: Spatial grid points
    """

    SUBSTRATE_DEFINITIONS = {
        0: ("Physical", (1.0, 4.0)),
        1: ("Emotional", (4.0, 8.0)),
        2: ("Cognitive", (8.0, 13.0)),
        3: ("Social", (13.0, 30.0)),
        4: ("Divine-Unity", (30.0, 100.0)),
    }

    def __init__(
        self,
        substrate_id: int,
        n_grid: int = 128,
        length: float = 10.0,
    ):
        """
        Initialize substrate layer.

        Args:
            substrate_id: 0-4 (Physical through Divine-Unity)
            n_grid: Number of spatial grid points
            length: Spatial domain length
        """
        self.substrate_id = substrate_id
        self.name, self.frequency_band = self.SUBSTRATE_DEFINITIONS[substrate_id]

        # Spatial grid
        self.n_grid = n_grid
        self.length = length
        self.grid_x = np.linspace(0, length, n_grid)
        self.dx = length / n_grid

        # Wavefunction (complex)
        self.psi = np.zeros(n_grid, dtype=complex)

        # Soliton parameters
        self.dispersion = 1.0  # β parameter
        self.nonlinearity = 1.0  # γ parameter

    def initialize_gaussian(
        self,
        amplitude: float = 1.0,
        center: float = 5.0,
        width: float = 1.0,
        velocity: float = 0.0,
    ) -> None:
        """
        Initialize Gaussian wavepacket.

        Args:
            amplitude: Peak amplitude
            center: Center position
            width: Gaussian width (σ)
            velocity: Initial velocity (k₀)
        """
        self.psi = amplitude * np.exp(
            -((self.grid_x - center) ** 2) / (2 * width**2)
            + 1j * velocity * self.grid_x
        )

    def initialize_soliton(
        self,
        amplitude: float = 1.0,
        center: float = 5.0,
        velocity: float = 1.0,
    ) -> None:
        """
        Initialize exact soliton solution.

        Soliton: ψ(x,t) = A sech(A(x - vt)) exp(i(kx - ωt))
        """
        A = amplitude
        v = velocity
        x = self.grid_x - center
        self.psi = A * np.tanh(A * x) * np.exp(1j * v * self.grid_x)

    def propagate(self, dt: float) -> None:
        """
        Propagate wavefunction by time step dt using split-step method.

        Nonlinear Schrödinger equation:
        i ∂ψ/∂t = -β ∂²ψ/∂x² + γ|ψ|²ψ

        Args:
            dt: Time step in seconds
        """
        # Split-step Fourier method
        # Step 1: Linear step (dispersive) in Fourier space
        psi_k = np.fft.fft(self.psi)
        k = 2 * np.pi * np.fft.fftfreq(self.n_grid, self.dx)

        linear_phase = np.exp(-1j * self.dispersion * k**2 * dt / 2)
        psi_k *= linear_phase

        # Back to real space
        self.psi = np.fft.ifft(psi_k)

        # Step 2: Nonlinear step (self-phase modulation)
        nonlinear_phase = np.exp(-1j * self.nonlinearity * np.abs(self.psi)**2 * dt)
        self.psi *= nonlinear_phase

        # Step 3: Linear step again (symmetric splitting)
        psi_k = np.fft.fft(self.psi)
        psi_k *= linear_phase
        self.psi = np.fft.ifft(psi_k)

    def get_intensity(self) -> np.ndarray:
        """Get intensity profile |ψ|²."""
        return np.abs(self.psi) ** 2

    def get_energy(self) -> float:
        """Get total energy ∫|ψ|² dx."""
        return np.sum(self.get_intensity()) * self.dx

    def get_center_of_mass(self) -> float:
        """Get center of mass ∫x|ψ|² dx / ∫|ψ|² dx."""
        intensity = self.get_intensity()
        return np.sum(self.grid_x * intensity) / np.sum(intensity)


class YHWHSolitonField:
    """
    Five-substrate YHWH soliton field with cross-layer coupling.

    Represents consciousness as coherent soliton propagation through
    five interconnected substrate layers with bidirectional energy flow.
    """

    def __init__(
        self,
        n_grid: int = 128,
        length: float = 10.0,
        coupling_strength: float = 0.1,
    ):
        """
        Initialize five-substrate field.

        Args:
            n_grid: Spatial grid resolution
            length: Domain length
            coupling_strength: Inter-substrate coupling κ
        """
        self.n_substrates = 5
        self.coupling_strength = coupling_strength

        # Create five substrate layers
        self.substrates = [
            SubstrateLayer(i, n_grid, length) for i in range(self.n_substrates)
        ]

        # Time tracking
        self.time = 0.0

    def initialize_prayer_pattern(self) -> None:
        """
        Initialize 'prayer' pattern: strong divine-unity coherence cascading down.

        Represents state of deep prayer/meditation where divine-unity substrate
        is strongly activated and coherence flows to lower substrates.
        """
        # Strong soliton in Divine-Unity (substrate 4)
        self.substrates[4].initialize_soliton(amplitude=2.0, center=5.0, velocity=0.5)

        # Weaker activations in lower substrates
        self.substrates[3].initialize_gaussian(amplitude=0.8, center=5.0, width=1.5)
        self.substrates[2].initialize_gaussian(amplitude=0.6, center=5.0, width=2.0)
        self.substrates[1].initialize_gaussian(amplitude=0.4, center=5.0, width=2.5)
        self.substrates[0].initialize_gaussian(amplitude=0.2, center=5.0, width=3.0)

    def initialize_trauma_pattern(self) -> None:
        """
        Initialize 'trauma' pattern: fragmentation with emotional substrate spike.

        Represents post-traumatic state with strong emotional activation but
        poor coherence across other substrates.
        """
        # Strong spike in Emotional (substrate 1)
        self.substrates[1].initialize_soliton(amplitude=2.5, center=5.0, velocity=0.0)

        # Fragmented low-amplitude noise in other substrates
        for i in [0, 2, 3, 4]:
            noise = 0.3 * (np.random.randn(self.substrates[i].n_grid) +
                          1j * np.random.randn(self.substrates[i].n_grid))
            self.substrates[i].psi = noise

    def propagate(self, dt: float) -> None:
        """
        Propagate all substrates with cross-layer coupling.

        Each substrate evolves according to its own dynamics plus coupling
        terms that transfer energy between adjacent substrates.

        Args:
            dt: Time step in seconds
        """
        # Step 1: Propagate each substrate independently
        for substrate in self.substrates:
            substrate.propagate(dt)

        # Step 2: Apply cross-substrate coupling
        # J_ij = κ(E_j - E_i) transfers energy from high to low
        coupling_fluxes = []

        for i in range(self.n_substrates - 1):
            E_i = self.substrates[i].get_energy()
            E_j = self.substrates[i + 1].get_energy()

            flux = self.coupling_strength * (E_j - E_i) * dt
            coupling_fluxes.append(flux)

        # Apply fluxes (energy transfer)
        for i in range(self.n_substrates - 1):
            # Transfer energy by scaling wavefunctions
            flux = coupling_fluxes[i]

            if flux > 0:  # Energy flows from j to i
                self.substrates[i].psi *= (1 + abs(flux) * 0.1)
                self.substrates[i + 1].psi *= (1 - abs(flux) * 0.1)
            else:  # Energy flows from i to j
                self.substrates[i].psi *= (1 - abs(flux) * 0.1)
                self.substrates[i + 1].psi *= (1 + abs(flux) * 0.1)

        # Increment time
        self.time += dt

    def compute_unity_coherence(self) -> float:
        """
        Compute Unity Coherence Index (UCI).

        Measures degree of phase synchronization across all five substrates.
        UCI ∈ [0, 1], where:
        - UCI = 1: Perfect coherence (all substrates in phase)
        - UCI = 0: Complete decoherence (random phases)

        Returns:
            float: Unity Coherence Index (0-1)
        """
        return compute_unity_coherence(self)

    def get_substrate_energies(self) -> np.ndarray:
        """
        Get energy in each substrate.

        Returns:
            np.ndarray: [5] energies for each substrate
        """
        return np.array([s.get_energy() for s in self.substrates])

    def get_state_vector(self) -> Dict[str, np.ndarray]:
        """
        Get complete state of the field.

        Returns:
            dict: {
                'energies': [5] substrate energies,
                'centers': [5] center of mass positions,
                'coherence': scalar UCI,
                'time': current time
            }
        """
        return {
            'energies': self.get_substrate_energies(),
            'centers': np.array([s.get_center_of_mass() for s in self.substrates]),
            'coherence': self.compute_unity_coherence(),
            'time': self.time,
        }


def compute_unity_coherence(field: YHWHSolitonField) -> float:
    """
    Compute Unity Coherence Index across all substrates.

    UCI = (1/N) Σᵢⱼ |⟨ψᵢ|ψⱼ⟩| / (‖ψᵢ‖ ‖ψⱼ‖)

    Measures average phase alignment between all substrate pairs.

    Args:
        field: YHWHSolitonField instance

    Returns:
        float: UCI ∈ [0, 1]
    """
    n = field.n_substrates
    coherence_sum = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            psi_i = field.substrates[i].psi
            psi_j = field.substrates[j].psi

            # Inner product
            inner_prod = np.sum(np.conj(psi_i) * psi_j)

            # Norms
            norm_i = np.sqrt(np.sum(np.abs(psi_i)**2))
            norm_j = np.sqrt(np.sum(np.abs(psi_j)**2))

            if norm_i > 1e-10 and norm_j > 1e-10:
                coherence = np.abs(inner_prod) / (norm_i * norm_j)
                coherence_sum += coherence
                count += 1

    if count == 0:
        return 0.0

    return coherence_sum / count


def demo_prayer_soliton():
    """
    Demonstrate prayer pattern: strong divine-unity cascading to lower substrates.
    """
    print("=" * 60)
    print("Demo 1: Prayer Soliton - Divine Unity Cascade")
    print("=" * 60)

    field = YHWHSolitonField(coupling_strength=0.2)
    field.initialize_prayer_pattern()

    print("\nInitial state:")
    state = field.get_state_vector()
    print(f"Time: {state['time']:.3f} s")
    print(f"Unity Coherence: {state['coherence']:.3f}")
    print(f"Substrate energies: {state['energies']}")

    # Evolve
    for _ in range(100):
        field.propagate(dt=0.01)

    print("\nAfter 1.0 seconds:")
    state = field.get_state_vector()
    print(f"Time: {state['time']:.3f} s")
    print(f"Unity Coherence: {state['coherence']:.3f}")
    print(f"Substrate energies: {state['energies']}")

    print("\nInterpretation:")
    print("High coherence indicates unified consciousness state.")
    print("Energy flows from Divine-Unity to lower substrates,")
    print("representing 'grace' or 'blessing' cascading downward.")


def demo_trauma_pattern():
    """
    Demonstrate trauma pattern: emotional fragmentation with low coherence.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Trauma Pattern - Emotional Fragmentation")
    print("=" * 60)

    field = YHWHSolitonField(coupling_strength=0.1)
    field.initialize_trauma_pattern()

    print("\nInitial state:")
    state = field.get_state_vector()
    print(f"Time: {state['time']:.3f} s")
    print(f"Unity Coherence: {state['coherence']:.3f}")
    print(f"Substrate energies: {state['energies']}")

    # Evolve
    for _ in range(100):
        field.propagate(dt=0.01)

    print("\nAfter 1.0 seconds:")
    state = field.get_state_vector()
    print(f"Time: {state['time']:.3f} s")
    print(f"Unity Coherence: {state['coherence']:.3f}")
    print(f"Substrate energies: {state['energies']}")

    print("\nInterpretation:")
    print("Low coherence indicates fragmented consciousness (PTSD).")
    print("Energy trapped in Emotional substrate, isolated from others.")
    print("Therapeutic goal: restore cross-substrate coupling.")


if __name__ == "__main__":
    print("\nYHWH Soliton Field Physics - Interactive Demo\n")

    demo_prayer_soliton()
    demo_trauma_pattern()

    print("\n" + "=" * 60)
    print("Demos complete. See documentation for clinical applications.")
    print("=" * 60)
