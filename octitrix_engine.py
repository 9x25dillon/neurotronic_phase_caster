"""
Octitrix Engine - 8D Audio Structure Generator
Generates fractal and geometric structures for spatial audio manipulation
"""

import numpy as np
import random
from typing import List, Dict, Any


class Octitrix:
    """
    Core engine for generating 8-dimensional audio structures.
    Combines chaos theory, fractal mathematics, and euclidean rhythms
    to create spatial audio patterns.
    """

    def __init__(self, dimensions: int = 8):
        """
        Initialize the Octitrix engine.

        Args:
            dimensions: Number of spatial dimensions (default 8 for 8D audio)
        """
        self.dimensions = dimensions
        self.current_matrix = None
        self.state = {
            "last_update": None,
            "type": None,
            "params": {}
        }

    def generate_fractal(self, seed: int = 0) -> List[List[float]]:
        """
        Generate a chaotic/fractal based spatial structure.
        Uses logistic map and strange attractors to create complex patterns.

        Args:
            seed: Random seed for reproducible chaos

        Returns:
            8D matrix representing spatial positions
        """
        random.seed(seed)
        np.random.seed(seed)

        # Logistic map parameters (chaos theory)
        r = 3.9  # Chaotic regime
        iterations = self.dimensions * 10

        # Generate chaotic sequence using logistic map
        x = random.random()
        chaotic_sequence = []

        for _ in range(iterations):
            x = r * x * (1 - x)
            chaotic_sequence.append(x)

        # Reshape into 8D structure
        matrix = []
        for i in range(self.dimensions):
            row = []
            for j in range(self.dimensions):
                idx = i * self.dimensions + j
                if idx < len(chaotic_sequence):
                    value = chaotic_sequence[idx]
                else:
                    value = np.random.random()
                row.append(round(value, 4))
            matrix.append(row)

        self.current_matrix = matrix
        self.state = {
            "last_update": "now",
            "type": "fractal",
            "params": {"seed": seed}
        }

        return matrix

    def generate_euclidean(self, pulses: int = 4, steps: int = None) -> List[List[float]]:
        """
        Generate a euclidean rhythm based geometric structure.
        Uses Bjorklund's algorithm for rhythmic distribution.

        Args:
            pulses: Number of active pulses
            steps: Total steps (defaults to 8 for each dimension)

        Returns:
            8D matrix with euclidean rhythm patterns
        """
        if steps is None:
            steps = self.dimensions

        # Generate euclidean rhythm using Bjorklund's algorithm
        def euclidean_rhythm(k, n):
            """Generate euclidean rhythm with k pulses over n steps."""
            if k >= n:
                return [1] * n

            pattern = []
            counts = []
            remainders = []
            divisor = n - k

            remainders.append(k)
            level = 0

            while True:
                counts.append(divisor // remainders[level])
                remainders.append(divisor % remainders[level])
                divisor = remainders[level]
                level += 1
                if remainders[level] <= 1:
                    break

            counts.append(divisor)

            def build(level):
                if level == -1:
                    pattern.append(0)
                elif level == -2:
                    pattern.append(1)
                else:
                    for _ in range(counts[level]):
                        build(level - 1)
                    if remainders[level] != 0:
                        build(level - 2)

            build(level)
            return pattern[:n]

        # Create 8D matrix with euclidean patterns
        matrix = []
        for i in range(self.dimensions):
            row = []
            # Vary the pulse count slightly for each dimension
            dim_pulses = max(1, pulses + (i - self.dimensions // 2))
            rhythm = euclidean_rhythm(dim_pulses, steps)

            for j in range(self.dimensions):
                if j < len(rhythm):
                    value = float(rhythm[j])
                else:
                    value = 0.0
                row.append(value)
            matrix.append(row)

        self.current_matrix = matrix
        self.state = {
            "last_update": "now",
            "type": "euclidean",
            "params": {"pulses": pulses, "steps": steps}
        }

        return matrix

    def dispatch(self) -> Dict[str, Any]:
        """
        Dispatch current state to DAW/external systems.
        In a real implementation, this would send OSC/MIDI messages.

        Returns:
            Current state dictionary
        """
        if self.current_matrix is None:
            return {
                "status": "no_data",
                "message": "No structure generated yet"
            }

        # In production, send to DAW via OSC or MIDI
        # For now, return the state
        return {
            "status": "dispatched",
            "matrix": self.current_matrix,
            "state": self.state,
            "dimensions": self.dimensions
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return {
            "dimensions": self.dimensions,
            "current_state": self.state,
            "has_matrix": self.current_matrix is not None
        }

    def reset(self):
        """Reset the engine to initial state."""
        self.current_matrix = None
        self.state = {
            "last_update": None,
            "type": None,
            "params": {}
        }


if __name__ == "__main__":
    # Demo usage
    print("=== Octitrix Engine Demo ===\n")

    engine = Octitrix()

    print("1. Generating Fractal Structure:")
    fractal = engine.generate_fractal(seed=42)
    print(f"   Generated {len(fractal)}x{len(fractal[0])} matrix")
    print(f"   Sample: {fractal[0][:4]}...")

    print("\n2. Generating Euclidean Rhythm Structure:")
    euclidean = engine.generate_euclidean(pulses=3)
    print(f"   Generated {len(euclidean)}x{len(euclidean[0])} matrix")
    print(f"   Sample: {euclidean[0]}")

    print("\n3. Dispatching state:")
    result = engine.dispatch()
    print(f"   Status: {result['status']}")
    print(f"   Type: {result['state']['type']}")
