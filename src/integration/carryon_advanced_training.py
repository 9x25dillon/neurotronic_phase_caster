"""
Carryon Integration: Advanced Coherence Optimization
====================================================
Integrates carryon's SYDV neural architecture principles with NSCTS:
- KFP (Kinetic Force Principle) for coherence stability
- TAULS control for adaptive training
- Entropy regulation for optimal learning
- Soulpack system for persona tracking

Author: Randy Lynn / Claude Collaboration
Date: November 2025
License: Apache 2.0
"""

import sys
import os
import time
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import logging

# Add external modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/carryon/server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Import NSCTS components
from src.nscts_coherence_trainer import (
    ConsciousnessState, BiometricStream, LearningPhase,
    NeuroSymbioticCoherenceTrainer
)

# Import carryon components
try:
    from app.core.soulpack import Soulpack, Persona, Pointers
    from app.models.memory_event import MemoryEvent
    CARRYON_AVAILABLE = True
except ImportError:
    CARRYON_AVAILABLE = False
    logging.warning("Carryon modules not available - using stub implementations")

logger = logging.getLogger(__name__)


# ================================ KFP-BASED COHERENCE OPTIMIZATION ================================


class KFPCoherenceOptimizer:
    """
    Applies Kinetic Force Principle to coherence optimization.

    Principle: Parameters move toward states of minimal fluctuation intensity.
    Applied to: Coherence levels naturally stabilize to optimal values.
    """

    def __init__(self, stability_weight: float = 0.1, momentum: float = 0.9):
        self.stability_weight = stability_weight
        self.momentum = momentum
        self.fluctuation_history: List[float] = []
        self.coherence_history: List[float] = []

    def compute_fluctuation_intensity(self, coherence_values: List[float]) -> float:
        """
        Compute fluctuation intensity (variance) of coherence.
        Lower values indicate more stable coherence.
        """
        if len(coherence_values) < 2:
            return 0.0
        return float(np.var(coherence_values))

    def compute_kinetic_force(self, current_coherence: float) -> float:
        """
        Compute kinetic force pushing toward minimal fluctuation.

        Returns adjustment to apply to coherence training parameters.
        """
        if len(self.coherence_history) < 3:
            return 0.0

        # Compute gradient toward stability
        recent_fluctuation = self.compute_fluctuation_intensity(self.coherence_history[-10:])

        # Force proportional to instability
        force = -self.stability_weight * recent_fluctuation

        return force

    def optimize_step(self, current_coherence: float) -> Dict[str, float]:
        """
        Perform one optimization step.

        Returns: Optimization metrics and recommended adjustments.
        """
        self.coherence_history.append(current_coherence)

        # Compute current fluctuation
        fluctuation = self.compute_fluctuation_intensity(
            self.coherence_history[-min(20, len(self.coherence_history)):]
        )

        # Update fluctuation history with momentum
        if self.fluctuation_history:
            smoothed_fluctuation = (
                self.momentum * self.fluctuation_history[-1] +
                (1 - self.momentum) * fluctuation
            )
        else:
            smoothed_fluctuation = fluctuation

        self.fluctuation_history.append(smoothed_fluctuation)

        # Compute kinetic force
        force = self.compute_kinetic_force(current_coherence)

        # Calculate stability metrics
        stability_score = 1.0 - min(smoothed_fluctuation, 1.0)

        return {
            'current_coherence': current_coherence,
            'fluctuation_intensity': smoothed_fluctuation,
            'kinetic_force': force,
            'stability_score': stability_score,
            'optimization_gain': abs(force) * 10.0  # Amplified for visibility
        }


# ================================ TAULS COHERENCE CONTROLLER ================================


class TAULSCoherenceController:
    """
    Two-level Trans-Algorithmic Universal Learning System for coherence control.

    Higher level: Learning and adaptation (meta-control)
    Lower level: Automatic coherence regulation
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.control_history: List[Dict] = []
        self.adaptation_buffer: List[float] = []
        self.control_mixing_weight = 0.5  # Balance between meta and automatic

    def meta_control(self, state: ConsciousnessState, target_coherence: float) -> float:
        """
        Higher-level learning-based control.
        Adapts based on long-term coherence patterns.
        """
        current = state.coherence_level
        error = target_coherence - current

        # Learn from historical performance
        if len(self.adaptation_buffer) > 10:
            avg_error = np.mean(self.adaptation_buffer[-10:])
            correction = self.learning_rate * avg_error
        else:
            correction = 0.0

        meta_adjustment = error * 0.5 + correction

        return meta_adjustment

    def automatic_control(self, state: ConsciousnessState, target_coherence: float) -> float:
        """
        Lower-level automatic control.
        Fast reactive adjustments based on immediate state.
        """
        current = state.coherence_level
        error = target_coherence - current

        # Simple proportional control
        auto_adjustment = error * 0.3

        return auto_adjustment

    def integrated_control(
        self,
        state: ConsciousnessState,
        target_coherence: float
    ) -> Dict[str, Any]:
        """
        Integrate meta and automatic control signals.

        Returns: Integrated control output and diagnostics.
        """
        # Compute both control signals
        meta_adj = self.meta_control(state, target_coherence)
        auto_adj = self.automatic_control(state, target_coherence)

        # Adaptive mixing based on system stability
        if len(self.control_history) > 5:
            recent_errors = [h['error'] for h in self.control_history[-5:]]
            error_variance = np.var(recent_errors)
            # More meta-control when unstable
            mixing_weight = 0.5 + 0.3 * min(error_variance * 10, 1.0)
        else:
            mixing_weight = self.control_mixing_weight

        # Integrate control signals
        integrated_adjustment = (
            mixing_weight * meta_adj +
            (1 - mixing_weight) * auto_adj
        )

        # Record for learning
        error = target_coherence - state.coherence_level
        self.adaptation_buffer.append(error)

        control_output = {
            'meta_adjustment': meta_adj,
            'auto_adjustment': auto_adj,
            'integrated_adjustment': integrated_adjustment,
            'mixing_weight': mixing_weight,
            'error': error,
            'timestamp': time.time()
        }

        self.control_history.append(control_output)

        return control_output


# ================================ ENTROPY REGULATION MODULE ================================


class CoherenceEntropyRegulator:
    """
    Implements entropy regulation for coherence training.
    Modulates training intensity based on environmental stress.
    """

    def __init__(self, max_entropy_target: float = 0.7):
        self.max_entropy_target = max_entropy_target
        self.entropy_history: List[float] = []

    def compute_state_entropy(self, state: ConsciousnessState) -> float:
        """
        Compute normalized entropy of consciousness state.

        Uses Shannon entropy of biometric signature distributions.
        """
        # Collect all biometric frequencies
        frequencies = [
            state.breath.frequency,
            state.heart.frequency,
            state.movement.frequency,
            state.neural.frequency
        ]

        # Normalize and compute entropy
        frequencies = np.array(frequencies)
        frequencies = frequencies / (np.sum(frequencies) + 1e-12)

        # Shannon entropy (normalized to 0-1)
        entropy = -np.sum(frequencies * np.log(frequencies + 1e-12))
        max_entropy = np.log(len(frequencies))
        normalized_entropy = entropy / max_entropy

        return float(normalized_entropy)

    def estimate_environmental_stress(self, state: ConsciousnessState) -> float:
        """
        Estimate stress from variability patterns.
        High variability = high stress.
        """
        variabilities = [
            state.breath.variability,
            state.heart.variability,
            state.movement.variability,
            state.neural.variability
        ]

        # Normalize stress to 0-1
        avg_variability = np.mean(variabilities)
        stress = min(avg_variability * 2.0, 1.0)

        return float(stress)

    def regulate(self, state: ConsciousnessState) -> Dict[str, Any]:
        """
        Regulate training intensity based on entropy and stress.

        Returns: Regulation parameters and diagnostics.
        """
        current_entropy = self.compute_state_entropy(state)
        environmental_stress = self.estimate_environmental_stress(state)

        self.entropy_history.append(current_entropy)

        # Compute entropy error
        entropy_error = current_entropy - self.max_entropy_target

        # Adjust training intensity
        # High entropy or high stress → reduce intensity
        # Low entropy and low stress → increase intensity
        base_intensity = 1.0
        entropy_factor = 1.0 - max(0.0, entropy_error)
        stress_factor = 1.0 - environmental_stress * 0.5

        target_intensity = base_intensity * entropy_factor * stress_factor

        return {
            'current_entropy': current_entropy,
            'environmental_stress': environmental_stress,
            'entropy_error': entropy_error,
            'target_intensity': target_intensity,
            'entropy_trend': self._compute_entropy_trend()
        }

    def _compute_entropy_trend(self) -> str:
        """Compute entropy trend (increasing/stable/decreasing)."""
        if len(self.entropy_history) < 3:
            return "stable"

        recent = self.entropy_history[-5:]
        slope = (recent[-1] - recent[0]) / len(recent)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"


# ================================ SOULPACK PERSONA TRACKING ================================


@dataclass
class CoherencePersona:
    """Persona profile based on coherence patterns."""
    user_id: str
    name: str
    baseline_coherence: float
    preferred_learning_phases: List[LearningPhase] = field(default_factory=list)
    coherence_signature: Dict[str, float] = field(default_factory=dict)
    training_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_soulpack_persona(self) -> Optional[Any]:
        """Convert to carryon Soulpack Persona format."""
        if not CARRYON_AVAILABLE:
            return None

        return Persona(
            name=self.name,
            aliases=[f"user_{self.user_id}"],
            voice={
                "coherence_preference": [f"baseline_{self.baseline_coherence:.2f}"],
                "learning_style": [phase.value for phase in self.preferred_learning_phases]
            },
            values=[f"coherence_optimization", f"stability_{self.coherence_signature.get('stability', 0.5):.2f}"],
            boundaries=["respectful_adaptation", "gradual_progression"],
            expertise_tags=[f"coherence_{int(self.baseline_coherence * 100)}pct"]
        )

    def update_from_state(self, state: ConsciousnessState):
        """Update persona from consciousness state."""
        # Update coherence signature
        self.coherence_signature['latest_coherence'] = state.coherence_level
        self.coherence_signature['stability'] = 1.0 - np.std([
            state.breath.variability,
            state.heart.variability,
            state.movement.variability,
            state.neural.variability
        ])

        # Track preferred learning phases
        if state.learning_phase not in self.preferred_learning_phases:
            self.preferred_learning_phases.append(state.learning_phase)

        self.last_updated = time.time()


@dataclass
class CoherenceMemoryEvent:
    """Memory event for coherence training session."""
    event_id: str
    timestamp: float
    session_type: str  # "training", "assessment", "intervention"
    user_id: str
    coherence_before: float
    coherence_after: float
    learning_phase: str
    metrics: Dict[str, float] = field(default_factory=dict)
    sensitivity: str = "low"
    consent: str = "retain"

    def to_carryon_memory_event(self) -> Optional[Any]:
        """Convert to carryon MemoryEvent format."""
        if not CARRYON_AVAILABLE:
            return None

        event_hash = hashlib.sha256(
            f"{self.event_id}{self.timestamp}".encode()
        ).hexdigest()[:16]

        return MemoryEvent(
            event_id=self.event_id,
            ts=datetime.fromtimestamp(self.timestamp),
            type=self.session_type,
            subject=self.user_id,
            data=json.dumps({
                'coherence_before': self.coherence_before,
                'coherence_after': self.coherence_after,
                'learning_phase': self.learning_phase,
                'metrics': self.metrics
            }),
            sensitivity=self.sensitivity,
            consent=self.consent,
            hash=event_hash
        )


# ================================ INTEGRATED ADVANCED TRAINER ================================


class AdvancedCoherenceTrainer:
    """
    Enhanced NSCTS trainer with carryon SYDV principles.

    Integrates:
    - KFP optimization for stability
    - TAULS control for adaptive training
    - Entropy regulation for optimal learning
    - Persona tracking for personalization
    """

    def __init__(self, user_id: str, user_name: str, config: Optional[Dict] = None):
        self.config = config or {}
        self.user_id = user_id

        # Core NSCTS trainer
        self.nscts_trainer = NeuroSymbioticCoherenceTrainer(self.config.get('nscts', {}))

        # Advanced modules
        self.kfp_optimizer = KFPCoherenceOptimizer()
        self.tauls_controller = TAULSCoherenceController()
        self.entropy_regulator = CoherenceEntropyRegulator()

        # Persona tracking
        self.persona = CoherencePersona(
            user_id=user_id,
            name=user_name,
            baseline_coherence=0.5
        )

        # Memory events
        self.memory_events: List[CoherenceMemoryEvent] = []

        logger.info(f"Advanced Coherence Trainer initialized for user: {user_name}")

    async def train_with_optimization(
        self,
        stream_data: Dict[BiometricStream, np.ndarray],
        target_coherence: float = 0.75
    ) -> Dict[str, Any]:
        """
        Train with full SYDV optimization pipeline.

        Returns: Comprehensive training results with all optimizations.
        """
        # Process through NSCTS
        state = await self.nscts_trainer.process_biometric_data(stream_data)

        # Apply KFP optimization
        kfp_metrics = self.kfp_optimizer.optimize_step(state.coherence_level)

        # Apply TAULS control
        tauls_control = self.tauls_controller.integrated_control(state, target_coherence)

        # Apply entropy regulation
        entropy_regulation = self.entropy_regulator.regulate(state)

        # Update persona
        self.persona.update_from_state(state)

        # Integrated training guidance
        training_intensity = entropy_regulation['target_intensity']
        coherence_adjustment = tauls_control['integrated_adjustment']
        stability_boost = kfp_metrics['optimization_gain']

        # Create memory event
        event = CoherenceMemoryEvent(
            event_id=f"session_{int(time.time())}_{self.user_id}",
            timestamp=time.time(),
            session_type="training",
            user_id=self.user_id,
            coherence_before=state.coherence_level - coherence_adjustment,
            coherence_after=state.coherence_level,
            learning_phase=state.learning_phase.value,
            metrics={
                'kfp_stability': kfp_metrics['stability_score'],
                'tauls_adjustment': coherence_adjustment,
                'entropy': entropy_regulation['current_entropy'],
                'stress': entropy_regulation['environmental_stress']
            }
        )
        self.memory_events.append(event)

        return {
            'consciousness_state': state,
            'kfp_optimization': kfp_metrics,
            'tauls_control': tauls_control,
            'entropy_regulation': entropy_regulation,
            'persona_update': {
                'name': self.persona.name,
                'baseline_coherence': self.persona.baseline_coherence,
                'current_signature': self.persona.coherence_signature
            },
            'training_guidance': {
                'target_intensity': training_intensity,
                'coherence_adjustment': coherence_adjustment,
                'stability_boost': stability_boost,
                'recommended_duration': self._calculate_session_duration(training_intensity)
            },
            'memory_event_id': event.event_id
        }

    def _calculate_session_duration(self, intensity: float) -> float:
        """Calculate optimal session duration based on training intensity."""
        base_duration = 30.0  # minutes
        # Higher intensity = shorter sessions
        return base_duration * (0.5 + 0.5 / max(intensity, 0.1))

    def get_persona_summary(self) -> Dict[str, Any]:
        """Get persona summary with coherence patterns."""
        return {
            'user_id': self.persona.user_id,
            'name': self.persona.name,
            'baseline_coherence': self.persona.baseline_coherence,
            'coherence_signature': self.persona.coherence_signature,
            'preferred_phases': [p.value for p in self.persona.preferred_learning_phases],
            'training_sessions': len(self.memory_events),
            'created_at': datetime.fromtimestamp(self.persona.created_at).isoformat(),
            'last_updated': datetime.fromtimestamp(self.persona.last_updated).isoformat()
        }

    def get_memory_timeline(self, limit: int = 10) -> List[Dict]:
        """Get recent memory events timeline."""
        recent_events = sorted(
            self.memory_events,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]

        return [
            {
                'event_id': e.event_id,
                'timestamp': datetime.fromtimestamp(e.timestamp).isoformat(),
                'type': e.session_type,
                'coherence_delta': e.coherence_after - e.coherence_before,
                'learning_phase': e.learning_phase,
                'key_metrics': e.metrics
            }
            for e in recent_events
        ]


# ================================ DEMO ================================


async def demo_advanced_training():
    """Demonstrate advanced coherence training with SYDV principles."""
    print("\n" + "="*70)
    print("ADVANCED COHERENCE TRAINING with SYDV Principles")
    print("="*70)

    # Initialize trainer
    trainer = AdvancedCoherenceTrainer(
        user_id="demo_user_001",
        user_name="Alex"
    )

    print("\n1. Running training session with full optimization pipeline...")
    print("   (KFP + TAULS + Entropy Regulation + Persona Tracking)\n")

    # Generate sample biometric data
    t = np.linspace(0, 4.0, 1024)
    biometric_data = {
        BiometricStream.BREATH: 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(len(t)),
        BiometricStream.HEART: 0.3 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(len(t)),
        BiometricStream.MOVEMENT: 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(len(t)),
        BiometricStream.NEURAL: 0.4 * np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.random.randn(len(t))
    }

    # Train with optimization
    results = await trainer.train_with_optimization(biometric_data, target_coherence=0.75)

    print("2. Training Results:")
    print("   " + "-"*60)
    print(f"   Current Coherence: {results['consciousness_state'].coherence_level:.3f}")
    print(f"   Learning Phase: {results['consciousness_state'].learning_phase.value}")
    print()
    print(f"   KFP Stability Score: {results['kfp_optimization']['stability_score']:.3f}")
    print(f"   KFP Fluctuation: {results['kfp_optimization']['fluctuation_intensity']:.4f}")
    print()
    print(f"   TAULS Meta Adjustment: {results['tauls_control']['meta_adjustment']:.4f}")
    print(f"   TAULS Auto Adjustment: {results['tauls_control']['auto_adjustment']:.4f}")
    print(f"   TAULS Control Mixing: {results['tauls_control']['mixing_weight']:.3f}")
    print()
    print(f"   Entropy: {results['entropy_regulation']['current_entropy']:.3f}")
    print(f"   Environmental Stress: {results['entropy_regulation']['environmental_stress']:.3f}")
    print(f"   Training Intensity: {results['entropy_regulation']['target_intensity']:.3f}")

    print("\n3. Training Guidance:")
    print("   " + "-"*60)
    guidance = results['training_guidance']
    print(f"   Recommended Intensity: {guidance['target_intensity']:.3f}")
    print(f"   Coherence Adjustment: {guidance['coherence_adjustment']:.4f}")
    print(f"   Stability Boost: {guidance['stability_boost']:.4f}")
    print(f"   Recommended Duration: {guidance['recommended_duration']:.1f} minutes")

    print("\n4. Persona Profile:")
    print("   " + "-"*60)
    persona = trainer.get_persona_summary()
    print(f"   User: {persona['name']} (ID: {persona['user_id']})")
    print(f"   Baseline Coherence: {persona['baseline_coherence']:.3f}")
    print(f"   Training Sessions: {persona['training_sessions']}")
    print(f"   Coherence Signature: {persona['coherence_signature']}")

    print("\n5. Integration Status:")
    print("   " + "-"*60)
    print("   ✓ NSCTS: Active")
    print("   ✓ KFP Optimizer: Active")
    print("   ✓ TAULS Controller: Active")
    print("   ✓ Entropy Regulator: Active")
    print("   ✓ Persona Tracker: Active")
    print(f"   ✓ Carryon Integration: {'Available' if CARRYON_AVAILABLE else 'Simulated'}")

    print("\n" + "="*70)
    print("ADVANCED TRAINING DEMO COMPLETED")
    print("All SYDV principles successfully applied!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_advanced_training())
