"""
NSCTS-LiMp-Eopiez Integration Bridge
====================================
Unified pipeline integrating:
- NSCTS: NeuroSymbiotic Coherence Training System
- LiMp: Entropy Engine for token transformation
- Eopiez: Message Vectorizer for symbolic state representation

Author: Randy Lynn / Claude Collaboration
Date: November 2025
License: Open Source
"""

import sys
import os
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
import time

# Add external modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/LiMp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Import NSCTS components
from src.nscts_coherence_trainer import (
    BiometricSignature, ConsciousnessState, BiometricStream,
    NeuroSymbioticCoherenceTrainer, LearningPhase, CoherenceState
)

# Import LiMp components
from entropy_engine.core import Token, EntropyNode, EntropyEngine

# Import Eopiez components
try:
    from .eopiez_vectorizer import (
        IntegratedEopiezVectorizer,
        BiometricMotifToken,
        VectorizedBiometricState
    )
except ImportError:
    # When run as main script
    from eopiez_vectorizer import (
        IntegratedEopiezVectorizer,
        BiometricMotifToken,
        VectorizedBiometricState
    )

logger = logging.getLogger(__name__)


# ================================ ADAPTERS ================================


@dataclass
class EnhancedBiometricState:
    """Extended biometric state with entropy transformations and symbolic representation."""
    original_state: ConsciousnessState
    entropy_tokens: Dict[BiometricStream, Token] = field(default_factory=dict)
    entropy_traces: Dict[BiometricStream, List[float]] = field(default_factory=dict)
    symbolic_vector: Optional[np.ndarray] = None
    motif_analysis: Optional[Dict] = None
    information_density: float = 0.0
    timestamp: float = field(default_factory=time.time)


class BiometricToTokenAdapter:
    """Converts NSCTS BiometricSignatures to LiMp Tokens."""

    @staticmethod
    def signature_to_token(signature: BiometricSignature) -> Token:
        """
        Convert a biometric signature into an entropy token.

        Token value encodes: frequency, amplitude, phase, complexity
        """
        # Create a composite value encoding all signature properties
        composite_value = {
            "stream": signature.stream.value,
            "frequency": signature.frequency,
            "amplitude": signature.amplitude,
            "variability": signature.variability,
            "phase": signature.phase,
            "complexity": signature.complexity,
            "timestamp": signature.timestamp
        }

        # Serialize to string for token
        token_value = json.dumps(composite_value, sort_keys=True)
        return Token(token_value)

    @staticmethod
    def token_to_signature(token: Token, stream: BiometricStream) -> BiometricSignature:
        """Convert a token back to biometric signature (after transformations)."""
        try:
            value_dict = json.loads(token.value)
            return BiometricSignature(
                stream=stream,
                frequency=value_dict.get("frequency", 0.0),
                amplitude=value_dict.get("amplitude", 0.0),
                variability=value_dict.get("variability", 0.0),
                phase=value_dict.get("phase", 0.0),
                complexity=value_dict.get("complexity", 0.0),
                timestamp=value_dict.get("timestamp", time.time())
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to decode token, using defaults: {e}")
            return BiometricSignature(
                stream=stream,
                frequency=1.0,
                amplitude=0.5,
                variability=0.1,
                phase=0.0,
                complexity=0.5,
                timestamp=time.time()
            )


class ConsciousnessToMotifAdapter:
    """Converts ConsciousnessState to Eopiez motif tokens."""

    @staticmethod
    def state_to_motif_tokens(state: ConsciousnessState) -> List[Dict]:
        """
        Convert consciousness state to motif tokens for Eopiez vectorization.

        Each biometric signature becomes a motif token.
        """
        motif_tokens = []

        # Generate motif tokens from each biometric stream
        for signature in [state.breath, state.heart, state.movement, state.neural]:
            motif_token = {
                "type": signature.stream.value,
                "frequency": signature.frequency,
                "amplitude": signature.amplitude,
                "phase": signature.phase,
                "complexity": signature.complexity,
                "entropy": signature.complexity  # Use complexity as entropy proxy
            }
            motif_tokens.append(motif_token)

        # Add coherence state as a meta-motif
        motif_tokens.append({
            "type": "coherence_state",
            "coherence_level": state.coherence_level,
            "learning_phase": state.learning_phase.value,
            "entropy": 1.0 - state.coherence_level  # Higher coherence = lower entropy
        })

        return motif_tokens


# ================================ ENTROPY TRANSFORMATIONS ================================


class BiometricEntropyTransformations:
    """Entropy-based transformations for biometric signals."""

    @staticmethod
    def harmonic_enhance(value: str, entropy: float) -> str:
        """Enhance harmonic components based on entropy."""
        try:
            data = json.loads(value)
            # Boost frequency coherence when entropy is low
            if entropy < 7.5:  # Low entropy = high coherence
                data["frequency"] *= 1.05
                data["amplitude"] *= 1.02
            return json.dumps(data, sort_keys=True)
        except:
            return value

    @staticmethod
    def phase_align(value: str, entropy: float) -> str:
        """Align phase relationships based on entropy."""
        try:
            data = json.loads(value)
            # Normalize phase toward coherent states
            if data.get("phase"):
                target_phase = np.pi if entropy > 7.5 else 0
                data["phase"] = 0.9 * data["phase"] + 0.1 * target_phase
            return json.dumps(data, sort_keys=True)
        except:
            return value

    @staticmethod
    def complexity_filter(value: str, entropy: float) -> str:
        """Filter complexity based on entropy threshold."""
        try:
            data = json.loads(value)
            # Reduce complexity noise when entropy is high
            if entropy > 8.0 and data.get("complexity"):
                data["complexity"] *= 0.95
            return json.dumps(data, sort_keys=True)
        except:
            return value


# ================================ INTEGRATED PIPELINE ================================


class NSCTSLiMpEopiezPipeline:
    """
    Complete integration pipeline:
    Raw Biometrics → NSCTS → LiMp Entropy → Eopiez Vectorization → Enhanced Training
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize NSCTS trainer
        self.nscts_trainer = NeuroSymbioticCoherenceTrainer(self.config.get('nscts', {}))

        # Initialize LiMp entropy engine
        self.entropy_engine = self._build_entropy_engine()

        # Initialize Eopiez vectorizer (real implementation)
        eopiez_config = self.config.get('eopiez', {})
        self.eopiez_vectorizer = IntegratedEopiezVectorizer(
            embedding_dim=eopiez_config.get('embedding_dim', 64),
            julia_url=eopiez_config.get('julia_url', 'http://localhost:9000')
        )

        # Adapters
        self.token_adapter = BiometricToTokenAdapter()
        self.motif_adapter = ConsciousnessToMotifAdapter()

        # State tracking
        self.enhanced_states: List[EnhancedBiometricState] = []

        logger.info("NSCTS-LiMp-Eopiez Pipeline initialized (Real Eopiez vectorization enabled)")

    def _build_entropy_engine(self) -> EntropyEngine:
        """Build the LiMp entropy transformation graph."""
        # Root node: harmonic enhancement
        root = EntropyNode(
            name="harmonic_enhance",
            transform_function=BiometricEntropyTransformations.harmonic_enhance,
            entropy_limit=10.0
        )

        # Child nodes: phase alignment and complexity filtering
        phase_node = EntropyNode(
            name="phase_align",
            transform_function=BiometricEntropyTransformations.phase_align,
            entropy_limit=10.0
        )

        complexity_node = EntropyNode(
            name="complexity_filter",
            transform_function=BiometricEntropyTransformations.complexity_filter,
            entropy_limit=10.0
        )

        root.add_child(phase_node)
        root.add_child(complexity_node)

        return EntropyEngine(root, max_depth=3)

    async def process_biometric_data(
        self,
        stream_data: Dict[BiometricStream, np.ndarray]
    ) -> EnhancedBiometricState:
        """
        Complete pipeline processing:
        1. NSCTS biometric processing
        2. LiMp entropy transformations
        3. Eopiez symbolic vectorization (simulated)
        4. Return enhanced state
        """
        # Step 1: NSCTS Processing
        consciousness_state = await self.nscts_trainer.process_biometric_data(stream_data)

        # Step 2: LiMp Entropy Transformations
        entropy_tokens = {}
        entropy_traces = {}

        for stream, signature in [
            (BiometricStream.BREATH, consciousness_state.breath),
            (BiometricStream.HEART, consciousness_state.heart),
            (BiometricStream.MOVEMENT, consciousness_state.movement),
            (BiometricStream.NEURAL, consciousness_state.neural)
        ]:
            # Convert to token
            token = self.token_adapter.signature_to_token(signature)

            # Process through entropy engine
            self.entropy_engine.run(token)

            # Store results
            entropy_tokens[stream] = token
            entropy_traces[stream] = [e for _, e in self.entropy_engine.trace()]

        # Step 3: Eopiez Symbolic Vectorization (real implementation with Julia/Python fallback)
        detected_motifs, vectorized_state = self.eopiez_vectorizer.process_consciousness_state(consciousness_state)

        # Extract results
        symbolic_vector = vectorized_state.vector
        information_density = vectorized_state.information_density
        motif_analysis = {
            "detected_motifs": {m.motif_type.value: {"weight": m.weight, "confidence": m.confidence} for m in detected_motifs},
            "entropy_score": vectorized_state.entropy_score,
            "information_density": information_density,
            "embedding_dim": len(symbolic_vector),
            "num_motifs": len(detected_motifs),
            "symbolic_expression": vectorized_state.symbolic_expression,
            "motif_configuration": vectorized_state.motif_configuration
        }

        # Create enhanced state
        enhanced_state = EnhancedBiometricState(
            original_state=consciousness_state,
            entropy_tokens=entropy_tokens,
            entropy_traces=entropy_traces,
            symbolic_vector=symbolic_vector,
            motif_analysis=motif_analysis,
            information_density=information_density
        )

        self.enhanced_states.append(enhanced_state)

        return enhanced_state

    async def get_enhanced_guidance(
        self,
        enhanced_state: EnhancedBiometricState
    ) -> Dict[str, Any]:
        """
        Generate enhanced guidance incorporating all three systems.
        """
        # Get base NSCTS guidance
        base_guidance = await self.nscts_trainer.get_coherence_guidance(
            enhanced_state.original_state
        )

        # Add entropy insights
        entropy_insights = {
            "entropy_evolution": {
                stream.value: traces[-1] if traces else 0.0
                for stream, traces in enhanced_state.entropy_traces.items()
            },
            "entropy_stability": self._calculate_entropy_stability(enhanced_state),
            "transformation_depth": len(enhanced_state.entropy_traces.get(BiometricStream.BREATH, []))
        }

        # Add symbolic insights
        symbolic_insights = {
            "information_density": enhanced_state.information_density,
            "motif_count": enhanced_state.motif_analysis.get("num_motifs", 0) if enhanced_state.motif_analysis else 0,
            "symbolic_embedding_magnitude": float(np.linalg.norm(enhanced_state.symbolic_vector)) if enhanced_state.symbolic_vector is not None else 0.0
        }

        # Merge all guidance
        enhanced_guidance = {
            **base_guidance,
            "entropy_insights": entropy_insights,
            "symbolic_insights": symbolic_insights,
            "integration_timestamp": enhanced_state.timestamp
        }

        return enhanced_guidance

    def _calculate_entropy_stability(self, state: EnhancedBiometricState) -> float:
        """Calculate how stable entropy is across transformations."""
        all_traces = []
        for traces in state.entropy_traces.values():
            all_traces.extend(traces)

        if len(all_traces) < 2:
            return 1.0

        return 1.0 - (np.std(all_traces) / (np.mean(all_traces) + 1e-12))

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary."""
        if not self.enhanced_states:
            return {"error": "No states processed"}

        # Get NSCTS summary
        nscts_summary = self.nscts_trainer.get_session_summary()

        # Calculate entropy metrics
        all_entropy_traces = []
        for state in self.enhanced_states:
            for traces in state.entropy_traces.values():
                all_entropy_traces.extend(traces)

        # Calculate information density metrics
        info_densities = [s.information_density for s in self.enhanced_states]

        return {
            **nscts_summary,
            "entropy_metrics": {
                "mean_entropy": float(np.mean(all_entropy_traces)) if all_entropy_traces else 0.0,
                "entropy_range": [float(np.min(all_entropy_traces)), float(np.max(all_entropy_traces))] if all_entropy_traces else [0.0, 0.0],
                "entropy_stability": float(1.0 - np.std(all_entropy_traces) / (np.mean(all_entropy_traces) + 1e-12)) if all_entropy_traces else 1.0
            },
            "symbolic_metrics": {
                "mean_information_density": float(np.mean(info_densities)) if info_densities else 0.0,
                "info_density_trend": "increasing" if len(info_densities) > 1 and info_densities[-1] > info_densities[0] else "stable"
            },
            "integration_status": {
                "nscts": "active",
                "limp": "active",
                "eopiez": f"active (Python implementation - Julia service: {self.eopiez_vectorizer.julia_client.available})"
            },
            "total_enhanced_states": len(self.enhanced_states)
        }


# ================================ DEMO ================================


async def demo_integrated_pipeline():
    """Demonstrate the complete integrated pipeline."""
    print("\n" + "="*70)
    print("NSCTS-LiMp-Eopiez INTEGRATED PIPELINE DEMO")
    print("="*70)

    # Initialize pipeline
    pipeline = NSCTSLiMpEopiezPipeline()

    print("\n1. Processing biometric data through integrated pipeline...")
    print("   (NSCTS → LiMp Entropy → Eopiez Vectorization)\n")

    # Generate sample biometric data
    t = np.linspace(0, 4.0, 1024)
    biometric_data = {
        BiometricStream.BREATH: 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(len(t)),
        BiometricStream.HEART: 0.3 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(len(t)),
        BiometricStream.MOVEMENT: 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(len(t)),
        BiometricStream.NEURAL: 0.4 * np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.random.randn(len(t))
    }

    # Process through pipeline
    enhanced_state = await pipeline.process_biometric_data(biometric_data)
    guidance = await pipeline.get_enhanced_guidance(enhanced_state)

    print("2. Integrated Processing Results:")
    print("   " + "-"*60)
    print(f"   NSCTS Coherence: {guidance['overall_coherence']:.3f}")
    print(f"   Learning Phase: {guidance['learning_phase']}")
    print(f"   Coherence State: {guidance['current_coherence_state']}")
    print()
    print(f"   Entropy Stability: {guidance['entropy_insights']['entropy_stability']:.3f}")
    print(f"   Transformation Depth: {guidance['entropy_insights']['transformation_depth']}")
    print()
    print(f"   Information Density: {guidance['symbolic_insights']['information_density']:.3f}")
    print(f"   Motif Count: {guidance['symbolic_insights']['motif_count']}")
    print(f"   Symbolic Embedding Magnitude: {guidance['symbolic_insights']['symbolic_embedding_magnitude']:.3f}")

    print("\n3. Pipeline Summary:")
    print("   " + "-"*60)
    summary = pipeline.get_pipeline_summary()
    print(f"   Total Enhanced States: {summary['total_enhanced_states']}")
    print(f"   Mean Entropy: {summary['entropy_metrics']['mean_entropy']:.3f}")
    print(f"   Entropy Stability: {summary['entropy_metrics']['entropy_stability']:.3f}")
    print(f"   Information Density: {summary['symbolic_metrics']['mean_information_density']:.3f}")

    print("\n4. Integration Status:")
    print("   " + "-"*60)
    for system, status in summary['integration_status'].items():
        print(f"   {system.upper()}: {status}")

    print("\n" + "="*70)
    print("INTEGRATED PIPELINE DEMO COMPLETED")
    print("All three systems working in harmony!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_integrated_pipeline())
