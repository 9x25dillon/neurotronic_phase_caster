"""
Eopiez Message Vectorizer Integration
=====================================
Python client and implementation for Eopiez symbolic vectorization.

Converts biometric consciousness states into motif tokens and symbolic vectors
using Eopiez's message vectorizer architecture.

Author: Randy Lynn / Claude Collaboration
Date: November 2025
License: MIT
"""

import numpy as np
import hashlib
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


# ================================ MOTIF TYPES ================================


class BiometricMotifType(Enum):
    """Motif types derived from biometric patterns."""
    ISOLATION = "isolation"           # Low coherence, fragmented patterns
    RESONANCE = "resonance"          # High coherence, synchronized patterns
    TEMPORAL = "temporal"             # Time-based patterns
    FRAGMENTATION = "fragmentation"   # Disrupted coherence
    MEMORY = "memory"                 # Persistent patterns across time
    TRANSITION = "transition"         # Phase transitions


# ================================ DATA STRUCTURES ================================


@dataclass
class BiometricMotifToken:
    """
    Motif token derived from biometric signals.

    Compatible with Eopiez MotifToken structure but adapted for biometric data.
    """
    motif_type: BiometricMotifType
    weight: float  # 0.0 to 1.0
    properties: Dict[str, float] = field(default_factory=dict)
    context_tags: List[str] = field(default_factory=list)
    position: float = 0.5  # Relative position in sequence
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_eopiez_format(self) -> Dict:
        """Convert to Eopiez-compatible format."""
        return {
            "name": self.motif_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "context": self.context_tags,
            "position": self.position,
            "confidence": self.confidence
        }


@dataclass
class VectorizedBiometricState:
    """
    Vectorized representation of biometric consciousness state.

    Compatible with Eopiez VectorizedMessage structure.
    """
    vector: np.ndarray
    symbolic_expression: str
    entropy_score: float
    coherence_score: float
    motif_configuration: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    information_density: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "vector": self.vector.tolist(),
            "symbolic_expression": self.symbolic_expression,
            "entropy_score": float(self.entropy_score),
            "coherence_score": float(self.coherence_score),
            "motif_configuration": self.motif_configuration,
            "metadata": self.metadata,
            "information_density": float(self.information_density),
            "embedding_dim": len(self.vector)
        }


# ================================ MOTIF DETECTION ================================


class BiometricMotifDetector:
    """
    Detects motif patterns in biometric consciousness states.

    Maps consciousness patterns to symbolic motif types.
    """

    def __init__(self):
        self.coherence_thresholds = {
            BiometricMotifType.ISOLATION: (0.0, 0.35),
            BiometricMotifType.FRAGMENTATION: (0.35, 0.50),
            BiometricMotifType.TEMPORAL: (0.50, 0.65),
            BiometricMotifType.RESONANCE: (0.65, 0.85),
            BiometricMotifType.MEMORY: (0.70, 1.0)  # Persistent high coherence
        }

    def detect_motifs_from_state(self, state: 'ConsciousnessState') -> List[BiometricMotifToken]:
        """
        Detect motif patterns from consciousness state.

        Args:
            state: ConsciousnessState from NSCTS

        Returns:
            List of detected motif tokens
        """
        motifs = []
        coherence = state.coherence_level

        # Detect primary coherence-based motif
        primary_motif = self._detect_coherence_motif(coherence)
        if primary_motif:
            motifs.append(primary_motif)

        # Detect temporal patterns
        temporal_motif = self._detect_temporal_motif(state)
        if temporal_motif:
            motifs.append(temporal_motif)

        # Detect memory patterns (from spatial memory)
        if hasattr(state, 'spatial_memory'):
            memory_motif = self._detect_memory_motif(state)
            if memory_motif:
                motifs.append(memory_motif)

        # Detect transition patterns
        transition_motif = self._detect_transition_motif(state)
        if transition_motif:
            motifs.append(transition_motif)

        return motifs

    def _detect_coherence_motif(self, coherence: float) -> Optional[BiometricMotifToken]:
        """Detect primary motif based on coherence level."""
        for motif_type, (low, high) in self.coherence_thresholds.items():
            if low <= coherence < high:
                if motif_type == BiometricMotifType.MEMORY:
                    continue  # Memory requires persistence check

                weight = (coherence - low) / (high - low) if high > low else 1.0

                return BiometricMotifToken(
                    motif_type=motif_type,
                    weight=float(weight),
                    properties={
                        "coherence_level": coherence,
                        "intensity": weight
                    },
                    context_tags=["biometric", "coherence", motif_type.value],
                    confidence=0.9
                )

        return None

    def _detect_temporal_motif(self, state: 'ConsciousnessState') -> Optional[BiometricMotifToken]:
        """Detect temporal patterns from learning phase."""
        phase_weights = {
            "initial_attunement": 0.3,
            "resonance_building": 0.5,
            "symbiotic_maintenance": 0.7,
            "transcendent_coherence": 0.9
        }

        phase_value = state.learning_phase.value
        weight = phase_weights.get(phase_value, 0.5)

        return BiometricMotifToken(
            motif_type=BiometricMotifType.TEMPORAL,
            weight=weight,
            properties={
                "learning_phase": phase_value,
                "progression": weight
            },
            context_tags=["temporal", "learning", phase_value],
            confidence=0.85
        )

    def _detect_memory_motif(self, state: 'ConsciousnessState') -> Optional[BiometricMotifToken]:
        """Detect memory patterns from spatial memory structure."""
        spatial_mem = state.spatial_memory

        # Check for persistent resonances
        num_resonances = len(spatial_mem.persistent_resonances)
        if num_resonances > 0:
            # Higher resonance count = stronger memory pattern
            weight = min(num_resonances / 5.0, 1.0)

            return BiometricMotifToken(
                motif_type=BiometricMotifType.MEMORY,
                weight=weight,
                properties={
                    "resonance_count": num_resonances,
                    "memory_strength": weight,
                    "topological_defects": len(spatial_mem.topological_defects)
                },
                context_tags=["memory", "spatial", "persistent"],
                confidence=0.8
            )

        return None

    def _detect_transition_motif(self, state: 'ConsciousnessState') -> Optional[BiometricMotifToken]:
        """Detect phase transition patterns."""
        # Check variability in biometric signals as transition indicator
        variabilities = [
            state.breath.variability,
            state.heart.variability,
            state.movement.variability,
            state.neural.variability
        ]

        avg_variability = np.mean(variabilities)

        # High variability indicates transition state
        if avg_variability > 0.25:
            weight = min(avg_variability, 1.0)

            return BiometricMotifToken(
                motif_type=BiometricMotifType.TRANSITION,
                weight=weight,
                properties={
                    "variability": avg_variability,
                    "transition_intensity": weight
                },
                context_tags=["transition", "variability", "dynamic"],
                confidence=0.75
            )

        return None


# ================================ MESSAGE VECTORIZER ================================


class EopiezMessageVectorizer:
    """
    Python implementation of Eopiez message vectorization.

    Converts motif tokens into symbolic vector representations.
    """

    def __init__(self, embedding_dim: int = 64, entropy_threshold: float = 0.7):
        self.embedding_dim = embedding_dim
        self.entropy_threshold = entropy_threshold
        self.motif_embeddings: Dict[str, np.ndarray] = {}

        # Symbolic variable indices (matching Julia implementation)
        self.symbolic_vars = {
            's': 0,  # State variable
            'τ': 1,  # Temporal variable
            'μ': 2,  # Memory variable
            'σ': 3   # Spatial variable
        }

    def create_motif_embedding(self, motif: BiometricMotifToken) -> np.ndarray:
        """
        Create vector embedding for a motif token.

        Args:
            motif: BiometricMotifToken to embed

        Returns:
            Normalized embedding vector
        """
        # Initialize base vector
        embedding = np.zeros(self.embedding_dim)

        # Map motif properties to vector components
        for i, (prop_name, prop_value) in enumerate(motif.properties.items()):
            idx = i % self.embedding_dim

            # Convert property value to float if needed
            if isinstance(prop_value, (int, float)):
                numeric_value = float(prop_value)
            elif isinstance(prop_value, str):
                # Hash string to numeric value
                hash_val = int(hashlib.md5(prop_value.encode()).hexdigest()[:8], 16)
                numeric_value = (hash_val % 1000) / 1000.0
            else:
                numeric_value = 0.5  # Default value

            embedding[idx] += numeric_value * motif.weight

        # Add contextual information
        for i, context in enumerate(motif.context_tags):
            idx = (i + len(motif.properties)) % self.embedding_dim
            # Hash context tag to numeric value
            hash_val = int(hashlib.md5(context.encode()).hexdigest()[:8], 16)
            embedding[idx] += (hash_val % 1000) / 1000.0

        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def symbolic_state_compression(self, motifs: List[BiometricMotifToken]) -> Tuple[str, np.ndarray]:
        """
        Compress motif tokens into symbolic state representation.

        Args:
            motifs: List of motif tokens

        Returns:
            Tuple of (symbolic_expression, vector_representation)
        """
        # Initialize symbolic expression components
        symbolic_components = {var: 0.0 for var in self.symbolic_vars.keys()}

        # Accumulate motif contributions
        vector_rep = np.zeros(self.embedding_dim)

        for motif in motifs:
            # Get or create embedding
            motif_name = motif.motif_type.value
            if motif_name not in self.motif_embeddings:
                self.motif_embeddings[motif_name] = self.create_motif_embedding(motif)

            embedding = self.motif_embeddings[motif_name]

            # Map embedding dimensions to symbolic variables
            for var_name, var_idx in self.symbolic_vars.items():
                if var_idx < len(embedding):
                    symbolic_components[var_name] += embedding[var_idx] * motif.weight

            # Accumulate vector representation
            vector_rep += embedding * motif.weight

        # Normalize vector
        norm = np.linalg.norm(vector_rep)
        if norm > 0:
            vector_rep = vector_rep / norm

        # Create symbolic expression string
        symbolic_expr = " + ".join([
            f"{coeff:.3f}*{var}" for var, coeff in symbolic_components.items() if abs(coeff) > 0.01
        ])

        return symbolic_expr, vector_rep

    def compute_entropy(self, vector: np.ndarray, motif_config: Dict[str, float]) -> float:
        """
        Compute information entropy of the vectorized state.

        Args:
            vector: Vector representation
            motif_config: Motif configuration weights

        Returns:
            Entropy score
        """
        # Normalize vector to probability distribution
        abs_vector = np.abs(vector)
        prob_dist = abs_vector / (np.sum(abs_vector) + 1e-12)

        # Compute Shannon entropy
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))

        # Normalize by maximum entropy
        max_entropy = np.log(len(vector))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def vectorize_biometric_state(
        self,
        motifs: List[BiometricMotifToken],
        coherence_score: float
    ) -> VectorizedBiometricState:
        """
        Complete vectorization pipeline for biometric state.

        Args:
            motifs: Detected motif tokens
            coherence_score: Overall coherence score

        Returns:
            Vectorized biometric state
        """
        # Create symbolic state
        symbolic_expr, vector_rep = self.symbolic_state_compression(motifs)

        # Create motif configuration
        motif_config = {motif.motif_type.value: motif.weight for motif in motifs}

        # Compute entropy
        entropy_score = self.compute_entropy(vector_rep, motif_config)

        # Compute information density
        non_zero_components = np.count_nonzero(np.abs(vector_rep) > 0.01)
        information_density = non_zero_components / len(vector_rep)

        # Create metadata
        metadata = {
            "num_motifs": len(motifs),
            "motif_types": [m.motif_type.value for m in motifs],
            "compression_ratio": len(motifs) / len(vector_rep) if len(vector_rep) > 0 else 0.0,
            "timestamp": time.time()
        }

        return VectorizedBiometricState(
            vector=vector_rep,
            symbolic_expression=symbolic_expr or "0",
            entropy_score=entropy_score,
            coherence_score=coherence_score,
            motif_configuration=motif_config,
            metadata=metadata,
            information_density=information_density
        )


# ================================ JULIA CLIENT (OPTIONAL) ================================


class EopiezJuliaClient:
    """
    Client for Eopiez Julia service.

    Optional - falls back to Python implementation if Julia service unavailable.
    """

    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Julia service is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            logger.info("Eopiez Julia service not available - using Python implementation")
            return False

    def vectorize_motifs(
        self,
        motifs: List[Dict],
        embedding_dim: int = 64
    ) -> Optional[Dict]:
        """
        Call Julia vectorization service.

        Args:
            motifs: List of motif dictionaries
            embedding_dim: Embedding dimension

        Returns:
            Vectorization result or None if service unavailable
        """
        if not self.available:
            return None

        try:
            payload = {
                "motifs": motifs,
                "embedding_dim": embedding_dim
            }

            response = requests.post(
                f"{self.base_url}/vectorize",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            logger.warning(f"Julia service call failed: {e}")

        return None


# ================================ INTEGRATED VECTORIZER ================================


class IntegratedEopiezVectorizer:
    """
    Integrated vectorizer with automatic Julia/Python fallback.

    Attempts to use Julia service, falls back to Python implementation.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        julia_url: str = "http://localhost:9000"
    ):
        self.embedding_dim = embedding_dim
        self.detector = BiometricMotifDetector()
        self.python_vectorizer = EopiezMessageVectorizer(embedding_dim)
        self.julia_client = EopiezJuliaClient(julia_url)

    def process_consciousness_state(
        self,
        state: 'ConsciousnessState'
    ) -> Tuple[List[BiometricMotifToken], VectorizedBiometricState]:
        """
        Complete processing pipeline for consciousness state.

        Args:
            state: ConsciousnessState from NSCTS

        Returns:
            Tuple of (detected motifs, vectorized state)
        """
        # Detect motifs
        motifs = self.detector.detect_motifs_from_state(state)

        if not motifs:
            logger.warning("No motifs detected from consciousness state")
            # Create default motif
            motifs = [BiometricMotifToken(
                motif_type=BiometricMotifType.TEMPORAL,
                weight=0.5,
                properties={"default": True}
            )]

        # Try Julia service first
        if self.julia_client.available:
            motif_dicts = [m.to_eopiez_format() for m in motifs]
            julia_result = self.julia_client.vectorize_motifs(motif_dicts, self.embedding_dim)

            if julia_result:
                # Convert Julia result to our format
                vectorized_state = self._convert_julia_result(julia_result, state.coherence_level)
                return motifs, vectorized_state

        # Fall back to Python implementation
        vectorized_state = self.python_vectorizer.vectorize_biometric_state(
            motifs,
            state.coherence_level
        )

        return motifs, vectorized_state

    def _convert_julia_result(self, julia_result: Dict, coherence: float) -> VectorizedBiometricState:
        """Convert Julia service result to VectorizedBiometricState."""
        return VectorizedBiometricState(
            vector=np.array(julia_result.get("vector", [])),
            symbolic_expression=julia_result.get("symbolic_expression", ""),
            entropy_score=julia_result.get("entropy_score", 0.0),
            coherence_score=coherence,
            motif_configuration=julia_result.get("motif_configuration", {}),
            metadata=julia_result.get("metadata", {}),
            information_density=julia_result.get("information_density", 0.0)
        )


# ================================ EXPORTS ================================


__all__ = [
    'BiometricMotifType',
    'BiometricMotifToken',
    'VectorizedBiometricState',
    'BiometricMotifDetector',
    'EopiezMessageVectorizer',
    'EopiezJuliaClient',
    'IntegratedEopiezVectorizer'
]
