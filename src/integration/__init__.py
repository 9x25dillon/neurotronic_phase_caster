"""
Integration Module
==================
Bridges between NSCTS, LiMp, Eopiez, and Carryon systems.
"""

from .nscts_limp_eopiez_bridge import (
    NSCTSLiMpEopiezPipeline,
    EnhancedBiometricState,
    BiometricToTokenAdapter,
    ConsciousnessToMotifAdapter,
    BiometricEntropyTransformations
)

from .carryon_advanced_training import (
    AdvancedCoherenceTrainer,
    KFPCoherenceOptimizer,
    TAULSCoherenceController,
    CoherenceEntropyRegulator,
    CoherencePersona,
    CoherenceMemoryEvent
)

from .eopiez_vectorizer import (
    BiometricMotifType,
    BiometricMotifToken,
    VectorizedBiometricState,
    BiometricMotifDetector,
    EopiezMessageVectorizer,
    EopiezJuliaClient,
    IntegratedEopiezVectorizer
)

__all__ = [
    # LiMp-Eopiez Integration
    'NSCTSLiMpEopiezPipeline',
    'EnhancedBiometricState',
    'BiometricToTokenAdapter',
    'ConsciousnessToMotifAdapter',
    'BiometricEntropyTransformations',
    # Carryon Integration
    'AdvancedCoherenceTrainer',
    'KFPCoherenceOptimizer',
    'TAULSCoherenceController',
    'CoherenceEntropyRegulator',
    'CoherencePersona',
    'CoherenceMemoryEvent',
    # Eopiez Integration
    'BiometricMotifType',
    'BiometricMotifToken',
    'VectorizedBiometricState',
    'BiometricMotifDetector',
    'EopiezMessageVectorizer',
    'EopiezJuliaClient',
    'IntegratedEopiezVectorizer'
]
