# NSCTS Integration Guide

## Overview

The Neurotronic Phase Caster project integrates four powerful systems for advanced biometric signal processing, entropy transformation, symbolic state representation, and neural optimization:

1. **NSCTS** (NeuroSymbiotic Coherence Training System) - Biometric processing and coherence analysis
2. **LiMp** - Entropy engine for token-based transformations
3. **Eopiez** - Message vectorizer for symbolic state representation
4. **Carryon** - SYDV neural architecture with KFP optimization, TAULS control, and persona tracking

## Architecture

```
┌─────────────────┐
│ Raw Biometrics  │
│  (4 streams)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  NSCTS Processing                   │
│  - Biometric signatures extraction  │
│  - Coherence analysis               │
│  - Learning phase management        │
│  - Spatial memory (EFL)             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  LiMp Entropy Engine                │
│  - Token transformation graphs      │
│  - Entropy tracking (SHA256)        │
│  - Dynamic branching                │
│  - Transformation memory            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Eopiez Vectorization               │
│  - Motif token generation           │
│  - Symbolic state representation    │
│  - Vector embeddings (64-128D)      │
│  - Entropy scoring                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Enhanced Training Output           │
│  - Integrated coherence guidance    │
│  - Entropy insights                 │
│  - Symbolic representations         │
└─────────────────────────────────────┘
```

## Components

### 1. Adapters

#### BiometricToTokenAdapter
Converts NSCTS `BiometricSignature` objects to LiMp `Token` objects:
- Encodes frequency, amplitude, phase, complexity into token values
- Bidirectional conversion support
- Maintains timestamp and stream metadata

#### ConsciousnessToMotifAdapter
Converts NSCTS `ConsciousnessState` to Eopiez motif tokens:
- Creates motif tokens from each biometric stream
- Adds meta-motif for coherence state
- Preserves learning phase information

### 2. Entropy Transformations

Three core transformations applied through LiMp's graph:

1. **Harmonic Enhancement** - Boosts frequency coherence when entropy is low
2. **Phase Alignment** - Normalizes phase relationships toward coherent states
3. **Complexity Filtering** - Reduces complexity noise under high entropy

### 3. Enhanced Biometric State

Extended state object containing:
- Original NSCTS consciousness state
- Entropy tokens for each biometric stream
- Entropy trace evolution
- Symbolic vector representation (64D)
- Motif analysis results
- Information density metrics

### 4. Integrated Pipeline

`NSCTSLiMpEopiezPipeline` orchestrates the complete workflow:
- Single entry point: `process_biometric_data()`
- Returns `EnhancedBiometricState` with all analysis
- Provides enhanced guidance via `get_enhanced_guidance()`
- Session summaries with metrics from all three systems

## Installation

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/9x25dillon/neurotronic_phase_caster
cd neurotronic_phase_caster
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e external/LiMp
```

For full Eopiez integration (Julia required):
```bash
# Install Julia 1.9+
# Then install Eopiez package
julia -e 'using Pkg; Pkg.develop(path="external/Eopiez")'
```

## Usage

### Basic Integration

```python
from src.integration import NSCTSLiMpEopiezPipeline
import numpy as np
from src.nscts_coherence_trainer import BiometricStream

# Initialize pipeline
pipeline = NSCTSLiMpEopiezPipeline()

# Prepare biometric data (numpy arrays)
biometric_data = {
    BiometricStream.BREATH: np.array([...]),  # Respiratory signal
    BiometricStream.HEART: np.array([...]),   # Cardiac signal
    BiometricStream.MOVEMENT: np.array([...]), # Locomotion signal
    BiometricStream.NEURAL: np.array([...])   # EEG signal
}

# Process through integrated pipeline
enhanced_state = await pipeline.process_biometric_data(biometric_data)

# Get enhanced guidance
guidance = await pipeline.get_enhanced_guidance(enhanced_state)

print(f"Coherence: {guidance['overall_coherence']:.3f}")
print(f"Entropy Stability: {guidance['entropy_insights']['entropy_stability']:.3f}")
print(f"Information Density: {guidance['symbolic_insights']['information_density']:.3f}")
```

### Custom Configuration

```python
config = {
    'nscts': {
        'sample_rate': 256.0,
        'coherence_threshold': 0.6
    }
}

pipeline = NSCTSLiMpEopiezPipeline(config)
```

### Session Summaries

```python
# After processing multiple states
summary = pipeline.get_pipeline_summary()

print(f"Total States: {summary['total_enhanced_states']}")
print(f"Mean Entropy: {summary['entropy_metrics']['mean_entropy']:.3f}")
print(f"Coherence Improvement: {summary['coherence_improvement']:.3f}")
```

## Demo

Run the integrated demo:

```bash
python src/integration/nscts_limp_eopiez_bridge.py
```

Output includes:
- NSCTS coherence metrics
- Entropy stability and transformation depth
- Information density and motif analysis
- Integrated system status

## Integration Features

### From NSCTS
- ✅ Real-time biometric processing (Welch's method, Hilbert transform)
- ✅ Coherence analysis with phase locking detection
- ✅ Learning phase transitions (4 phases)
- ✅ Spatial memory (EFL-MEM-1.0 format)

### From LiMp
- ✅ Entropy-based token transformations
- ✅ Dynamic transformation graphs
- ✅ SHA256 entropy calculation
- ✅ Complete transformation memory/audit trails

### From Eopiez
- ✅ Motif token generation from biometric patterns
- ✅ Symbolic vector embeddings (configurable dimensions)
- ✅ Entropy scoring for information complexity
- ⚠️ Julia service integration (simulated in demo, can connect to real service)

## Production Deployment

### Connecting to Julia Services

To connect Eopiez's Julia backend in production:

1. Start the Eopiez Julia server:
```bash
cd external/Eopiez
julia --project=. src/motif_detection/motif_server.jl
```

2. Update pipeline configuration:
```python
config = {
    'eopiez': {
        'julia_base_url': 'http://localhost:9000',
        'use_real_service': True
    }
}
```

3. Replace `_simulate_eopiez_vectorization()` with real HTTP calls to Julia endpoints

### Performance Considerations

- **NSCTS**: Real-time capable (256 Hz sampling)
- **LiMp**: Lightweight transformations, minimal overhead
- **Eopiez**: Julia service adds ~50-200ms latency (async recommended)

For high-throughput applications, consider:
- Batching biometric windows
- Async processing with `asyncio`
- Caching entropy transformations
- Distributed Julia workers

## Metrics and Monitoring

The integrated pipeline provides comprehensive metrics:

### Coherence Metrics (NSCTS)
- Overall coherence (0-1)
- Stream-pair coherence
- Learning phase progress
- Phase transitions count

### Entropy Metrics (LiMp)
- Mean entropy across transformations
- Entropy stability (variance)
- Transformation depth
- Entropy evolution traces

### Symbolic Metrics (Eopiez)
- Information density
- Motif count and confidence
- Embedding magnitude
- Entropy scores

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'entropy_engine'
```
**Solution**: Ensure LiMp is in Python path:
```python
sys.path.insert(0, 'external/LiMp')
```

### Julia Service Connection
```
Connection refused to localhost:9000
```
**Solution**: Start Julia service or use simulated mode (default)

### Entropy Calculation Issues
**Symptom**: All entropy values near 7.5
**Solution**: Check token value encoding - ensure JSON serialization is consistent

## Future Enhancements

- [ ] Real-time Julia service integration
- [ ] GPU acceleration for Eopiez vectorization
- [ ] Adaptive entropy thresholds based on coherence
- [ ] Multi-session coherence tracking
- [ ] WebSocket streaming for live monitoring
- [ ] Integration with THz Coherence Wearable hardware

## Contributing

See individual repository contribution guidelines:
- [NSCTS Contributing](../CONTRIBUTING.md)
- [LiMp Contributing](external/LiMp/CONTRIBUTING.md)
- [Eopiez Contributing](external/Eopiez/CONTRIBUTING.md)

## License

This integration layer: Open Source - See [LICENSE](../LICENSE)

Component licenses:
- NSCTS: Open Source
- LiMp: See external/LiMp/LICENSE
- Eopiez: MIT License

## Support

For integration-specific issues:
- GitHub Issues: https://github.com/9x25dillon/neurotronic_phase_caster/issues

For component-specific issues, refer to respective repositories.

---

**Author**: Randy Lynn / Claude Collaboration
**Date**: November 2025
**Version**: 1.0.0

## Carryon Advanced Training Integration

### Overview

Carryon brings SYDV (Sydney Variant) neural architecture principles to NSCTS coherence training:

- **KFP (Kinetic Force Principle)** - Optimizes toward minimal fluctuation intensity
- **TAULS** (Trans-Algorithmic Universal Learning System) - Two-level adaptive control
- **Entropy Regulation** - Maintains active stability under environmental stress
- **Persona Tracking** - Builds user-specific coherence profiles
- **Memory Events** - Records training sessions with consent policies

### Advanced Training Usage

```python
from src.integration import AdvancedCoherenceTrainer
import numpy as np
from src.nscts_coherence_trainer import BiometricStream

# Initialize advanced trainer with user identity
trainer = AdvancedCoherenceTrainer(
    user_id="user_001",
    user_name="Alice"
)

# Prepare biometric data
biometric_data = {
    BiometricStream.BREATH: np.array([...]),
    BiometricStream.HEART: np.array([...]),
    BiometricStream.MOVEMENT: np.array([...]),
    BiometricStream.NEURAL: np.array([...])
}

# Train with full SYDV optimization
results = await trainer.train_with_optimization(
    biometric_data,
    target_coherence=0.75
)

# Access optimization results
print(f"KFP Stability: {results['kfp_optimization']['stability_score']:.3f}")
print(f"TAULS Adjustment: {results['tauls_control']['integrated_adjustment']:.4f}")
print(f"Entropy: {results['entropy_regulation']['current_entropy']:.3f}")
print(f"Training Intensity: {results['training_guidance']['target_intensity']:.3f}")
```

### Persona Tracking

```python
# Get user persona summary
persona = trainer.get_persona_summary()
print(f"Baseline Coherence: {persona['baseline_coherence']:.3f}")
print(f"Preferred Learning Phases: {persona['preferred_phases']}")
print(f"Total Sessions: {persona['training_sessions']}")

# Get memory timeline
timeline = trainer.get_memory_timeline(limit=10)
for event in timeline:
    print(f"{event['timestamp']}: {event['type']} - Δ{event['coherence_delta']:.3f}")
```

### KFP Optimization

The Kinetic Force Principle optimizer stabilizes coherence by minimizing fluctuation intensity:

```python
from src.integration import KFPCoherenceOptimizer

optimizer = KFPCoherenceOptimizer(
    stability_weight=0.1,
    momentum=0.9
)

# Perform optimization step
metrics = optimizer.optimize_step(current_coherence=0.65)

print(f"Fluctuation Intensity: {metrics['fluctuation_intensity']:.4f}")
print(f"Kinetic Force: {metrics['kinetic_force']:.4f}")
print(f"Stability Score: {metrics['stability_score']:.3f}")
```

### TAULS Control

Two-level control system adapts training based on both immediate state and long-term patterns:

```python
from src.integration import TAULSCoherenceController

controller = TAULSCoherenceController(learning_rate=0.01)

# Get integrated control signal
control = controller.integrated_control(state, target_coherence=0.75)

print(f"Meta Control: {control['meta_adjustment']:.4f}")
print(f"Auto Control: {control['auto_adjustment']:.4f}")
print(f"Integrated: {control['integrated_adjustment']:.4f}")
print(f"Mixing Weight: {control['mixing_weight']:.3f}")
```

### Entropy Regulation

Modulates training intensity based on system entropy and environmental stress:

```python
from src.integration import CoherenceEntropyRegulator

regulator = CoherenceEntropyRegulator(max_entropy_target=0.7)

# Regulate training intensity
regulation = regulator.regulate(state)

print(f"Current Entropy: {regulation['current_entropy']:.3f}")
print(f"Environmental Stress: {regulation['environmental_stress']:.3f}")
print(f"Target Intensity: {regulation['target_intensity']:.3f}")
print(f"Entropy Trend: {regulation['entropy_trend']}")
```

### Complete Integration Example

```python
from src.integration import AdvancedCoherenceTrainer, NSCTSLiMpEopiezPipeline

# Option 1: Full SYDV optimization with persona tracking
advanced_trainer = AdvancedCoherenceTrainer(
    user_id="user_001",
    user_name="Alice"
)
results = await advanced_trainer.train_with_optimization(biometric_data)

# Option 2: Entropy + Symbolic processing without persona
pipeline = NSCTSLiMpEopiezPipeline()
enhanced_state = await pipeline.process_biometric_data(biometric_data)

# Both provide comprehensive coherence optimization with different focuses
```

## System Comparison

| Feature | NSCTS | + LiMp/Eopiez | + Carryon |
|---------|-------|---------------|-----------|
| Biometric Processing | ✅ | ✅ | ✅ |
| Coherence Analysis | ✅ | ✅ | ✅ |
| Learning Phases | ✅ | ✅ | ✅ |
| Entropy Transformations | ❌ | ✅ | ✅ |
| Symbolic Vectors | ❌ | ✅ | ✅ |
| KFP Optimization | ❌ | ❌ | ✅ |
| TAULS Control | ❌ | ❌ | ✅ |
| Entropy Regulation | ❌ | ❌ | ✅ |
| Persona Tracking | ❌ | ❌ | ✅ |
| Memory Events | ❌ | ❌ | ✅ |

## Integration Roadmap

### Phase 1: Foundation (Complete)
- ✅ NSCTS core implementation
- ✅ LiMp entropy transformations
- ✅ Eopiez symbolic vectorization
- ✅ Carryon SYDV principles

### Phase 2: Advanced Features (In Progress)
- ⏳ Real-time Julia service integration (Eopiez)
- ⏳ PyTorch-based KFP layers (full SYDV)
- ⏳ Distributed training optimization
- ⏳ Multi-user persona management

### Phase 3: Production Deployment (Planned)
- 🔜 Hardware integration (THz wearable)
- 🔜 Clinical trial data collection
- 🔜 Real-time biometric streaming
- 🔜 Cloud-based persona storage

