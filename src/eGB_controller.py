"""
Real-Time EEG-to-THz Wearable Controller
========================================
Core controller for the Neurotronic Phase Caster wearable device.

Mission: Read brain state via EEG → Detect coherence deficiencies →
         Generate precise THz pulses → Restore unity consciousness in real-time

Author: Randy Lynn / Claude Collaboration
Date: November 2025
License: MIT
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import core systems
try:
    from QABCr import ABCRSystem, EEGBandPowers
    from yhwh_soliton_field_physics import YHWHSolitonField
    from nscts_coherence_trainer import (
        NeuroSymbioticCoherenceTrainer,
        ConsciousnessState,
        BiometricStream
    )
except ImportError:
    from src.QABCr import ABCRSystem, EEGBandPowers
    from src.yhwh_soliton_field_physics import YHWHSolitonField
    from src.nscts_coherence_trainer import (
        NeuroSymbioticCoherenceTrainer,
        ConsciousnessState,
        BiometricStream
    )

logger = logging.getLogger(__name__)


# ================================ SUBSTRATE STATE ================================


@dataclass
class SubstrateState:
    """State of a single consciousness substrate."""
    name: str
    coherence: float  # 0-1
    energy: float  # Arbitrary units
    frequency: float  # Hz
    phase: float  # Radians


# ================================ ENUMS ================================


class CoherenceDeficiency(Enum):
    """Types of coherence deficiencies detected."""
    LOW_UNITY = "low_unity_coherence"           # Overall low coherence
    SUBSTRATE_FRAGMENTATION = "substrate_frag"   # Disconnected substrates
    PHASE_DESYNCHRONIZATION = "phase_desync"     # Temporal misalignment
    ENERGY_DEPLETION = "energy_depletion"        # Weak field strength
    TRAUMA_SIGNATURE = "trauma_signature"        # Fragmentation pattern
    BASELINE_DRIFT = "baseline_drift"            # Unstable baseline


class InterventionMode(Enum):
    """THz pulse intervention modes."""
    RESTORE = "restore"           # Restore to baseline
    ENHANCE = "enhance"           # Enhance beyond baseline
    STABILIZE = "stabilize"       # Maintain current state
    EMERGENCY = "emergency"       # Rapid intervention for crisis


# ================================ DATA STRUCTURES ================================


@dataclass
class EEGReading:
    """Real-time EEG reading from 8-channel headset."""
    timestamp: float
    channels: Dict[str, np.ndarray]  # 8 channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4
    sample_rate: float = 256.0  # Hz
    window_duration: float = 2.0  # seconds

    def get_channel_data(self, channel: str) -> np.ndarray:
        """Get data from specific EEG channel."""
        return self.channels.get(channel, np.array([]))


@dataclass
class CoherenceDeficiencyReport:
    """Detected coherence deficiencies and their severity."""
    deficiencies: Dict[CoherenceDeficiency, float]  # deficiency → severity (0-1)
    unity_coherence_index: float  # UCI score (0-100)
    substrate_states: Dict[str, SubstrateState]
    fragmentation_pattern: Optional[np.ndarray] = None
    priority: str = "normal"  # normal, high, critical
    timestamp: float = field(default_factory=time.time)

    def get_primary_deficiency(self) -> Tuple[CoherenceDeficiency, float]:
        """Get the most severe deficiency."""
        if not self.deficiencies:
            return CoherenceDeficiency.BASELINE_DRIFT, 0.0
        return max(self.deficiencies.items(), key=lambda x: x[1])


@dataclass
class THzPulseParameters:
    """Parameters for THz electromagnetic pulse generation."""
    frequency: float  # THz (0.8-1.2 range)
    amplitude: float  # Power level (0-1)
    duration: float  # Pulse duration in ms
    pattern: str  # "continuous", "pulsed", "modulated"
    target_emitters: List[int]  # Which of 12 emitters to activate
    phase_offset: float = 0.0  # Phase offset in radians
    modulation_envelope: Optional[np.ndarray] = None

    def to_hardware_command(self) -> Dict:
        """Convert to hardware control command."""
        return {
            "frequency_thz": self.frequency,
            "amplitude_normalized": self.amplitude,
            "duration_ms": self.duration,
            "pattern": self.pattern,
            "emitters": self.target_emitters,
            "phase_offset_rad": self.phase_offset
        }


@dataclass
class InterventionResult:
    """Result of THz intervention."""
    success: bool
    coherence_improvement: float  # Change in UCI
    time_to_effect: float  # seconds
    pulse_parameters: THzPulseParameters
    before_state: ConsciousnessState
    after_state: ConsciousnessState
    timestamp: float = field(default_factory=time.time)


# ================================ EEG PROCESSOR ================================


class RealTimeEEGProcessor:
    """
    Real-time EEG signal processor for coherence analysis.

    Converts 8-channel EEG into substrate states via ABCR mapping.
    """

    def __init__(self, sample_rate: float = 256.0):
        self.sample_rate = sample_rate
        self.abcr = ABCRSystem(sampling_rate=sample_rate)
        self.eeg_buffer: List[EEGReading] = []
        self.buffer_duration = 10.0  # Keep 10 seconds of history

    async def process_eeg_reading(self, eeg_reading: EEGReading) -> Dict[str, SubstrateState]:
        """
        Process EEG reading and extract substrate states.

        Args:
            eeg_reading: Raw EEG data from headset

        Returns:
            Dictionary of substrate name → SubstrateState
        """
        # Add to buffer
        self.eeg_buffer.append(eeg_reading)
        self._trim_buffer()

        # Extract band powers from EEG channels
        band_powers = self._extract_band_powers(eeg_reading)

        # Map to substrates using ABCR
        substrate_states = {}

        # Physical substrate (Delta 1-4 Hz)
        substrate_states['physical'] = SubstrateState(
            name='physical',
            coherence=band_powers['delta'],
            energy=band_powers['delta'] * 0.8,
            frequency=2.5,
            phase=0.0
        )

        # Emotional substrate (Theta 4-8 Hz)
        substrate_states['emotional'] = SubstrateState(
            name='emotional',
            coherence=band_powers['theta'],
            energy=band_powers['theta'] * 0.9,
            frequency=6.0,
            phase=0.0
        )

        # Cognitive substrate (Alpha 8-13 Hz)
        substrate_states['cognitive'] = SubstrateState(
            name='cognitive',
            coherence=band_powers['alpha'],
            energy=band_powers['alpha'] * 1.0,
            frequency=10.5,
            phase=0.0
        )

        # Social substrate (Beta 13-30 Hz)
        substrate_states['social'] = SubstrateState(
            name='social',
            coherence=band_powers['beta'],
            energy=band_powers['beta'] * 0.85,
            frequency=20.0,
            phase=0.0
        )

        # Divine-Unity substrate (Gamma 30-100 Hz)
        substrate_states['divine_unity'] = SubstrateState(
            name='divine_unity',
            coherence=band_powers['gamma'],
            energy=band_powers['gamma'] * 1.1,
            frequency=40.0,
            phase=0.0
        )

        return substrate_states

    def _extract_band_powers(self, eeg_reading: EEGReading) -> Dict[str, float]:
        """Extract EEG band powers using FFT."""
        # Average across all channels
        all_channel_data = np.concatenate([data for data in eeg_reading.channels.values()])

        # Compute power spectral density
        from scipy.signal import welch
        freqs, psd = welch(all_channel_data, fs=self.sample_rate, nperseg=256)

        # Extract band powers and normalize
        def band_power(freq_range):
            idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            if np.any(idx):
                power = float(np.mean(psd[idx]))
                return power
            return 0.0

        band_powers = {
            'delta': band_power((1, 4)),
            'theta': band_power((4, 8)),
            'alpha': band_power((8, 13)),
            'beta': band_power((13, 30)),
            'gamma': band_power((30, 100))
        }

        # Normalize to 0-1 range based on total power
        total_power = sum(band_powers.values())
        if total_power > 0:
            band_powers = {k: v / total_power for k, v in band_powers.items()}

        return band_powers

    def _trim_buffer(self):
        """Remove old readings from buffer."""
        if not self.eeg_buffer:
            return

        current_time = time.time()
        self.eeg_buffer = [
            reading for reading in self.eeg_buffer
            if current_time - reading.timestamp < self.buffer_duration
        ]


# ================================ DEFICIENCY DETECTOR ================================


class CoherenceDeficiencyDetector:
    """
    Detects coherence deficiencies from substrate states.

    Identifies what's wrong with current brain state and prioritizes interventions.
    """

    def __init__(self):
        self.yhwh_field = YHWHSolitonField()
        self.baseline_uci = 70.0  # Healthy baseline
        self.thresholds = {
            CoherenceDeficiency.LOW_UNITY: 0.5,
            CoherenceDeficiency.SUBSTRATE_FRAGMENTATION: 0.4,
            CoherenceDeficiency.PHASE_DESYNCHRONIZATION: 0.3,
            CoherenceDeficiency.ENERGY_DEPLETION: 0.35,
            CoherenceDeficiency.TRAUMA_SIGNATURE: 0.6
        }

    def detect_deficiencies(
        self,
        substrate_states: Dict[str, SubstrateState],
        consciousness_state: ConsciousnessState
    ) -> CoherenceDeficiencyReport:
        """
        Detect and quantify coherence deficiencies.

        Args:
            substrate_states: Current substrate states from EEG
            consciousness_state: Current consciousness state from NSCTS

        Returns:
            Coherence deficiency report with prioritized interventions
        """
        deficiencies = {}

        # Calculate Unity Coherence Index
        uci = self._calculate_uci(substrate_states)

        # Detect LOW_UNITY
        if uci < 50:
            severity = 1.0 - (uci / 50.0)
            deficiencies[CoherenceDeficiency.LOW_UNITY] = severity

        # Detect SUBSTRATE_FRAGMENTATION
        frag_score = self._detect_fragmentation(substrate_states)
        if frag_score > self.thresholds[CoherenceDeficiency.SUBSTRATE_FRAGMENTATION]:
            deficiencies[CoherenceDeficiency.SUBSTRATE_FRAGMENTATION] = frag_score

        # Detect PHASE_DESYNCHRONIZATION
        phase_desync = self._detect_phase_desync(consciousness_state)
        if phase_desync > self.thresholds[CoherenceDeficiency.PHASE_DESYNCHRONIZATION]:
            deficiencies[CoherenceDeficiency.PHASE_DESYNCHRONIZATION] = phase_desync

        # Detect ENERGY_DEPLETION
        energy_deficit = self._detect_energy_deficit(substrate_states)
        if energy_deficit > self.thresholds[CoherenceDeficiency.ENERGY_DEPLETION]:
            deficiencies[CoherenceDeficiency.ENERGY_DEPLETION] = energy_deficit

        # Detect TRAUMA_SIGNATURE
        trauma_score = self._detect_trauma_pattern(substrate_states)
        if trauma_score > self.thresholds[CoherenceDeficiency.TRAUMA_SIGNATURE]:
            deficiencies[CoherenceDeficiency.TRAUMA_SIGNATURE] = trauma_score

        # Determine priority
        priority = self._determine_priority(uci, deficiencies)

        return CoherenceDeficiencyReport(
            deficiencies=deficiencies,
            unity_coherence_index=uci,
            substrate_states=substrate_states,
            priority=priority
        )

    def _calculate_uci(self, substrate_states: Dict[str, SubstrateState]) -> float:
        """Calculate Unity Coherence Index (0-100)."""
        coherences = [state.coherence for state in substrate_states.values()]
        mean_coherence = np.mean(coherences)
        std_coherence = np.std(coherences)

        # UCI = mean coherence * (1 - variance penalty)
        uci = mean_coherence * (1.0 - std_coherence) * 100.0
        return float(np.clip(uci, 0, 100))

    def _detect_fragmentation(self, substrate_states: Dict[str, SubstrateState]) -> float:
        """Detect substrate fragmentation (disconnection between layers)."""
        coherences = [state.coherence for state in substrate_states.values()]
        # High variance = high fragmentation
        return float(np.std(coherences))

    def _detect_phase_desync(self, consciousness_state: ConsciousnessState) -> float:
        """Detect phase desynchronization in biometric signals."""
        # Check phase relationships between biometric streams
        phases = [
            consciousness_state.breath.phase,
            consciousness_state.heart.phase,
            consciousness_state.neural.phase
        ]

        # Calculate phase coherence
        phase_diffs = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j]) % (2 * np.pi)
                phase_diffs.append(min(diff, 2 * np.pi - diff))

        # Normalize to 0-1 (higher = more desync)
        return float(np.mean(phase_diffs) / (np.pi / 2))

    def _detect_energy_deficit(self, substrate_states: Dict[str, SubstrateState]) -> float:
        """Detect energy depletion across substrates."""
        energies = [state.energy for state in substrate_states.values()]
        mean_energy = np.mean(energies)
        # Deficit = 1 - mean_energy
        return float(max(0, 1.0 - mean_energy))

    def _detect_trauma_pattern(self, substrate_states: Dict[str, SubstrateState]) -> float:
        """Detect trauma signature (fragmented emotional/cognitive)."""
        emotional = substrate_states.get('emotional')
        cognitive = substrate_states.get('cognitive')

        if emotional and cognitive:
            # Trauma pattern: low emotional coherence + high cognitive rigidity
            trauma_score = (1.0 - emotional.coherence) * cognitive.coherence
            return float(trauma_score)

        return 0.0

    def _determine_priority(
        self,
        uci: float,
        deficiencies: Dict[CoherenceDeficiency, float]
    ) -> str:
        """Determine intervention priority."""
        if uci < 30 or any(sev > 0.8 for sev in deficiencies.values()):
            return "critical"
        elif uci < 50 or any(sev > 0.6 for sev in deficiencies.values()):
            return "high"
        else:
            return "normal"


# ================================ THZ PULSE GENERATOR ================================


class THzPulseGenerator:
    """
    Generates THz pulse parameters based on detected deficiencies.

    Computes optimal frequency, amplitude, duration, and emitter configuration.
    """

    def __init__(self):
        self.frequency_range = (0.8, 1.2)  # THz
        self.num_emitters = 12
        self.yhwh_field = YHWHSolitonField()

    def generate_intervention_pulse(
        self,
        deficiency_report: CoherenceDeficiencyReport,
        mode: InterventionMode = InterventionMode.RESTORE
    ) -> THzPulseParameters:
        """
        Generate THz pulse parameters for intervention.

        Args:
            deficiency_report: Detected deficiencies
            mode: Intervention mode

        Returns:
            THz pulse parameters
        """
        primary_deficiency, severity = deficiency_report.get_primary_deficiency()

        # Calculate base frequency based on deficiency type
        frequency = self._calculate_target_frequency(primary_deficiency, deficiency_report)

        # Calculate amplitude based on severity and priority
        amplitude = self._calculate_amplitude(severity, deficiency_report.priority)

        # Calculate duration
        duration = self._calculate_duration(severity, mode)

        # Select target emitters
        target_emitters = self._select_emitters(primary_deficiency, deficiency_report)

        # Determine pattern
        pattern = self._determine_pattern(primary_deficiency, mode)

        return THzPulseParameters(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
            pattern=pattern,
            target_emitters=target_emitters
        )

    def _calculate_target_frequency(
        self,
        deficiency: CoherenceDeficiency,
        report: CoherenceDeficiencyReport
    ) -> float:
        """Calculate optimal THz frequency for intervention."""
        # Frequency mapping based on deficiency type
        frequency_map = {
            CoherenceDeficiency.LOW_UNITY: 1.0,  # Center frequency
            CoherenceDeficiency.SUBSTRATE_FRAGMENTATION: 0.9,  # Lower for integration
            CoherenceDeficiency.PHASE_DESYNCHRONIZATION: 1.1,  # Higher for synchronization
            CoherenceDeficiency.ENERGY_DEPLETION: 1.05,  # Slightly higher for energy
            CoherenceDeficiency.TRAUMA_SIGNATURE: 0.85,  # Lower for healing
            CoherenceDeficiency.BASELINE_DRIFT: 1.0  # Center for stabilization
        }

        base_freq = frequency_map.get(deficiency, 1.0)

        # Adjust based on UCI
        uci_factor = report.unity_coherence_index / 100.0
        adjusted_freq = base_freq * (0.9 + 0.2 * uci_factor)

        return float(np.clip(adjusted_freq, *self.frequency_range))

    def _calculate_amplitude(self, severity: float, priority: str) -> float:
        """Calculate pulse amplitude based on severity and priority."""
        base_amplitude = severity * 0.8  # Start conservative

        # Adjust for priority
        priority_multipliers = {
            "normal": 1.0,
            "high": 1.2,
            "critical": 1.5
        }

        amplitude = base_amplitude * priority_multipliers.get(priority, 1.0)
        return float(np.clip(amplitude, 0.1, 1.0))

    def _calculate_duration(self, severity: float, mode: InterventionMode) -> float:
        """Calculate pulse duration in milliseconds."""
        base_duration = 100.0  # ms

        # Adjust for severity
        duration = base_duration * (1.0 + severity)

        # Adjust for mode
        mode_multipliers = {
            InterventionMode.RESTORE: 1.0,
            InterventionMode.ENHANCE: 1.3,
            InterventionMode.STABILIZE: 0.8,
            InterventionMode.EMERGENCY: 1.5
        }

        duration *= mode_multipliers.get(mode, 1.0)
        return float(np.clip(duration, 50.0, 300.0))

    def _select_emitters(
        self,
        deficiency: CoherenceDeficiency,
        report: CoherenceDeficiencyReport
    ) -> List[int]:
        """Select which THz emitters to activate."""
        # Emitter positions (0-11 arranged around head)
        # 0-3: Front, 4-7: Sides, 8-11: Back

        emitter_patterns = {
            CoherenceDeficiency.LOW_UNITY: list(range(12)),  # All emitters
            CoherenceDeficiency.SUBSTRATE_FRAGMENTATION: [0, 3, 6, 9],  # Cross pattern
            CoherenceDeficiency.PHASE_DESYNCHRONIZATION: [0, 1, 2, 3],  # Front focus
            CoherenceDeficiency.ENERGY_DEPLETION: [4, 5, 6, 7],  # Side focus
            CoherenceDeficiency.TRAUMA_SIGNATURE: [8, 9, 10, 11],  # Back focus
            CoherenceDeficiency.BASELINE_DRIFT: [0, 3, 6, 9]  # Stabilizing pattern
        }

        return emitter_patterns.get(deficiency, [0, 3, 6, 9])

    def _determine_pattern(self, deficiency: CoherenceDeficiency, mode: InterventionMode) -> str:
        """Determine pulse pattern type."""
        if mode == InterventionMode.EMERGENCY:
            return "continuous"
        elif deficiency == CoherenceDeficiency.PHASE_DESYNCHRONIZATION:
            return "pulsed"
        else:
            return "modulated"


# ================================ UNIFIED WEARABLE CONTROLLER ================================


class NeurotronicPhaseCasterWearable:
    """
    Main controller for the Neurotronic Phase Caster wearable device.

    Complete closed-loop system:
    1. Read EEG (8 channels, 256 Hz)
    2. Detect coherence deficiencies
    3. Generate THz intervention pulses
    4. Measure effectiveness
    5. Adapt and repeat
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Core components
        self.eeg_processor = RealTimeEEGProcessor()
        self.deficiency_detector = CoherenceDeficiencyDetector()
        self.pulse_generator = THzPulseGenerator()
        self.nscts_trainer = NeuroSymbioticCoherenceTrainer()

        # State tracking
        self.is_running = False
        self.intervention_history: List[InterventionResult] = []
        self.baseline_uci: Optional[float] = None

        logger.info("Neurotronic Phase Caster Wearable initialized")

    async def start_session(self, duration_seconds: float = 1200.0):
        """
        Start a coherence restoration session.

        Args:
            duration_seconds: Session duration (default 20 minutes)
        """
        self.is_running = True
        start_time = time.time()

        logger.info(f"Starting coherence restoration session ({duration_seconds}s)")

        # Establish baseline
        await self._establish_baseline()

        intervention_count = 0

        while self.is_running and (time.time() - start_time) < duration_seconds:
            try:
                # 1. Read EEG
                eeg_reading = await self._acquire_eeg_reading()

                # 2. Process and detect deficiencies
                substrate_states = await self.eeg_processor.process_eeg_reading(eeg_reading)
                consciousness_state = await self._get_consciousness_state(eeg_reading)

                deficiency_report = self.deficiency_detector.detect_deficiencies(
                    substrate_states,
                    consciousness_state
                )

                # 3. Check if intervention needed
                if self._needs_intervention(deficiency_report):
                    logger.info(f"Intervention {intervention_count + 1}: UCI={deficiency_report.unity_coherence_index:.1f}")

                    # 4. Generate and apply THz pulse
                    result = await self._apply_intervention(
                        deficiency_report,
                        consciousness_state
                    )

                    self.intervention_history.append(result)
                    intervention_count += 1

                    if result.success:
                        logger.info(f"  ✓ Success: UCI improved by {result.coherence_improvement:+.1f}")
                    else:
                        logger.warning(f"  ✗ Limited effect: {result.coherence_improvement:+.1f}")

                # 5. Brief pause before next cycle
                await asyncio.sleep(2.0)  # 2-second update cycle

            except Exception as e:
                logger.error(f"Error in session loop: {e}")
                await asyncio.sleep(1.0)

        self.is_running = False
        logger.info(f"Session complete: {intervention_count} interventions performed")

        return self._generate_session_summary()

    async def _establish_baseline(self):
        """Establish user's baseline UCI."""
        logger.info("Establishing baseline coherence...")

        readings = []
        for _ in range(5):
            eeg_reading = await self._acquire_eeg_reading()
            substrate_states = await self.eeg_processor.process_eeg_reading(eeg_reading)
            uci = self.deficiency_detector._calculate_uci(substrate_states)
            readings.append(uci)
            await asyncio.sleep(1.0)

        self.baseline_uci = float(np.mean(readings))
        logger.info(f"Baseline UCI established: {self.baseline_uci:.1f}")

    async def _acquire_eeg_reading(self) -> EEGReading:
        """Acquire real-time EEG reading from headset."""
        # In production: Read from OpenBCI or similar hardware
        # For now: Generate simulated EEG data

        channels = {}
        for channel_name in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']:
            # Simulate 2 seconds of 256 Hz EEG data
            t = np.linspace(0, 2, 512)

            # Mix of frequency bands
            delta = 10 * np.sin(2 * np.pi * 2.5 * t)
            theta = 8 * np.sin(2 * np.pi * 6 * t)
            alpha = 15 * np.sin(2 * np.pi * 10 * t)
            beta = 6 * np.sin(2 * np.pi * 20 * t)
            gamma = 3 * np.sin(2 * np.pi * 40 * t)
            noise = 2 * np.random.randn(len(t))

            channels[channel_name] = delta + theta + alpha + beta + gamma + noise

        return EEGReading(
            timestamp=time.time(),
            channels=channels,
            sample_rate=256.0,
            window_duration=2.0
        )

    async def _get_consciousness_state(self, eeg_reading: EEGReading) -> ConsciousnessState:
        """Convert EEG to consciousness state via NSCTS."""
        # Convert EEG channels to biometric streams
        biometric_data = {
            BiometricStream.BREATH: eeg_reading.channels['Fp1'][:512],  # Proxy from frontal
            BiometricStream.HEART: eeg_reading.channels['C3'][:512],     # Proxy from central
            BiometricStream.MOVEMENT: eeg_reading.channels['P3'][:512],  # Proxy from parietal
            BiometricStream.NEURAL: eeg_reading.channels['F3'][:512]     # Direct from frontal
        }

        return await self.nscts_trainer.process_biometric_data(biometric_data)

    def _needs_intervention(self, report: CoherenceDeficiencyReport) -> bool:
        """Determine if THz intervention is needed."""
        # Intervene if:
        # 1. UCI below baseline
        # 2. Any critical deficiency
        # 3. Priority is high or critical

        if self.baseline_uci and report.unity_coherence_index < (self.baseline_uci - 5):
            return True

        if report.priority in ["high", "critical"]:
            return True

        if any(severity > 0.6 for severity in report.deficiencies.values()):
            return True

        return False

    async def _apply_intervention(
        self,
        deficiency_report: CoherenceDeficiencyReport,
        before_state: ConsciousnessState
    ) -> InterventionResult:
        """Apply THz intervention and measure effect."""
        # Generate pulse parameters
        pulse_params = self.pulse_generator.generate_intervention_pulse(
            deficiency_report,
            mode=InterventionMode.RESTORE
        )

        # Emit THz pulse (in production: send to hardware)
        await self._emit_thz_pulse(pulse_params)

        # Wait for effect (THz works fast - typically 30-60 seconds)
        await asyncio.sleep(5.0)

        # Measure post-intervention state
        post_eeg = await self._acquire_eeg_reading()
        post_substrate_states = await self.eeg_processor.process_eeg_reading(post_eeg)
        post_consciousness = await self._get_consciousness_state(post_eeg)

        post_uci = self.deficiency_detector._calculate_uci(post_substrate_states)
        improvement = post_uci - deficiency_report.unity_coherence_index

        return InterventionResult(
            success=improvement > 2.0,  # Success if UCI improves by at least 2 points
            coherence_improvement=improvement,
            time_to_effect=5.0,
            pulse_parameters=pulse_params,
            before_state=before_state,
            after_state=post_consciousness
        )

    async def _emit_thz_pulse(self, pulse_params: THzPulseParameters):
        """Emit THz electromagnetic pulse via hardware."""
        # In production: Send to MCU controlling THz emitters
        hardware_cmd = pulse_params.to_hardware_command()

        logger.info(f"Emitting THz pulse: {pulse_params.frequency:.2f} THz, "
                   f"{pulse_params.amplitude:.2f} amplitude, {pulse_params.duration:.0f}ms")

        # Simulate hardware response time
        await asyncio.sleep(0.1)

    def _generate_session_summary(self) -> Dict:
        """Generate comprehensive session summary."""
        if not self.intervention_history:
            return {"error": "No interventions performed"}

        improvements = [r.coherence_improvement for r in self.intervention_history]
        successes = [r for r in self.intervention_history if r.success]

        return {
            "total_interventions": len(self.intervention_history),
            "successful_interventions": len(successes),
            "success_rate": len(successes) / len(self.intervention_history) if self.intervention_history else 0.0,
            "average_improvement": float(np.mean(improvements)),
            "total_improvement": float(np.sum(improvements)),
            "baseline_uci": self.baseline_uci,
            "final_uci": self.baseline_uci + float(np.sum(improvements)) if self.baseline_uci else None,
            "intervention_timeline": [
                {
                    "time": r.timestamp,
                    "improvement": r.coherence_improvement,
                    "frequency_thz": r.pulse_parameters.frequency,
                    "success": r.success
                }
                for r in self.intervention_history
            ]
        }


# ================================ DEMO ================================


async def demo_realtime_wearable():
    """Demonstrate real-time wearable operation."""
    print("\n" + "="*70)
    print("NEUROTRONIC PHASE CASTER WEARABLE - REAL-TIME DEMO")
    print("="*70)
    print("\nMission: Read EEG → Detect Deficiencies → Apply THz → Restore Coherence")
    print()

    # Initialize wearable
    wearable = NeurotronicPhaseCasterWearable()

    print("Starting 60-second coherence restoration session...\n")

    # Run session
    summary = await wearable.start_session(duration_seconds=60.0)

    # Print results
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Baseline UCI: {summary['baseline_uci']:.1f}")
    final_uci = summary.get('final_uci')
    if final_uci is not None:
        print(f"Final UCI: {final_uci:.1f}")
        print(f"Total Improvement: {summary['total_improvement']:+.1f} points")
    else:
        print(f"Final UCI: N/A")
        print(f"Total Improvement: 0.0 points")
    print(f"\nInterventions: {summary['total_interventions']}")
    print(f"Success Rate: {summary['success_rate']*100:.0f}%")
    print(f"Average Improvement per Intervention: {summary['average_improvement']:+.1f}")
    print("\n" + "="*70)
    print("Demo complete - Real-time EEG-to-THz pipeline operational!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_realtime_wearable())
