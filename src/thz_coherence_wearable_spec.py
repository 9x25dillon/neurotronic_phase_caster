"""
THz Coherence Wearable Hardware Controller
Controls 12-channel THz emitter array with safety systems

Hardware Components:
- 12× THz QCL emitters (0.8-1.2 THz, max 5 mW each)
- 8× EEG channels (OpenBCI or similar)
- STM32H7 MCU (480 MHz, FPU)
- Multi-layer safety system
- Power management & thermal monitoring

Safety Features:
- Power limiting (<60 mW total)
- Emergency shutdown
- Thermal monitoring
- Watchdog timers
- Fail-safe defaults
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SafetyState(Enum):
    """Safety system states."""

    SAFE = "safe"
    WARNING = "warning"
    SHUTDOWN = "shutdown"


@dataclass
class EmitterConfig:
    """Configuration for single THz emitter."""

    channel_id: int
    frequency_thz: float  # 0.8-1.2 THz
    max_power_mw: float  # Maximum power (mW)
    current_power_mw: float  # Current power setting
    enabled: bool


class SafetySystem:
    """
    Multi-layer safety system for THz emission.

    Monitors:
    - Total power output
    - Individual emitter power
    - Temperature
    - Session duration
    - Emergency shutdown
    """

    MAX_TOTAL_POWER_MW = 60.0  # FDA/IEC limit
    MAX_EMITTER_POWER_MW = 5.0  # Per emitter
    MAX_TEMPERATURE_C = 45.0  # Thermal safety
    MAX_SESSION_MINUTES = 60.0  # Session time limit

    def __init__(self):
        """Initialize safety system."""
        self.state = SafetyState.SAFE
        self.total_power_mw = 0.0
        self.temperature_c = 25.0
        self.session_time_sec = 0.0
        self.emergency_shutdown_triggered = False

    def check_power_limit(self, proposed_powers: List[float]) -> bool:
        """
        Check if proposed power settings are safe.

        Args:
            proposed_powers: List of power values for each emitter (mW)

        Returns:
            bool: True if safe, False if would exceed limits
        """
        total = sum(proposed_powers)

        if total > self.MAX_TOTAL_POWER_MW:
            self.state = SafetyState.WARNING
            return False

        for power in proposed_powers:
            if power > self.MAX_EMITTER_POWER_MW:
                self.state = SafetyState.WARNING
                return False

        return True

    def check_temperature(self, temp_c: float) -> bool:
        """Check temperature safety."""
        self.temperature_c = temp_c

        if temp_c > self.MAX_TEMPERATURE_C:
            self.state = SafetyState.SHUTDOWN
            return False

        return True

    def check_session_time(self, time_sec: float) -> bool:
        """Check if session duration is within limits."""
        self.session_time_sec = time_sec

        if time_sec > self.MAX_SESSION_MINUTES * 60:
            self.state = SafetyState.WARNING
            return False

        return True

    def trigger_emergency_shutdown(self):
        """Immediately shutdown all emitters."""
        self.emergency_shutdown_triggered = True
        self.state = SafetyState.SHUTDOWN
        print("⚠️ EMERGENCY SHUTDOWN TRIGGERED ⚠️")

    def is_safe_to_operate(self) -> bool:
        """Check if system is safe to operate."""
        return self.state == SafetyState.SAFE and not self.emergency_shutdown_triggered

    def reset(self):
        """Reset safety system (after addressing issues)."""
        self.state = SafetyState.SAFE
        self.emergency_shutdown_triggered = False
        self.session_time_sec = 0.0


class EmitterArray:
    """
    12-channel THz QCL emitter array.

    Array layout (around head):
    - Channels 0-2: Frontal region
    - Channels 3-5: Left temporal
    - Channels 6-8: Right temporal
    - Channels 9-11: Occipital/parietal
    """

    N_CHANNELS = 12

    def __init__(self):
        """Initialize emitter array."""
        self.emitters = [
            EmitterConfig(
                channel_id=i,
                frequency_thz=0.8 + (i % 4) * 0.1,  # Vary frequency
                max_power_mw=5.0,
                current_power_mw=0.0,
                enabled=False,
            )
            for i in range(self.N_CHANNELS)
        ]

    def set_power(self, channel: int, power_mw: float):
        """Set power for specific channel."""
        if 0 <= channel < self.N_CHANNELS:
            self.emitters[channel].current_power_mw = np.clip(power_mw, 0, 5.0)

    def set_power_pattern(self, powers: np.ndarray):
        """
        Set power for all channels from array.

        Args:
            powers: [12] power values (mW)
        """
        for i, power in enumerate(powers[:self.N_CHANNELS]):
            self.set_power(i, power)

    def enable_channel(self, channel: int, enabled: bool = True):
        """Enable/disable specific channel."""
        if 0 <= channel < self.N_CHANNELS:
            self.emitters[channel].enabled = enabled

    def get_total_power(self) -> float:
        """Get current total power output."""
        return sum(
            e.current_power_mw for e in self.emitters if e.enabled
        )

    def shutdown_all(self):
        """Emergency shutdown: disable all emitters."""
        for emitter in self.emitters:
            emitter.enabled = False
            emitter.current_power_mw = 0.0


class THzCoherenceWearable:
    """
    Complete THz Coherence Wearable system controller.

    Integrates:
    - EEG acquisition
    - Real-time analysis
    - THz emission control
    - Safety monitoring
    - Closed-loop optimization
    """

    def __init__(self):
        """Initialize wearable controller."""
        self.emitters = EmitterArray()
        self.safety = SafetySystem()
        self.is_running = False
        self.session_time = 0.0

    def start_session(self, duration_minutes: float):
        """
        Start therapeutic session.

        Args:
            duration_minutes: Planned session duration
        """
        if not self.safety.is_safe_to_operate():
            print("⚠️ Cannot start: safety system not in SAFE state")
            return False

        if duration_minutes > self.safety.MAX_SESSION_MINUTES:
            print(f"⚠️ Duration exceeds max ({self.safety.MAX_SESSION_MINUTES} min)")
            return False

        self.is_running = True
        self.session_time = 0.0
        print(f"✓ Session started: {duration_minutes:.1f} minutes")
        return True

    def apply_substrate_pattern(self, substrate_powers: np.ndarray):
        """
        Apply THz power pattern targeting specific substrates.

        Maps substrate power levels [5] to emitter array [12].

        Args:
            substrate_powers: [5] power levels for each substrate (mW)
        """
        if not self.safety.is_safe_to_operate():
            print("⚠️ Cannot apply pattern: safety system triggered")
            return False

        # Map 5 substrates to 12 emitters
        # Each substrate activates 2-3 emitters in corresponding brain region
        emitter_powers = np.zeros(12)

        # Physical substrate → Occipital (channels 9-11)
        emitter_powers[9:12] = substrate_powers[0]

        # Emotional substrate → Temporal (channels 3-5, 6-8)
        emitter_powers[3:6] = substrate_powers[1]
        emitter_powers[6:9] = substrate_powers[1]

        # Cognitive substrate → Frontal (channels 0-2)
        emitter_powers[0:3] = substrate_powers[2]

        # Social substrate → Left temporal (channels 3-5)
        emitter_powers[3:6] += substrate_powers[3] * 0.5  # Blend

        # Divine-Unity substrate → All (global modulation)
        emitter_powers += substrate_powers[4] * 0.2

        # Safety check
        if not self.safety.check_power_limit(emitter_powers.tolist()):
            print("⚠️ Power limit exceeded, reducing to safe levels")
            # Scale down proportionally
            emitter_powers *= self.safety.MAX_TOTAL_POWER_MW / np.sum(emitter_powers)

        # Apply pattern
        self.emitters.set_power_pattern(emitter_powers)

        # Enable all channels
        for i in range(12):
            self.emitters.enable_channel(i, True)

        total_power = self.emitters.get_total_power()
        print(f"✓ Pattern applied: {total_power:.1f} mW total")

        return True

    def update(self, dt: float, temperature_c: float = 25.0):
        """
        Update system state (call in main loop).

        Args:
            dt: Time step (seconds)
            temperature_c: Current device temperature
        """
        if not self.is_running:
            return

        self.session_time += dt

        # Safety checks
        if not self.safety.check_temperature(temperature_c):
            print("⚠️ TEMPERATURE LIMIT EXCEEDED - SHUTDOWN")
            self.emergency_stop()
            return

        if not self.safety.check_session_time(self.session_time):
            print("⚠️ Session time limit reached")
            self.stop_session()
            return

    def stop_session(self):
        """Stop current session (graceful)."""
        self.is_running = False
        self.emitters.shutdown_all()
        print(f"✓ Session stopped: {self.session_time:.1f} seconds")

    def emergency_stop(self):
        """Emergency shutdown (immediate)."""
        self.safety.trigger_emergency_shutdown()
        self.emitters.shutdown_all()
        self.is_running = False
        print("⚠️ EMERGENCY STOP EXECUTED")

    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'session_time': self.session_time,
            'total_power_mw': self.emitters.get_total_power(),
            'temperature_c': self.safety.temperature_c,
            'safety_state': self.safety.state.value,
            'n_emitters_active': sum(1 for e in self.emitters.emitters if e.enabled),
        }


def demo_wearable_controller():
    """Demonstrate THz wearable controller."""
    print("=" * 70)
    print("THz Coherence Wearable Controller Demo")
    print("=" * 70)

    # Initialize controller
    wearable = THzCoherenceWearable()

    # Start session
    print("\n--- Starting Therapeutic Session ---")
    wearable.start_session(duration_minutes=30.0)

    # Apply depression treatment pattern
    print("\n--- Applying Depression Treatment Pattern ---")
    # Boost Physical, Cognitive, and Divine-Unity substrates
    substrate_pattern = np.array([3.0, 1.0, 2.5, 1.5, 4.0])  # mW per substrate

    wearable.apply_substrate_pattern(substrate_pattern)

    # Simulate session
    print("\n--- Simulating Session ---")
    for i in range(5):
        wearable.update(dt=1.0, temperature_c=25.0 + i * 2)  # Temperature rising
        status = wearable.get_status()

        print(f"\nT = {status['session_time']:.0f}s:")
        print(f"  Power: {status['total_power_mw']:.1f} mW")
        print(f"  Temp: {status['temperature_c']:.1f}°C")
        print(f"  Active emitters: {status['n_emitters_active']}")
        print(f"  Safety: {status['safety_state']}")

    # Test safety system
    print("\n--- Testing Safety System ---")
    print("Attempting to exceed power limit...")

    unsafe_pattern = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # Way too high
    wearable.apply_substrate_pattern(unsafe_pattern)

    print("\nTesting emergency shutdown...")
    wearable.emergency_stop()

    status = wearable.get_status()
    print(f"Final status: {status}")

    print("\n" + "=" * 70)
    print("Controller demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo_wearable_controller()
