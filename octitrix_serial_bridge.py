"""
Octitrix Serial Bridge
Connects Arduino hardware controller to the Octitrix FastAPI server
Listens for commands from Arduino and forwards them as HTTP requests
"""

import serial
import serial.tools.list_ports
import requests
import time
import argparse
import sys
from typing import Optional


class OctitrixSerialBridge:
    """
    Bridge between Arduino serial controller and FastAPI server.
    Translates serial commands into HTTP API calls.
    """

    def __init__(self, port: str = None, baud_rate: int = 9600, api_url: str = "http://localhost:8081"):
        """
        Initialize the serial bridge.

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM3'). Auto-detect if None.
            baud_rate: Serial communication speed (default 9600)
            api_url: Base URL of the Octitrix API server
        """
        self.port = port
        self.baud_rate = baud_rate
        self.api_url = api_url.rstrip('/')
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False

    def list_ports(self):
        """List all available serial ports."""
        ports = serial.tools.list_ports.comports()
        print("\n=== Available Serial Ports ===")
        if not ports:
            print("No serial ports found!")
            return []

        for i, port in enumerate(ports):
            print(f"{i+1}. {port.device}")
            print(f"   Description: {port.description}")
            print(f"   HWID: {port.hwid}")
            print()

        return [port.device for port in ports]

    def auto_detect_port(self) -> Optional[str]:
        """
        Auto-detect Arduino port.

        Returns:
            Port name if found, None otherwise
        """
        ports = serial.tools.list_ports.comports()

        # Look for common Arduino identifiers
        arduino_keywords = ['Arduino', 'CH340', 'CP2102', 'USB Serial', 'ttyUSB', 'ttyACM']

        for port in ports:
            port_info = f"{port.description} {port.hwid}".lower()
            if any(keyword.lower() in port_info for keyword in arduino_keywords):
                print(f"[*] Auto-detected Arduino on: {port.device}")
                return port.device

        return None

    def connect_serial(self) -> bool:
        """
        Connect to the Arduino via serial port.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.port:
            self.port = self.auto_detect_port()

        if not self.port:
            print("[!] No Arduino found. Please specify port manually.")
            return False

        try:
            print(f"[*] Connecting to {self.port} at {self.baud_rate} baud...")
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset after connection

            # Wait for ready signal
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if "OCTITRIX:READY" in line:
                        print("[✓] Arduino connected and ready!")
                        return True

            print("[✓] Serial connection established")
            return True

        except serial.SerialException as e:
            print(f"[!] Failed to connect to {self.port}: {e}")
            return False

    def check_api_server(self) -> bool:
        """
        Check if the API server is running.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=2)
            if response.status_code == 200:
                print(f"[✓] API server is online at {self.api_url}")
                return True
            else:
                print(f"[!] API server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[!] Cannot reach API server at {self.api_url}")
            print(f"    Error: {e}")
            print(f"    Make sure the server is running: python octitrix_server.py")
            return False

    def send_command(self, command: str) -> bool:
        """
        Send command to the API server.

        Args:
            command: Command string (e.g., "fractal", "rhythm:4")

        Returns:
            True if successful, False otherwise
        """
        try:
            parts = command.split(':')
            cmd_type = parts[0].lower()

            if cmd_type == "fractal":
                response = requests.post(f"{self.api_url}/shape/fractal", json={})
                print(f"[→] Fractal generated (status: {response.status_code})")

            elif cmd_type == "rhythm":
                pulses = int(parts[1]) if len(parts) > 1 else 4
                response = requests.post(
                    f"{self.api_url}/shape/euclidean",
                    json={"pulses": pulses}
                )
                print(f"[→] Euclidean rhythm generated with {pulses} pulses (status: {response.status_code})")

            elif cmd_type == "dispatch":
                response = requests.post(f"{self.api_url}/dispatch")
                print(f"[→] State dispatched (status: {response.status_code})")

            elif cmd_type == "reset":
                response = requests.post(f"{self.api_url}/reset")
                print(f"[→] Engine reset (status: {response.status_code})")

            else:
                print(f"[!] Unknown command: {cmd_type}")
                return False

            # Send acknowledgment back to Arduino
            if self.serial_conn:
                self.serial_conn.write(b"ACK\n")

            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            print(f"[!] API request failed: {e}")
            return False
        except Exception as e:
            print(f"[!] Command error: {e}")
            return False

    def run(self):
        """Main bridge loop - listen for Arduino commands and forward to API."""
        if not self.serial_conn:
            print("[!] Not connected to serial port")
            return

        if not self.check_api_server():
            print("[!] API server not available. Start it first!")
            return

        print("\n" + "="*60)
        print("=== OCTITRIX SERIAL BRIDGE ACTIVE ===")
        print("="*60)
        print(f"Serial: {self.port} @ {self.baud_rate} baud")
        print(f"API: {self.api_url}")
        print("Listening for Arduino commands... (Ctrl+C to stop)")
        print("="*60 + "\n")

        self.running = True

        try:
            while self.running:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                    if not line:
                        continue

                    print(f"[←] Arduino: {line}")

                    # Handle commands
                    if line.startswith("CMD:"):
                        command = line[4:]  # Remove "CMD:" prefix
                        self.send_command(command)

                    elif line.startswith("OCTITRIX:"):
                        # Status messages from Arduino
                        print(f"    {line}")

                time.sleep(0.01)  # Small delay to prevent CPU spinning

        except KeyboardInterrupt:
            print("\n[*] Stopping bridge...")
            self.running = False

        except Exception as e:
            print(f"[!] Error in main loop: {e}")
            self.running = False

        finally:
            if self.serial_conn:
                self.serial_conn.close()
                print("[*] Serial connection closed")

    def disconnect(self):
        """Disconnect from serial port."""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None


def main():
    parser = argparse.ArgumentParser(
        description="Octitrix Serial Bridge - Connect Arduino to FastAPI server"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port (e.g., /dev/ttyUSB0 or COM3). Auto-detect if not specified."
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Baud rate (default: 9600)"
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8081",
        help="API server URL (default: http://localhost:8081)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available serial ports and exit"
    )

    args = parser.parse_args()

    bridge = OctitrixSerialBridge(
        port=args.port,
        baud_rate=args.baud,
        api_url=args.api
    )

    if args.list:
        bridge.list_ports()
        sys.exit(0)

    # Connect to serial
    if not bridge.connect_serial():
        print("\n[!] Failed to connect to Arduino")
        print("    Available ports:")
        bridge.list_ports()
        sys.exit(1)

    # Run bridge
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        bridge.disconnect()
        print("[*] Bridge shutdown complete")


if __name__ == "__main__":
    main()
