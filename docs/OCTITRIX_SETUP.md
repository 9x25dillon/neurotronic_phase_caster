# Octitrix 8D Audio System - Setup Guide

## Overview

**Octitrix** is an 8-dimensional audio structure generator that combines fractal mathematics, chaos theory, and euclidean rhythms to create complex spatial audio patterns. The system consists of three main components:

1. **Octitrix Engine** - Core mathematics and pattern generation
2. **FastAPI Server** - REST API for remote control
3. **Arduino Controller** - Hardware interface with optional Serial Bridge

## Architecture

```
┌─────────────────┐
│ Arduino         │
│ Controller      │
│ (Physical HW)   │
└────────┬────────┘
         │ Serial (9600 baud)
         ▼
┌─────────────────┐
│ Serial Bridge   │
│ (Python)        │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│ FastAPI Server  │
│ (Port 8081)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Octitrix Engine │
│ (8D Generator)  │
└─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `numpy` - Numerical computing
- `pyserial` - Serial communication (for Arduino)
- `requests` - HTTP client (for Serial Bridge)

### 2. Test the Engine

```bash
python octitrix_engine.py
```

You should see output demonstrating fractal and euclidean pattern generation.

### 3. Start the API Server

Choose one of three modes:

#### Server Mode (API only)
```bash
python octitrix_server.py --mode server
```

#### CLI Mode (Command line only)
```bash
python octitrix_server.py --mode cli
```

#### Hybrid Mode (Both API and CLI)
```bash
python octitrix_server.py --mode hybrid
```

The server will start on `http://localhost:8081`

### 4. Access API Documentation

Open your browser to:
```
http://localhost:8081/docs
```

This provides interactive Swagger documentation for all API endpoints.

## API Usage

### Available Endpoints

#### `GET /`
Health check and status
```bash
curl http://localhost:8081/
```

#### `GET /status`
Get current engine state
```bash
curl http://localhost:8081/status
```

#### `POST /shape/fractal`
Generate fractal structure
```bash
curl -X POST http://localhost:8081/shape/fractal \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'
```

#### `POST /shape/euclidean`
Generate euclidean rhythm
```bash
curl -X POST http://localhost:8081/shape/euclidean \
  -H "Content-Type: application/json" \
  -d '{"pulses": 5, "steps": 8}'
```

#### `POST /dispatch`
Re-send current state to DAW
```bash
curl -X POST http://localhost:8081/dispatch
```

#### `POST /reset`
Reset engine to initial state
```bash
curl -X POST http://localhost:8081/reset
```

## CLI Commands

When running in `cli` or `hybrid` mode, you can use these commands:

- `fractal [seed]` - Generate fractal structure (optional seed)
- `rhythm <pulses>` - Generate euclidean rhythm with N pulses
- `dispatch` - Send current state to DAW
- `status` - Show engine status
- `reset` - Reset engine state
- `exit` - Exit CLI mode

Example session:
```
Octitrix> fractal 12345
>> Fractal Structure Generated & Sent (seed: 12345)

Octitrix> rhythm 7
>> Euclidean Structure (7 pulses) Generated & Sent

Octitrix> status
>> Engine Status:
   Dimensions: 8
   Has Matrix: True
   Current Type: euclidean
```

## Arduino Hardware Setup

### 1. Hardware Requirements

- Arduino Uno, Nano, or Mega
- (Optional) 4 push buttons for manual control
- (Optional) Potentiometer for parameter control
- USB cable for serial connection

### 2. Pin Configuration

| Component       | Pin  | Purpose                    |
|----------------|------|----------------------------|
| Fractal Button | D2   | Trigger fractal generation |
| Rhythm Button  | D3   | Trigger rhythm generation  |
| Dispatch Button| D4   | Re-send current state      |
| Reset Button   | D5   | Reset engine               |
| Potentiometer  | A0   | Control pulse count (1-16) |

### 3. Upload Sketch

1. Open `arduino/octitrix_controller.ino` in Arduino IDE
2. Select your board type (Tools > Board)
3. Select your serial port (Tools > Port)
4. Click Upload

### 4. Verify Connection

Open Serial Monitor (9600 baud) - you should see:
```
OCTITRIX:READY
OCTITRIX:MODE:AUTO
```

## Serial Bridge Setup

The Serial Bridge connects your Arduino to the FastAPI server.

### 1. Start the FastAPI Server

```bash
python octitrix_server.py --mode server
```

### 2. List Available Ports

```bash
python octitrix_serial_bridge.py --list
```

### 3. Start the Bridge

Auto-detect Arduino:
```bash
python octitrix_serial_bridge.py
```

Specify port manually:
```bash
python octitrix_serial_bridge.py --port /dev/ttyUSB0
```

On Windows:
```bash
python octitrix_serial_bridge.py --port COM3
```

### 4. Custom API URL

```bash
python octitrix_serial_bridge.py --api http://192.168.1.100:8081
```

## Complete System Startup

To run the complete Octitrix system with hardware control:

### Terminal 1: Start API Server
```bash
python octitrix_server.py --mode server --port 8081
```

### Terminal 2: Start Serial Bridge
```bash
python octitrix_serial_bridge.py
```

### Terminal 3 (Optional): Monitor with CLI
```bash
python octitrix_server.py --mode cli
```

Now your Arduino can control the Octitrix engine via buttons/sensors!

## Operating Modes

The Arduino controller supports two modes:

### Auto-Demo Mode (Default)
- Automatically generates patterns every 5 seconds
- Alternates between fractal and rhythm
- Potentiometer controls rhythm pulse count
- Good for demonstrations and testing

### Manual Mode
- Triggered by button presses only
- Full control via hardware interface
- Potentiometer sets rhythm parameters

Switch modes by sending `MODE:MANUAL` or `MODE:AUTO` via serial.

## Integration with DAW

The `dispatch()` function sends the current state to your Digital Audio Workstation.

To integrate with your DAW, modify `octitrix_engine.py` to add:
- **OSC** (Open Sound Control) - for Ableton, Bitwig, Reaper
- **MIDI** - for general MIDI CC control
- **WebSocket** - for web-based DAWs

Example OSC integration:
```python
from pythonosc import udp_client

class Octitrix:
    def __init__(self):
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

    def dispatch(self):
        # Send matrix to DAW via OSC
        for i, row in enumerate(self.current_matrix):
            for j, value in enumerate(row):
                self.osc_client.send_message(f"/octitrix/{i}/{j}", value)
```

## Troubleshooting

### Server won't start
- Check if port 8081 is already in use
- Try a different port: `--port 8082`

### Arduino not detected
- Check USB connection
- Install CH340/CP2102 drivers if needed
- Use `--list` to see available ports
- Specify port manually with `--port`

### Bridge can't reach API
- Ensure API server is running first
- Check firewall settings
- Verify API URL with `--api` parameter

### Serial communication errors
- Check baud rate matches (9600)
- Ensure Arduino sketch is uploaded
- Try different USB cable/port
- Check Arduino IDE Serial Monitor works first

## Advanced Configuration

### Custom Port
```bash
python octitrix_server.py --port 9000
```

### Custom Host (Network Access)
```bash
python octitrix_server.py --host 192.168.1.100 --port 8081
```

### Custom Dimensions
Edit `octitrix_engine.py`:
```python
engine = Octitrix(dimensions=16)  # 16D instead of 8D
```

## Mathematical Background

### Fractal Generation
Uses the **logistic map** from chaos theory:
```
x(n+1) = r * x(n) * (1 - x(n))
```
With r=3.9 (deep in chaotic regime) to generate unpredictable yet deterministic patterns.

### Euclidean Rhythms
Implements **Bjorklund's algorithm** to distribute k pulses over n steps as evenly as possible. Used in music from many cultures (Cuban tresillo, African bell patterns, etc.).

## License

Part of the Neurotronic Phase Caster project.

## Support

For issues or questions, refer to the main project documentation.
