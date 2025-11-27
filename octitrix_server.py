"""
Octitrix 8D Audio Server
FastAPI server providing REST API for 8D audio structure generation
Supports hybrid mode with CLI for real-time performance
"""

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from octitrix_engine import Octitrix
import threading
import time
from typing import Optional


# --- PYDANTIC MODELS ---

class FractalRequest(BaseModel):
    seed: Optional[int] = None


class EuclideanRequest(BaseModel):
    pulses: int = 4
    steps: Optional[int] = None


# --- INIT ---
app = FastAPI(
    title="Octitrix 8D Server",
    description="API for shaping 8D Audio Structures using fractal mathematics and euclidean rhythms",
    version="1.0.0"
)
engine = Octitrix()


# --- API ROUTES (For External Tools/Web Apps) ---

@app.get("/")
def root():
    """Root endpoint - health check and info."""
    return {
        "message": "Octitrix 8D Engine Online. Connect DAW to port 8081.",
        "status": "ready",
        "engine_state": engine.get_state()
    }


@app.get("/status")
def get_status():
    """Get current engine status and state."""
    return engine.get_state()


@app.post("/shape/fractal")
def shape_fractal(request: FractalRequest = None):
    """
    Generates a chaos/fractal based structure.

    Args:
        seed: Random seed for reproducible patterns (optional, defaults to timestamp)

    Returns:
        Generated fractal matrix and metadata
    """
    try:
        seed = request.seed if request and request.seed is not None else int(time.time())
        matrix = engine.generate_fractal(seed)
        result = engine.dispatch()

        return {
            "type": "fractal",
            "matrix": matrix,
            "seed": seed,
            "dispatch_status": result["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fractal generation failed: {str(e)}")


@app.post("/shape/euclidean")
def shape_euclidean(request: EuclideanRequest):
    """
    Generates a rhythmic geometric structure using euclidean rhythms.

    Args:
        pulses: Number of active pulses in the rhythm (default 4)
        steps: Total number of steps (optional, defaults to 8)

    Returns:
        Generated euclidean matrix and metadata
    """
    try:
        matrix = engine.generate_euclidean(
            pulses=request.pulses,
            steps=request.steps
        )
        result = engine.dispatch()

        return {
            "type": "euclidean",
            "matrix": matrix,
            "pulses": request.pulses,
            "steps": request.steps,
            "dispatch_status": result["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Euclidean generation failed: {str(e)}")


@app.post("/dispatch")
def force_dispatch():
    """
    Re-sends current state to DAW.

    Returns:
        Current engine state and dispatch confirmation
    """
    try:
        return engine.dispatch()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dispatch failed: {str(e)}")


@app.post("/reset")
def reset_engine():
    """Reset the engine to initial state."""
    engine.reset()
    return {
        "status": "reset",
        "message": "Engine has been reset to initial state"
    }


# --- CLI & THREADING ---

def cli_loop():
    """
    A live command line interface for real-time performance.
    Allows the user to type commands while the server runs.
    """
    print("\n" + "="*50)
    print("--- OCTITRIX CLI MODE ---")
    print("="*50)
    print("Commands:")
    print("  fractal [seed]  - Generate fractal structure")
    print("  rhythm <pulses> - Generate euclidean rhythm")
    print("  dispatch        - Send current state to DAW")
    print("  status          - Show engine status")
    print("  reset           - Reset engine state")
    print("  exit            - Exit CLI mode")
    print("="*50 + "\n")

    while True:
        try:
            cmd = input("Octitrix> ").strip().split()
            if not cmd:
                continue

            if cmd[0] == "exit":
                print(">> Exiting CLI mode...")
                break

            elif cmd[0] == "fractal":
                seed = int(cmd[1]) if len(cmd) > 1 else int(time.time())
                engine.generate_fractal(seed)
                engine.dispatch()
                print(f">> Fractal Structure Generated & Sent (seed: {seed})")

            elif cmd[0] == "rhythm":
                pulses = int(cmd[1]) if len(cmd) > 1 else 4
                engine.generate_euclidean(pulses)
                engine.dispatch()
                print(f">> Euclidean Structure ({pulses} pulses) Generated & Sent")

            elif cmd[0] == "dispatch":
                result = engine.dispatch()
                print(f">> Signal Dispatched - Status: {result['status']}")

            elif cmd[0] == "status":
                state = engine.get_state()
                print(f">> Engine Status:")
                print(f"   Dimensions: {state['dimensions']}")
                print(f"   Has Matrix: {state['has_matrix']}")
                print(f"   Current Type: {state['current_state'].get('type', 'None')}")

            elif cmd[0] == "reset":
                engine.reset()
                print(">> Engine Reset")

            else:
                print(f"Unknown command: {cmd[0]}")
                print("Type 'exit' to quit or see available commands above")

        except KeyboardInterrupt:
            print("\n>> Interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")


# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Octitrix 8D Audio Tool - Fractal & Geometric Audio Structure Generator"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["server", "cli", "hybrid"],
        help="Operating mode: server (API only), cli (command line only), or hybrid (both)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Server port (default: 8081)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    if args.mode == "server" or args.mode == "hybrid":
        # Start CLI in background thread if hybrid
        if args.mode == "hybrid":
            cli_thread = threading.Thread(target=cli_loop, daemon=True)
            cli_thread.start()

        print(f"[*] Starting Octitrix API Server on {args.host}:{args.port}...")
        print(f"[*] Mode: {args.mode}")
        print(f"[*] API Documentation: http://localhost:{args.port}/docs")

        uvicorn.run(app, host=args.host, port=args.port)

    elif args.mode == "cli":
        cli_loop()
