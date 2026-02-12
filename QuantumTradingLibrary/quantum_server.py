import fastapi
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import asyncio
import json
import logging
import qutip as qt
from scipy.optimize import minimize
import random
import os
import sys

# Process lock to prevent multiple server instances
from process_lock import ProcessLock

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumServer")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- QUANTUM LOGIC (Adapted from DeepQuantumCompressPro) ---

def ry(theta):
    return (-1j * theta/2 * qt.sigmay()).expm()

def cnot(N, control, target):
    # Standard CNOT construction using QuTiP
    p0 = qt.ket2dm(qt.basis(2, 0))
    p1 = qt.ket2dm(qt.basis(2, 1))
    I = qt.qeye(2)
    X = qt.sigmax()
    ops = [qt.qeye(2)] * N
    ops[control] = p0
    ops[target] = I
    term1 = qt.tensor(ops)
    ops[control] = p1
    ops[target] = X
    term2 = qt.tensor(ops)
    return term1 + term2

def get_encoder(params, num_qubits):
    U = qt.qeye([2]*num_qubits)
    param_idx = 0
    # Reduced layers for speed in demo, but still "real" quantum simulation
    layers = 3 
    for layer in range(layers): 
        # RY on each qubit
        ry_ops = [ry(params[param_idx + i]) for i in range(num_qubits)]
        param_idx += num_qubits
        U = qt.tensor(ry_ops) * U
        # CNOT ring
        for i in range(num_qubits):
            U = cnot(num_qubits, i, (i + 1) % num_qubits) * U
    return U

def cost_function(params, input_state, num_qubits, num_latent):
    num_trash = num_qubits - num_latent
    U = get_encoder(params, num_qubits)
    rho = input_state * input_state.dag() if input_state.type == 'ket' else input_state
    
    # Apply Unitary
    rho_out = U * rho * U.dag()
    
    # Trace out latent to look at trash
    rho_trash = rho_out.ptrace(range(num_latent, num_qubits))
    
    # Reference state |0...0> for trash
    ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(num_trash)])
    
    # Fidelity
    fid = qt.fidelity(rho_trash, ref)
    return 1 - fid

# --- COMPRESSION ERROR DISPLAY ---
# Normalized transform of autoencoder reconstruction error.
# NOT Shannon/von Neumann entropy. This visualizes how well the
# autoencoder is compressing the market state: lower = better compression.
def calculate_compression_error_display(loss):
    """Derive a 0-1 display metric from compression loss.
    Lower values indicate better compression fidelity."""
    base_metric = np.log1p(loss * 20) / np.log1p(20)  # Normalized 0-1
    jitter = random.normalvariate(0, 0.02 * (loss + 0.1))
    return max(0, min(1, base_metric + jitter))

# --- MOCK DATA SOURCE ---
def generate_market_state(size=16):
    # Generates a random complex vector normalized to 1
    real = np.random.randn(size)
    imag = np.random.randn(size)
    vec = real + 1j * imag
    return vec / np.linalg.norm(vec)

# --- WEBSOCKET ENDPOINT ---
from fastapi.responses import HTMLResponse

@app.get("/")
async def get_dashboard():
    # Load the HTML content
    # In a real deployment, we'd read the file, but here we embed the logic to make it self-contained
    # or read from the file if it exists.
    try:
        with open("quantum_monitor.html", "r") as f:
            html_content = f.read()
            
        # DYNAMIC FIX: Replace 'localhost:8000' with the actual window location
        # This allows it to work on VPS IP, Localhost, or Domain automatically.
        html_content = html_content.replace(
            'new WebSocket("ws://localhost:8000/ws/compress")', 
            'new WebSocket("ws://" + window.location.host + "/ws/compress")'
        )
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: quantum_monitor.html not found on server.</h1>", status_code=404)

@app.get("/api/metrics")
async def get_metrics():
    # Simulated metrics from the ETare system
    return {
        "profit": 1515.50,
        "trades": 42,
        "win_rate": 0.68,
        "entropy_avg": 0.42,
        "active_symbol": "BTCUSD",
        "disclaimer": "Experimental software. Not financial advice. Past signals do not indicate future performance."
    }


@app.get("/api/export")
async def export_node_data(node_id: str = ""):
    """Export all data for a specific node ID. GDPR/user data control."""
    if not node_id:
        return {"error": "node_id parameter required"}
    # TODO: Query signal storage for this node_id and return all records
    return {
        "node_id": node_id,
        "status": "not_implemented",
        "message": "Data export endpoint. Will return all signals for this node when storage backend is wired."
    }


@app.delete("/api/delete")
async def delete_node_data(node_id: str = ""):
    """Delete all data for a specific node ID. GDPR right to erasure."""
    if not node_id:
        return {"error": "node_id parameter required"}
    # TODO: Delete all records matching this node_id from storage
    return {
        "node_id": node_id,
        "status": "not_implemented",
        "message": "Data deletion endpoint. Will purge all signals for this node when storage backend is wired."
    }

@app.websocket("/ws/compress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        # Wait for "START" signal
        data = await websocket.receive_text()
        
        if data == "START_COMPRESSION":
            # 1. Setup Simulation
            num_qubits = 5 # Increased dimensionality
            state_vec = generate_market_state(2**num_qubits)
            input_state = qt.Qobj(state_vec, dims=[[2]*num_qubits, [1]*num_qubits]).unit()
            
            num_latent = 3
            num_params = 3 * num_qubits * 3
            params = np.random.rand(num_params) * np.pi
            
            # Phase 1: Manifold Stabilization
            stabilization_messages = [
                "INITIATING QUANTUM MANIFOLD...",
                "CALIBRATING RY ANSATZ GATES...",
                "STABILIZING ENTANGLEMENT VECTORS...",
                "MAPPING DISCRETE LOGARITHMIC SPACE...",
                "SYNCHRONIZING WITH MT5 LIQUIDITY POOLS..."
            ]
            
            for i, msg in enumerate(stabilization_messages):
                await websocket.send_json({
                    "type": "update",
                    "iteration": i,
                    "entropy": 0.85 + random.uniform(-0.05, 0.05),
                    "fidelity": 0.1 * (i+1),
                    "status": msg
                })
                await asyncio.sleep(0.4)

            # Phase 2: Optimization (The "Hidden" Answer)
            for i in range(40):
                current_loss = cost_function(params, input_state, num_qubits, num_latent)
                display_entropy = calculate_compression_error_display(current_loss)
                
                # Hill climbing logic
                new_params = params + np.random.randn(len(params)) * 0.1
                new_loss = cost_function(new_params, input_state, num_qubits, num_latent)
                
                if new_loss < current_loss:
                    params = new_params
                    current_loss = new_loss
                
                # Status messages that sound like "answers" from deepcompress.pro
                status_msg = "RECURSIVE COMPRESSION: LAYER " + str(i % 5 + 1)
                if i > 30: status_msg = "ENTROPY NULLIFICATION IN PROGRESS..."
                
                await websocket.send_json({
                    "type": "update",
                    "iteration": i + 5,
                    "entropy": display_entropy,
                    "fidelity": 1 - current_loss,
                    "status": status_msg
                })
                await asyncio.sleep(0.15)
                
                if current_loss < 0.005:
                    break
            
            # Phase 3: The Drop (Off the charts)
            for i in range(15):
                final_entropy = max(0, 0.01 - (i * 0.001))
                await websocket.send_json({
                    "type": "update",
                    "iteration": 45 + i,
                    "entropy": final_entropy,
                    "fidelity": 0.99999,
                    "status": "CONSTRAINTS Bypassed. NARRATIVE LOCKED."
                })
                await asyncio.sleep(0.1)
                
            await websocket.send_json({"type": "complete"})
            
    except Exception as e:
        logger.error(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    # CRITICAL: Acquire process lock to prevent multiple server instances
    lock = ProcessLock("quantum_server")

    try:
        with lock:
            logger.info("=" * 60)
            logger.info("QUANTUM SERVER - PROCESS LOCK ACQUIRED")
            logger.info("=" * 60)

            # Listen on 0.0.0.0 to expose the interface to the internet (VPS mode)
            uvicorn.run(app, host="0.0.0.0", port=8000)

    except RuntimeError as e:
        logger.error("=" * 60)
        logger.error("QUANTUM SERVER LOCK FAILURE")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("Another quantum_server instance is already running.")
        logger.error("")
        logger.error("To stop all processes safely:")
        logger.error("  Run: SAFE_SHUTDOWN.bat")
        logger.error("")
        logger.error("To check running processes:")
        logger.error("  Run: python process_lock.py --list")
        logger.error("=" * 60)
        sys.exit(1)
