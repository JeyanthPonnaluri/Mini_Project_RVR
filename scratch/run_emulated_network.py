import subprocess
import time
import sys
import os

def main():
    print("==================================================")
    print("RUNNING EMULATED SYSTEM NETWORK LATENCY TEST")
    print("==================================================")
    
    # Path to server and client scripts
    emulated_dir = "D:/Mini_project_JP/src/emulated"
    python_exe = "C:/Users/HP/AppData/Local/Programs/Python/Python312/python.exe"
    
    # Check if packages are installed
    try:
        import fastapi
        import websockets
        import uvicorn
        print("[OK] FastAPI, websockets, and uvicorn are installed.")
    except ImportError:
        print("Installing websockets, fastapi, and uvicorn in Python 3.12...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "websockets", "fastapi", "uvicorn"])
        print("[OK] Package installation completed.")

    # Start FastAPI server in the background
    print("\n[SYSTEM] Starting FastAPI WebSocket server on port 8089...")
    server_process = subprocess.Popen([
        python_exe, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8089"
    ], cwd=emulated_dir)
    
    # Give the server a moment to boot
    time.sleep(2.0)
    
    start_time = time.time()
    
    # Start 3 client nodes in parallel
    print("[SYSTEM] Spawning 3 active client nodes (Hospital_A, Hospital_B, Hospital_C)...")
    clients = []
    for client_name in ["Hospital_A", "Hospital_B", "Hospital_C"]:
        # Add 50ms artificial delay to simulate network latency
        time.sleep(0.05)
        p = subprocess.Popen([
            python_exe, "client.py", client_name, "ws://127.0.0.1:8089/ws"
        ], cwd=emulated_dir)
        clients.append(p)
        
    # Wait for all clients to finish
    for p in clients:
        p.wait()
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n[SYSTEM] Emulated 3-node training round finished in: {elapsed_time:.3f} seconds")
    
    # Kill the server
    print("[SYSTEM] Shutting down FastAPI WebSocket server...")
    server_process.terminate()
    server_process.wait()
    print("[OK] Server shutdown successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
