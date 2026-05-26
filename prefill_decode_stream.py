#!/usr/bin/env python3
"""
prefill_decode_stream.py - Ultra-Fast KV-Cache Network Streaming

This module provides socket shims simulating high-speed RoCE/RDMA streaming transfers
of serialized, 3-bit compressed PagedAttention KV blocks between Prefill nodes
and Autoregressive Decode nodes.
"""

import socket
import threading
import time
import pickle
import numpy as np
from charon_quant import compress_kv_cache_numpy

# Network Streaming configuration
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 9811

class KVCacheStreamServer:
    """
    Simulates the receiver service running on the Autoregressive Decode node fleet.
    Listens on high-speed RDMA endpoints and loads incoming cache blocks directly into memory.
    """
    def __init__(self, host: str = STREAM_HOST, port: int = STREAM_PORT):
        self.host = host
        self.port = port
        self.received_cache = {}
        self.is_running = False
        self.server_thread = None

    def start(self):
        self.is_running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        
        self.server_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.server_thread.start()
        print(f"[RDMA Stream Server] Listening for prefill cache on: tcp://{self.host}:{self.port}")

    def _listen_loop(self):
        while self.is_running:
            try:
                conn, addr = self.sock.accept()
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception:
                break

    def _handle_client(self, conn: socket.socket):
        data_buffer = bytearray()
        while True:
            packet = conn.recv(65536)
            if not packet:
                break
            data_buffer.extend(packet)
        conn.close()

        if data_buffer:
            try:
                payload = pickle.loads(data_buffer)
                session_id = payload["session_id"]
                self.received_cache[session_id] = payload["kv_data"]
                print(f"[RDMA Stream Server] Successfully loaded streamed context blocks for '{session_id}' | "
                      f"Blocks: {len(payload['kv_data'])} | Size: {len(data_buffer) / 1024:.2f} KB")
            except Exception as ex:
                print(f"[RDMA Stream Server] Error decoding payload stream: {ex}")

    def stop(self):
        self.is_running = False
        if hasattr(self, "sock"):
            self.sock.close()
        if self.server_thread:
            self.server_thread.join(timeout=1.0)


class KVCacheStreamClient:
    """
    Simulates the streaming service running on compute-dense Prefill nodes.
    Serializes and streams generated prompt caches over the PCIe/network bus.
    """
    def __init__(self, host: str = STREAM_HOST, port: int = STREAM_PORT):
        self.host = host
        self.port = port

    def stream_cache_to_decode_node(self, session_id: str, magnitudes: np.ndarray, packed_indices: np.ndarray) -> float:
        """
        Streams compressed block tensors across network boundaries.
        """
        payload = {
            "session_id": session_id,
            "kv_data": {
                "magnitudes": magnitudes,
                "packed_indices": packed_indices
            }
        }
        
        serialized = pickle.dumps(payload)
        
        start_time = time.perf_counter()
        
        # Open socket and stream payload immediately
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            sock.sendall(serialized)
            sock.close()
        except Exception as ex:
            print(f"[RDMA Stream Client] Failed to connect to decode node: {ex}")
            return -1.0
            
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return duration_ms


if __name__ == "__main__":
    print("=" * 80)
    print("                 PROJECT CHARON: DISAGGREGATED RDMA STREAM SHIM")
    print("=" * 80)
    
    # 1. Start target receiver server on decode fleet
    server = KVCacheStreamServer()
    server.start()
    
    time.sleep(0.5)
    
    # 2. Simulate prefill GPU generating and compressing context cache
    print("\n[Prefill GPU] Processing prompt for 'session_agent_omega'...")
    # Mock context of 100 tokens, head dim 128
    mock_kv = np.random.uniform(-1.0, 1.0, size=(100, 128)).astype(np.float16)
    compressed = compress_kv_cache_numpy(mock_kv)
    
    # 3. Stream compressed caches across disaggregated cluster nodes via client
    print("[Prefill GPU] Initiating high-speed RDMA stream to decode node...")
    client = KVCacheStreamClient()
    duration = client.stream_cache_to_decode_node(
        session_id="session_agent_omega",
        magnitudes=compressed["magnitudes"],
        packed_indices=compressed["packed_indices"]
    )
    
    if duration > 0:
        # Measure metrics
        bandwidth_mbps = (compressed["magnitudes"].nbytes + compressed["packed_indices"].nbytes) * 8 / (duration / 1000) / (1024**2)
        print(f"\n[RDMA Stream Client] Transfer successfully completed in: {duration:.3f}ms")
        print(f"[RDMA Stream Client] Simulated Network Throughput: {bandwidth_mbps:.2f} Mbps")
        
    time.sleep(0.5)
    server.stop()
    print("\n" + "=" * 80)
