#!/usr/bin/env python3
"""
vllm_injector.py - Dynamic Allocator Patching & Restore Automation

This script locates the vLLM package, backs up the CpuGpuBlockAllocator implementation,
and safely injects the Project Charon Predictive memory management integration.
It also provides restoration commands to revert the patch instantly.
"""

import os
import sys
import argparse
import shutil
import importlib.util

def find_vllm_allocator_path() -> str:
    """
    Programmatically locates the path to vllm/core/block_manager/cpu_gpu_block_allocator.py.
    Checks site packages and imports, falling back to local workspace workspace directories.
    """
    # Check if a local stub is present in the workspace first for sandbox isolation
    local_path = os.path.join(os.getcwd(), "vllm", "core", "block_manager", "cpu_gpu_block_allocator.py")
    if os.path.exists(local_path):
        return os.path.abspath(local_path)

    # Attempt to locate the global/virtualenv vLLM installation package
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is not None and vllm_spec.submodule_search_locations:
        vllm_dir = vllm_spec.submodule_search_locations[0]
        global_path = os.path.join(vllm_dir, "core", "block_manager", "cpu_gpu_block_allocator.py")
        if os.path.exists(global_path):
            return os.path.abspath(global_path)

    # Fallback search in site-packages
    try:
        import site
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            possible_path = os.path.join(site_dir, "vllm", "core", "block_manager", "cpu_gpu_block_allocator.py")
            if os.path.exists(possible_path):
                return os.path.abspath(possible_path)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find vLLM CpuGpuBlockAllocator path globally or locally in workspace.\n"
        "Please ensure vLLM is installed or run in a sandbox workspace with 'vllm/core/block_manager/' stub structures."
    )


def inject_patch(file_path: str):
    """
    Injects charon_core predictive swap hooks directly into the vLLM allocator source file.
    """
    backup_path = file_path + ".backup"
    
    # 1. Handle Backup Integrity
    if not os.path.exists(backup_path):
        shutil.copyfile(file_path, backup_path)
        print(f"[Injector] Backup successfully created at: {backup_path}")
    else:
        print(f"[Injector] Backup already exists at: {backup_path}. Re-using it.")

    with open(backup_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Prevent Double Injection
    if "import charon_core" in content or "charon_engine" in content:
        print("[Injector] File has already been modified or contains Charon imports. Skipping.")
        return

    # 3. Formulate the code modifications
    print("[Injector] Preparing code injections...")
    
    # Add imports at the top
    imported_header = (
        "# --- BEGIN PROJECT CHARON INJECTIONS ---\n"
        "import charon_core\n"
        "import time\n"
        "global_charon_engine = charon_core.PredictiveEngine()\n"
        "# --- END PROJECT CHARON INJECTIONS ---\n\n"
    )
    
    modified_content = imported_header + content

    # Intercept allocations or swap_out methods. 
    # Here we replace the reactive swap_out with proactive eviction logic
    target_pattern = "def swap_out(self, session_id: str):"
    if target_pattern not in modified_content:
        # Try generic/fallback pattern for different vLLM versions
        target_pattern = "def swap_out(self, block"
        
    if target_pattern not in modified_content:
        raise ValueError(f"Could not locate 'swap_out' method signature in {file_path}")

    # Build the custom preemptive logic replacement
    patched_swap_out = (
        "def swap_out(self, session_id: str):\n"
        "        # --- CHARON EVICTION HOOKS ---\n"
        "        # Periodically monitor HMM state changes to preemptively swap\n"
        "        # out the lowest probability human interactive session\n"
        "        timestamp = time.time() * 1000.0\n"
        "        signal = charon_core.RequestSignal(session_id, timestamp, 256)\n"
        "        global_charon_engine.notify_request(signal)\n"
        "        \n"
        "        # Check if we should execute proactive eviction\n"
        "        decision = global_charon_engine.get_optimal_eviction_target()\n"
        "        if decision.should_evict and decision.session_id != session_id:\n"
        "            print(f'[Charon HMM Injector] Proactively evicting session: {decision.session_id}')\n"
        "            # Execute swap on the target session instead of standard self\n"
        "            target_to_swap = decision.session_id\n"
        "        else:\n"
        "            target_to_swap = session_id\n"
        "            \n"
        "        print(f'[Charon HMM Injector] Forwarding execution block to native allocator for: {target_to_swap}')\n"
        "        # Native logic starts below\n"
    )

    # Perform the injection replacement
    if "def swap_out(self, session_id: str):" in modified_content:
        modified_content = modified_content.replace(
            "def swap_out(self, session_id: str):",
            patched_swap_out
        )
    elif "def swap_out(self, block" in modified_content:
        # Fallback support for block structures
        patched_swap_out_blocks = patched_swap_out.replace("session_id: str", "block")
        modified_content = modified_content.replace(
            "def swap_out(self, block",
            patched_swap_out_blocks
        )

    # 4. Write back the patched file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"[Injector] Successfully patched vLLM allocator file: {file_path}")


def restore_backup(file_path: str):
    """
    Reverts all file modifications and restores the native vLLM implementation from backup.
    """
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        print(f"[Injector] Error: Backup file not found at {backup_path}. Reversion aborted.")
        return

    shutil.copyfile(backup_path, file_path)
    os.remove(backup_path)
    print(f"[Injector] Successfully restored original vLLM state. Backup file removed.")


def main():
    parser = argparse.ArgumentParser(description="Project Charon Memory Allocator Patcher")
    parser.add_argument("--restore", action="store_true", help="Revert the monkey patch and restore original file")
    args = parser.parse_args()

    try:
        allocator_path = find_vllm_allocator_path()
        print(f"[Injector] Found allocator path at: {allocator_path}")
        
        if args.restore:
            restore_backup(allocator_path)
        else:
            inject_patch(allocator_path)
            
    except Exception as ex:
        print(f"[Injector CRITICAL ERROR] {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
