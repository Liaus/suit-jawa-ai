"""Collect basic CPU, memory, GPU, and operating-system details."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any


def _windows_hardware() -> dict[str, Any]:
    script = (
        "$processors = @(Get-CimInstance Win32_Processor); "
        "$cpu = $processors[0].Name; "
        "$physical = ($processors | Measure-Object -Property NumberOfCores -Sum).Sum; "
        "$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / "
        "1GB, 2); "
        "$gpu = @(Get-CimInstance Win32_VideoController | "
        "ForEach-Object { $_.Name }); "
        "[PSCustomObject]@{ cpu = $cpu; physical_cores = $physical; "
        "ram_gb = $ram; gpu = $gpu } | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("gpu"), str):
        data["gpu"] = [data["gpu"]]
    return data


def _linux_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {}
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        text = cpu_info.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            info["cpu"] = match.group(1).strip()
    memory_info = Path("/proc/meminfo")
    if memory_info.exists():
        text = memory_info.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^MemTotal:\s*(\d+)\s*kB$", text, re.MULTILINE)
        if match:
            info["ram_gb"] = round(int(match.group(1)) / 1024 / 1024, 2)
    return info


def _nvidia_gpus() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_system_info() -> dict[str, Any]:
    """Return a stable set of machine facts for benchmark reports."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Tidak terdeteksi"),
        "physical_cores": "Tidak terdeteksi",
        "logical_cores": os.cpu_count() or "Tidak terdeteksi",
        "ram_gb": "Tidak terdeteksi",
        "gpu": [],
    }

    if platform.system() == "Windows":
        hardware = _windows_hardware()
    elif platform.system() == "Linux":
        hardware = _linux_hardware()
    else:
        hardware = {}

    info["cpu"] = hardware.get("cpu") or info["cpu"]
    info["ram_gb"] = hardware.get("ram_gb", info["ram_gb"])
    info["physical_cores"] = hardware.get("physical_cores", info["physical_cores"])
    info["gpu"] = hardware.get("gpu") or _nvidia_gpus()
    return info
