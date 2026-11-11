"""
System and health API routes for PruneQuest Studio.
"""

import os
import platform
from fastapi import APIRouter
import torch
from prunequest import __version__
from prunequest.orchestrator.hardware import HardwareInspector
from prunequest.studio.schemas import SystemInfoResponse

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health")
def get_health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@router.get("/system", response_model=SystemInfoResponse)
def get_system_info():
    """Returns detected hardware, PyTorch, CUDA, and memory information."""
    report = HardwareInspector.inspect()
    return SystemInfoResponse(
        os_name=platform.system(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=report.cuda_available,
        device_name=report.device_name,
        vram_total_gb=report.total_vram_gb,
        vram_free_gb=report.free_vram_gb,
        ram_total_gb=report.system_ram_gb,
        ram_free_gb=report.available_ram_gb,
        cpu_cores=os.cpu_count() or 1,
    )
