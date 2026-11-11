"""
Hardware inspection, memory estimation, and auto-device selection for PruneQuest.
"""

from dataclasses import dataclass
import os
import shutil
from typing import Optional
import torch


@dataclass
class HardwareReport:
    cuda_available: bool
    device_name: str
    total_vram_gb: float
    free_vram_gb: float
    system_ram_gb: float
    available_ram_gb: float
    free_disk_gb: float
    recommended_device: str
    is_cpu_only: bool


class HardwareInspector:
    """
    Inspects physical hardware headroom, available VRAM/RAM, and disk space
    to automatically determine device placement and preflight feasibility.
    """

    @classmethod
    def inspect(cls, target_dir: str = ".") -> HardwareReport:
        cuda_avail = torch.cuda.is_available()
        dev_name = "CPU"
        total_vram = 0.0
        free_vram = 0.0

        if cuda_avail:
            try:
                dev_name = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                total_vram = round(total_b / (1024**3), 2)
                free_vram = round(free_b / (1024**3), 2)
            except Exception:
                pass

        # System RAM inspection
        total_ram = 0.0
        avail_ram = 0.0
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_ram = round(mem.total / (1024**3), 2)
            avail_ram = round(mem.available / (1024**3), 2)
        except Exception:
            # Fallback to sysconf if psutil not present
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                total_ram = round((pages * page_size) / (1024**3), 2)
                avail_ram = total_ram * 0.5  # estimate
            except Exception:
                pass

        # Disk space
        check_path = target_dir if os.path.exists(target_dir) else "."
        total_disk, used_disk, free_disk = shutil.disk_usage(check_path)
        free_disk_gb = round(free_disk / (1024**3), 2)

        # Device recommendation: CUDA if >= 2.0GB VRAM available, else CPU
        rec_device = "cuda:0" if (cuda_avail and free_vram >= 2.0) else "cpu"

        return HardwareReport(
            cuda_available=cuda_avail,
            device_name=dev_name,
            total_vram_gb=total_vram,
            free_vram_gb=free_vram,
            system_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            free_disk_gb=free_disk_gb,
            recommended_device=rec_device,
            is_cpu_only=not cuda_avail,
        )

    @classmethod
    def resolve_device(cls, requested_device: Optional[str] = None) -> str:
        """
        Resolves the execution device.
        If 'auto' or None, automatically selects based on hardware inspection.
        If explicit (e.g. 'cuda:0', 'cpu'), validates and honors the override.
        """
        if not requested_device or requested_device.lower() in ("auto", "none"):
            report = cls.inspect()
            return report.recommended_device

        dev = requested_device.strip().lower()
        if dev.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError(
                    f"CUDA device '{requested_device}' was requested, but CUDA is not available on this system. "
                    "Please select --device cpu or ensure NVIDIA drivers and PyTorch CUDA are installed."
                )
        return requested_device

    @classmethod
    def estimate_memory_needs(cls, num_params: int) -> dict:
        """
        Estimates model weights and working memory requirements in gigabytes.
        """
        # bfloat16: 2 bytes per param
        weights_gb = round((num_params * 2) / (1024**3), 2)
        # Working memory buffer for activations & gradients during calibration/screening (~40% overhead)
        working_gb = round(weights_gb * 1.4, 2)
        # Checkpoint serialization space on disk
        disk_needed_gb = round(weights_gb * 1.2, 2)

        return {
            "weights_gb": weights_gb,
            "working_memory_gb": working_gb,
            "disk_needed_gb": disk_needed_gb,
        }
