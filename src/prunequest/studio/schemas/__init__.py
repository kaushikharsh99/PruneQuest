"""
Pydantic schemas for PruneQuest Studio API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SystemInfoResponse(BaseModel):
    os_name: str
    python_version: str
    torch_version: str
    cuda_available: bool
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_free_gb: float
    cpu_cores: int


class ModelInspectRequest(BaseModel):
    model_id: str
    revision: Optional[str] = None


class ModelInspectResponse(BaseModel):
    model_id: str
    supported: bool
    architecture: str
    tier: str
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_heads: Optional[int] = None
    estimated_params: Optional[int] = None
    estimated_weights_gb: Optional[float] = None
    recommendations: Optional[str] = None
    error: Optional[str] = None


class BenchmarkInfo(BaseModel):
    name: str
    description: str
    category: str
    recommended: bool = False


class DatasetInspectRequest(BaseModel):
    source: str
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    split: Optional[str] = None


class OptimizationPreflightRequest(BaseModel):
    model_id: str
    benchmark: str = "gsm8k"
    dataset: Optional[str] = None
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    metric: Optional[str] = None
    retain: float = 0.99
    preset: str = "balanced"
    strategy: str = "auto"
    device: str = "auto"
    output_dir: Optional[str] = None
    batch_size: Optional[int] = None
    max_iterations: Optional[int] = None
    calibration_dataset: Optional[str] = None


class OptimizationPreflightResponse(BaseModel):
    valid: bool
    model_id: str
    architecture: str
    strategy: str
    device: str
    weights_gb: float
    estimated_peak_memory_gb: float
    hardware_memory_gb: float
    preset: str
    target_retention: float
    max_iterations: int
    batch_size: int
    disk_space_available_gb: float
    disk_space_required_gb: float
    disk_sufficient: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class StartOptimizationRequest(BaseModel):
    model_id: str
    benchmark: str = "gsm8k"
    dataset: Optional[str] = None
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    metric: Optional[str] = None
    retain: float = 0.99
    preset: str = "balanced"
    strategy: str = "auto"
    device: str = "auto"
    output_dir: Optional[str] = None
    batch_size: Optional[int] = None
    max_iterations: Optional[int] = None
    calibration_dataset: Optional[str] = None


class StartOptimizationResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    status: str  # pending, running, cancelling, completed, failed, interrupted
    model_id: str
    benchmark: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
