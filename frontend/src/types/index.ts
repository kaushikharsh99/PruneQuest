export interface SystemInfo {
  os_name: string;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  device_name: string;
  vram_total_gb: number;
  vram_free_gb: number;
  ram_total_gb: number;
  ram_free_gb: number;
  cpu_cores: number;
}

export interface ModelInspectResult {
  model_id: string;
  supported: boolean;
  architecture: string;
  tier: string;
  num_layers?: number;
  hidden_size?: number;
  intermediate_size?: number;
  num_heads?: number;
  estimated_params?: number;
  estimated_weights_gb?: number;
  recommendations?: string;
  error?: string;
}

export interface BenchmarkInfo {
  name: string;
  description: string;
  category: string;
  recommended: boolean;
}

export interface DatasetInspectResult {
  source: string;
  source_type: string;
  detected_task: string;
  schema_info: {
    columns: string[];
    column_types: Record<string, string>;
    num_records: number;
    available_splits: string[];
  };
  field_mapping: {
    input_column: string;
    target_column: string;
    confidence: string;
    reasoning: string;
  };
  candidate_inputs: string[];
  candidate_targets: string[];
  warnings: string[];
}

export interface PreflightResponse {
  valid: boolean;
  model_id: string;
  architecture: string;
  strategy: string;
  device: string;
  weights_gb: number;
  estimated_peak_memory_gb: number;
  hardware_memory_gb: number;
  preset: string;
  target_retention: number;
  max_iterations: number;
  batch_size: number;
  disk_space_available_gb: number;
  disk_space_required_gb: number;
  disk_sufficient: boolean;
  warnings: string[];
  errors: string[];
}

export interface StartOptimizationRequest {
  model_id: string;
  benchmark?: string;
  dataset?: string;
  input_column?: string;
  target_column?: string;
  metric?: string;
  retain?: number;
  preset?: string;
  strategy?: string;
  device?: string;
  output_dir?: string;
  batch_size?: number;
  max_iterations?: number;
}

export interface RunSummaryItem {
  run_id: string;
  model_id: string;
  strategy: string;
  benchmark: string;
  status: string;
  compression_pct: number;
  accuracy_retention_pct: number;
  best_checkpoint?: string;
  created_at: string;
  resumable: boolean;
  run_dir: string;
}

export interface RunDetailItem {
  run_id: string;
  schema_version: number;
  model_id: string;
  strategy: string;
  benchmark: string;
  preset: string;
  device: string;
  status: string;
  resumable: boolean;
  retention_target_pct: number;
  original_parameters: number;
  final_parameters: number;
  parameters_pruned: number;
  compression_pct: number;
  baseline_accuracy: number;
  final_accuracy: number;
  accuracy_retention_pct: number;
  best_checkpoint?: string;
  reload_verified: boolean;
  elapsed_time_sec: number;
  created_at: string;
  run_dir: string;
  reports_dir: string;
  progress?: Record<string, any>;
  error?: string;
}

export interface OptimizationEventPayload {
  event_id?: string;
  run_id: string;
  sequence?: number;
  event_type: string;
  category?: string;
  phase: string;
  timestamp: string;
  payload: Record<string, any>;
}
