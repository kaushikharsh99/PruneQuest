import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle2,
  Sliders,
  ArrowRight,
} from 'lucide-react';
import {
  BenchmarkInfo,
  ModelInspectResult,
  PreflightResponse,
  StartOptimizationRequest,
} from '../types';
import {
  fetchBenchmarks,
  fetchPopularModels,
  inspectModel,
  runPreflight,
  startOptimization,
} from '../services/api';

interface NewOptimizationPageProps {
  onStartJob: (runId: string) => void;
}

export const NewOptimizationPage: React.FC<NewOptimizationPageProps> = ({ onStartJob }) => {
  // Form State
  const [modelId, setModelId] = useState<string>('Qwen/Qwen2.5-0.5B-Instruct');
  const [benchmarkSource, setBenchmarkSource] = useState<'native' | 'custom'>('native');
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('gsm8k');
  const [customDataset, setCustomDataset] = useState<string>('');
  const [inputColumn, setInputColumn] = useState<string>('');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [retention, setRetention] = useState<number>(0.99);
  const [preset, setPreset] = useState<string>('balanced');

  // Advanced Settings State
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [strategy, setStrategy] = useState<string>('auto');
  const [device, setDevice] = useState<string>('auto');
  const [batchSize, setBatchSize] = useState<number | undefined>(undefined);
  const [maxIterations, setMaxIterations] = useState<number | undefined>(undefined);
  const [outputDir, setOutputDir] = useState<string>('');

  // Catalog & Inspection State
  const [benchmarks, setBenchmarks] = useState<BenchmarkInfo[]>([]);
  const [popularModels, setPopularModels] = useState<any[]>([]);
  const [inspectingModel, setInspectingModel] = useState<boolean>(false);
  const [modelReport, setModelReport] = useState<ModelInspectResult | null>(null);

  // Preflight Planning State
  const [preflightPlan, setPreflightPlan] = useState<PreflightResponse | null>(null);
  const [preflightLoading, setPreflightLoading] = useState<boolean>(false);
  const [starting, setStarting] = useState<boolean>(false);
  const [startError, setStartError] = useState<string | null>(null);

  // Initial Data Fetch
  useEffect(() => {
    fetchBenchmarks().then(setBenchmarks).catch(console.error);
    fetchPopularModels().then(setPopularModels).catch(console.error);
  }, []);

  // Inspect model when modelId changes (debounced)
  useEffect(() => {
    if (!modelId.trim()) {
      setModelReport(null);
      return;
    }
    const timer = setTimeout(async () => {
      setInspectingModel(true);
      try {
        const report = await inspectModel(modelId.trim());
        setModelReport(report);
      } catch (e: any) {
        setModelReport({
          model_id: modelId,
          supported: false,
          architecture: 'unknown',
          tier: 'LEVEL_0',
          error: e.message,
        });
      } finally {
        setInspectingModel(false);
      }
    }, 400);

    return () => clearTimeout(timer);
  }, [modelId]);

  // Run Preflight Plan when primary configuration changes
  useEffect(() => {
    if (!modelId.trim() || (benchmarkSource === 'custom' && !customDataset.trim())) {
      setPreflightPlan(null);
      return;
    }

    const timer = setTimeout(async () => {
      setPreflightLoading(true);
      try {
        const req: StartOptimizationRequest = {
          model_id: modelId.trim(),
          benchmark: benchmarkSource === 'native' ? selectedBenchmark : undefined,
          dataset: benchmarkSource === 'custom' ? customDataset.trim() : undefined,
          input_column: inputColumn.trim() || undefined,
          target_column: targetColumn.trim() || undefined,
          retain: retention,
          preset,
          strategy,
          device,
          output_dir: outputDir.trim() || undefined,
          batch_size: batchSize,
          max_iterations: maxIterations,
        };
        const plan = await runPreflight(req);
        setPreflightPlan(plan);
      } catch (e: any) {
        console.error('Preflight error:', e);
      } finally {
        setPreflightLoading(false);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [
    modelId,
    benchmarkSource,
    selectedBenchmark,
    customDataset,
    inputColumn,
    targetColumn,
    retention,
    preset,
    strategy,
    device,
    batchSize,
    maxIterations,
    outputDir,
  ]);

  const handleStartOptimization = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const req: StartOptimizationRequest = {
        model_id: modelId.trim(),
        benchmark: benchmarkSource === 'native' ? selectedBenchmark : undefined,
        dataset: benchmarkSource === 'custom' ? customDataset.trim() : undefined,
        input_column: inputColumn.trim() || undefined,
        target_column: targetColumn.trim() || undefined,
        retain: retention,
        preset,
        strategy,
        device,
        output_dir: outputDir.trim() || undefined,
        batch_size: batchSize,
        max_iterations: maxIterations,
      };
      const res = await startOptimization(req);
      onStartJob(res.run_id);
    } catch (e: any) {
      setStartError(e.message || 'Failed to start optimization');
      setStarting(false);
    }
  };

  const isFormValid =
    modelId.trim().length > 0 &&
    (benchmarkSource === 'native' || customDataset.trim().length > 0) &&
    (!preflightPlan || preflightPlan.valid);

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-stone-900 tracking-tight">
          New Optimization
        </h1>
        <p className="text-sm text-stone-500 mt-1">
          Configure an autonomous pruning run. PruneQuest automatically handles profiling, candidate
          screening, iterative pruning, and export reload validation.
        </p>
      </div>

      {startError && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-4 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Unable to start optimization:</span> {startError}
          </div>
        </div>
      )}

      <div className="bg-white border border-surface-border rounded-2xl p-6 sm:p-8 shadow-sm space-y-8">
        {/* Step 1: Model Selection */}
        <div className="space-y-3">
          <label className="block text-sm font-bold text-stone-900">
            1. Target Hugging Face Model
          </label>
          <div className="relative">
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct or meta-llama/Llama-3.2-1B-Instruct"
              className="w-full px-4 py-2.5 rounded-xl border border-surface-border bg-surface-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-sage-500 text-stone-900 text-sm font-mono"
            />
            {inspectingModel && (
              <span className="absolute right-3 top-3 text-xs text-stone-400 animate-pulse">
                Inspecting architecture...
              </span>
            )}
          </div>

          {/* Quick Popular Model Chips */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-stone-400">Popular:</span>
            {popularModels.map((m) => (
              <button
                key={m.model_id}
                type="button"
                onClick={() => setModelId(m.model_id)}
                className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
                  modelId === m.model_id
                    ? 'bg-sage-100 border-sage-300 text-sage-900 font-medium'
                    : 'bg-surface-100 border-surface-border text-stone-600 hover:bg-surface-200'
                }`}
              >
                {m.name} ({m.params})
              </button>
            ))}
          </div>

          {/* Architecture Inspection Feedback Card */}
          {modelReport && (
            <div
              className={`p-3.5 rounded-xl border text-xs flex items-center justify-between gap-4 ${
                modelReport.supported
                  ? 'bg-emerald-50/60 border-emerald-200 text-emerald-900'
                  : 'bg-amber-50/60 border-amber-200 text-amber-900'
              }`}
            >
              <div className="flex items-center gap-2.5">
                {modelReport.supported ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
                )}
                <div>
                  <span className="font-semibold capitalize">
                    {modelReport.architecture} architecture
                  </span>{' '}
                  • {modelReport.tier}
                  {modelReport.estimated_params && (
                    <span className="ml-1 text-stone-600">
                      (~{(modelReport.estimated_params / 1e9).toFixed(2)}B params, ~
                      {modelReport.estimated_weights_gb} GB weights)
                    </span>
                  )}
                </div>
              </div>
              <span
                className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                  modelReport.supported
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-amber-100 text-amber-800'
                }`}
              >
                {modelReport.supported ? 'Supported' : 'Experimental / Untested'}
              </span>
            </div>
          )}
        </div>

        {/* Step 2: Benchmark or Custom Dataset */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="block text-sm font-bold text-stone-900">
              2. Evaluation Benchmark / Dataset
            </label>
            <div className="inline-flex rounded-lg border border-surface-border p-0.5 bg-surface-100 text-xs">
              <button
                type="button"
                onClick={() => setBenchmarkSource('native')}
                className={`px-3 py-1 rounded-md font-medium transition-colors ${
                  benchmarkSource === 'native'
                    ? 'bg-white text-stone-900 shadow-sm'
                    : 'text-stone-500 hover:text-stone-900'
                }`}
              >
                Catalog Benchmark
              </button>
              <button
                type="button"
                onClick={() => setBenchmarkSource('custom')}
                className={`px-3 py-1 rounded-md font-medium transition-colors ${
                  benchmarkSource === 'custom'
                    ? 'bg-white text-stone-900 shadow-sm'
                    : 'text-stone-500 hover:text-stone-900'
                }`}
              >
                Custom Dataset
              </button>
            </div>
          </div>

          {benchmarkSource === 'native' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {benchmarks.map((b) => (
                <div
                  key={b.name}
                  onClick={() => setSelectedBenchmark(b.name)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    selectedBenchmark === b.name
                      ? 'border-sage-500 bg-sage-50/60 ring-2 ring-sage-500/20'
                      : 'border-surface-border bg-surface-50 hover:bg-surface-100'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-stone-900 text-sm">{b.name}</span>
                    {b.recommended && (
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-sage-200 text-sage-800 px-1.5 py-0.5 rounded">
                        Recommended
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-stone-500 mt-1 line-clamp-2">{b.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3 p-4 rounded-xl border border-surface-border bg-surface-50">
              <div>
                <label className="block text-xs font-semibold text-stone-700 mb-1">
                  Local File Path or Hugging Face Dataset ID
                </label>
                <input
                  type="text"
                  value={customDataset}
                  onChange={(e) => setCustomDataset(e.target.value)}
                  placeholder="e.g. ./data/math.jsonl, data/test.parquet, or cais/mmlu"
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white text-sm focus:outline-none focus:ring-2 focus:ring-sage-500"
                />
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block font-medium text-stone-600 mb-1">
                    Input Column (Optional)
                  </label>
                  <input
                    type="text"
                    value={inputColumn}
                    onChange={(e) => setInputColumn(e.target.value)}
                    placeholder="Auto-detected if empty"
                    className="w-full px-3 py-1.5 rounded-lg border border-surface-border bg-white"
                  />
                </div>
                <div>
                  <label className="block font-medium text-stone-600 mb-1">
                    Target Column (Optional)
                  </label>
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    placeholder="Auto-detected if empty"
                    className="w-full px-3 py-1.5 rounded-lg border border-surface-border bg-white"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step 3: Retention Constraint */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="block text-sm font-bold text-stone-900">
              3. Minimum Performance Retention Constraint
            </label>
            <span className="text-sm font-mono font-bold text-sage-800 bg-sage-100 px-2 py-0.5 rounded">
              {(retention * 100).toFixed(1)}%
            </span>
          </div>
          <input
            type="range"
            min="0.80"
            max="1.0"
            step="0.005"
            value={retention}
            onChange={(e) => setRetention(parseFloat(e.target.value))}
            className="w-full accent-sage-700 cursor-pointer"
          />
          <div className="flex justify-between text-[11px] text-stone-400">
            <span>80.0% (Aggressive Pruning)</span>
            <span>95.0% (High Compression)</span>
            <span className="text-sage-800 font-semibold">99.0% (Default Safe)</span>
            <span>100.0% (Zero Degradation)</span>
          </div>
        </div>

        {/* Step 4: Optimization Preset */}
        <div className="space-y-3">
          <label className="block text-sm font-bold text-stone-900">
            4. Compute & Search Budget Preset
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                id: 'quick',
                name: 'Quick Check',
                desc: 'Fast screening and diagnostic pass. Best for sanity testing in under 5 minutes.',
              },
              {
                id: 'balanced',
                name: 'Balanced (Recommended)',
                desc: 'Optimal trade-off between search thoroughness and evaluation time.',
              },
              {
                id: 'thorough',
                name: 'Thorough Search',
                desc: 'Deep multi-step search with large acceptance verification budgets.',
              },
            ].map((p) => (
              <div
                key={p.id}
                onClick={() => setPreset(p.id)}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                  preset === p.id
                    ? 'border-sage-500 bg-sage-50/60 ring-2 ring-sage-500/20'
                    : 'border-surface-border bg-surface-50 hover:bg-surface-100'
                }`}
              >
                <div className="font-semibold text-stone-900 text-sm">{p.name}</div>
                <p className="text-xs text-stone-500 mt-1">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Step 5: Live Preflight Plan Card */}
        {preflightPlan && (
          <div className="bg-surface-100 border border-surface-border rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-sage-700" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-stone-800">
                  Preflight Diagnostic Plan
                </h3>
                {preflightLoading && (
                  <span className="w-3 h-3 border-2 border-sage-600 border-t-transparent rounded-full animate-spin inline-block ml-1" />
                )}
              </div>
              <span
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${
                  preflightPlan.valid
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-rose-100 text-rose-800'
                }`}
              >
                {preflightPlan.valid ? 'Ready to Prune' : 'Configuration Issues'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="bg-white p-2.5 rounded-lg border border-surface-border">
                <span className="text-stone-400 block text-[10px] uppercase">Strategy</span>
                <span className="font-bold text-stone-900 uppercase">
                  {preflightPlan.strategy}
                </span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-surface-border">
                <span className="text-stone-400 block text-[10px] uppercase">Peak VRAM</span>
                <span className="font-bold text-stone-900">
                  {preflightPlan.estimated_peak_memory_gb.toFixed(1)} GB /{' '}
                  {preflightPlan.hardware_memory_gb.toFixed(1)} GB
                </span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-surface-border">
                <span className="text-stone-400 block text-[10px] uppercase">Iterations</span>
                <span className="font-bold text-stone-900">{preflightPlan.max_iterations} max</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-surface-border">
                <span className="text-stone-400 block text-[10px] uppercase">Disk Check</span>
                <span className="font-bold text-stone-900">
                  {preflightPlan.disk_sufficient ? 'Sufficient' : 'Insufficient Space'}
                </span>
              </div>
            </div>

            {preflightPlan.warnings.length > 0 && (
              <div className="text-xs text-amber-800 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                <span className="font-semibold">Notice:</span> {preflightPlan.warnings.join(' • ')}
              </div>
            )}
            {preflightPlan.errors.length > 0 && (
              <div className="text-xs text-rose-800 bg-rose-50 p-2.5 rounded-lg border border-rose-200">
                <span className="font-semibold">Error:</span> {preflightPlan.errors.join(' • ')}
              </div>
            )}
          </div>
        )}

        {/* Collapsible Advanced Settings */}
        <div className="border-t border-surface-border pt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center justify-between w-full text-xs font-semibold text-stone-600 hover:text-stone-900 py-1"
          >
            <span>Advanced Search Configuration (Optional)</span>
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showAdvanced && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs pt-2">
              <div>
                <label className="block font-medium text-stone-700 mb-1">Pruning Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white"
                >
                  <option value="auto">Auto (Hardware & Architecture Guided)</option>
                  <option value="joint">Joint (JIPO Width + Depth)</option>
                  <option value="flap">FLAP (Width / Channel Pruning Only)</option>
                  <option value="block">Block (Depth / Layer Pruning Only)</option>
                </select>
              </div>

              <div>
                <label className="block font-medium text-stone-700 mb-1">Execution Device</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white"
                >
                  <option value="auto">Auto-detect best device</option>
                  <option value="cuda:0">cuda:0 (NVIDIA GPU)</option>
                  <option value="cpu">cpu (Inference fallback)</option>
                </select>
              </div>

              <div>
                <label className="block font-medium text-stone-700 mb-1">
                  Evaluation Batch Size
                </label>
                <input
                  type="number"
                  value={batchSize || ''}
                  onChange={(e) =>
                    setBatchSize(e.target.value ? parseInt(e.target.value) : undefined)
                  }
                  placeholder="Auto (based on VRAM)"
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white"
                />
              </div>

              <div>
                <label className="block font-medium text-stone-700 mb-1">Max Iterations</label>
                <input
                  type="number"
                  value={maxIterations || ''}
                  onChange={(e) =>
                    setMaxIterations(e.target.value ? parseInt(e.target.value) : undefined)
                  }
                  placeholder="Auto (based on preset)"
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white"
                />
              </div>

              <div>
                <label className="block font-medium text-stone-700 mb-1">Output Directory</label>
                <input
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="Auto (e.g. runs/model_timestamp)"
                  className="w-full px-3 py-2 rounded-lg border border-surface-border bg-white"
                />
              </div>
            </div>
          )}
        </div>

        {/* Start Button */}
        <div className="pt-2">
          <button
            type="button"
            disabled={!isFormValid || starting}
            onClick={handleStartOptimization}
            className={`w-full py-3.5 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-sm transition-all ${
              isFormValid && !starting
                ? 'bg-sage-700 hover:bg-sage-800 text-white hover:shadow-md cursor-pointer'
                : 'bg-stone-200 text-stone-400 cursor-not-allowed'
            }`}
          >
            {starting ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Initializing Worker...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Start Optimization</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
