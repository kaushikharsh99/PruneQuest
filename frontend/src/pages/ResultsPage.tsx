import React, { useState, useEffect } from 'react';
import {
  CheckCircle2,
  Copy,
  Download,
  ArrowRight,
  RotateCcw,
  AlertCircle,
} from 'lucide-react';
import { RunDetailItem } from '../types';
import { fetchRunDetails, fetchRunReport, resumeRun } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';

interface ResultsPageProps {
  runId: string;
  onNavigate: (tab: string, runId?: string) => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ runId, onNavigate }) => {
  const [details, setDetails] = useState<RunDetailItem | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState<boolean>(false);
  const [copiedPath, setCopiedPath] = useState<boolean>(false);
  const [reportMarkdown, setReportMarkdown] = useState<string | null>(null);
  const [showMarkdownModal, setShowMarkdownModal] = useState<boolean>(false);

  useEffect(() => {
    setLoading(true);
    fetchRunDetails(runId)
      .then((data) => {
        setDetails(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });

    fetchRunReport(runId)
      .then((data) => setReportMarkdown(data.summary_markdown))
      .catch(() => {});
  }, [runId]);

  const bestCheckpointPath = details?.best_checkpoint || `${details?.run_dir}/best`;

  const pythonSnippet = `from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = "${bestCheckpointPath}"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    torch_dtype="auto"
)
`;

  const handleCopySnippet = () => {
    navigator.clipboard.writeText(pythonSnippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2000);
  };

  const handleCopyPath = () => {
    navigator.clipboard.writeText(bestCheckpointPath);
    setCopiedPath(true);
    setTimeout(() => setCopiedPath(false), 2000);
  };

  const handleDownloadReport = () => {
    if (!reportMarkdown) return;
    const blob = new Blob([reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prunequest_summary_${runId}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleResume = async () => {
    try {
      const res = await resumeRun(runId);
      onNavigate('running', res.run_id);
    } catch (e: any) {
      alert(`Could not resume: ${e.message}`);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-24">
        <span className="w-6 h-6 border-2 border-sage-600 border-t-transparent rounded-full animate-spin inline-block" />
        <p className="text-xs text-stone-500 mt-2">Loading optimization results...</p>
      </div>
    );
  }

  if (error || !details) {
    return (
      <div className="bg-rose-50 border border-rose-200 rounded-2xl p-8 text-center max-w-lg mx-auto">
        <AlertCircle className="w-8 h-8 text-rose-500 mx-auto mb-2" />
        <h2 className="text-base font-bold text-rose-900">Run Not Found</h2>
        <p className="text-xs text-rose-700 mt-1">{error || 'Could not find details for this run.'}</p>
        <button
          onClick={() => onNavigate('library')}
          className="mt-4 bg-white border border-rose-200 text-rose-800 text-xs font-semibold px-4 py-2 rounded-xl"
        >
          Return to Library
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Hero Completion Card */}
      <div className="bg-white border border-surface-border rounded-2xl p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <StatusBadge status={details.status} />
              {details.reload_verified && (
                <span className="inline-flex items-center gap-1 bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-0.5 rounded-full text-xs font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  Reload Verified
                </span>
              )}
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-stone-900 tracking-tight">
              {details.model_id}
            </h1>
            <p className="text-xs text-stone-500 mt-1">
              Pruned with <span className="font-semibold capitalize">{details.strategy}</span> on{' '}
              <span className="font-semibold">{details.benchmark}</span> benchmark
            </p>
          </div>

          <div className="flex items-center gap-2">
            {details.resumable && (
              <button
                onClick={handleResume}
                className="inline-flex items-center gap-1.5 bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-200 text-xs font-semibold px-3.5 py-2 rounded-xl transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Resume Search</span>
              </button>
            )}
            {reportMarkdown && (
              <button
                onClick={handleDownloadReport}
                className="inline-flex items-center gap-1.5 bg-surface-100 hover:bg-surface-200 text-stone-700 border border-surface-border text-xs font-semibold px-3.5 py-2 rounded-xl transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Summary Report</span>
              </button>
            )}
            <button
              onClick={() => onNavigate('new')}
              className="inline-flex items-center gap-1.5 bg-sage-700 hover:bg-sage-800 text-white text-xs font-semibold px-4 py-2 rounded-xl shadow-sm transition-colors"
            >
              <span>Optimize Another</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* 4 Primary KPI Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
          <div className="bg-surface-50 border border-surface-border rounded-xl p-4">
            <span className="text-stone-400 text-xs uppercase font-medium block">
              Parameter Reduction
            </span>
            <div className="text-2xl font-extrabold text-stone-900 mt-0.5">
              -{details.compression_pct.toFixed(1)}%
            </div>
            <p className="text-[11px] text-stone-500 mt-1">
              {(details.original_parameters / 1e9).toFixed(2)}B →{' '}
              {(details.final_parameters / 1e9).toFixed(2)}B
            </p>
          </div>

          <div className="bg-surface-50 border border-surface-border rounded-xl p-4">
            <span className="text-stone-400 text-xs uppercase font-medium block">
              Accuracy Retention
            </span>
            <div className="text-2xl font-extrabold text-stone-900 mt-0.5">
              {details.accuracy_retention_pct.toFixed(1)}%
            </div>
            <p className="text-[11px] text-stone-500 mt-1">
              Target: ≥ {details.retention_target_pct.toFixed(1)}%
            </p>
          </div>

          <div className="bg-surface-50 border border-surface-border rounded-xl p-4">
            <span className="text-stone-400 text-xs uppercase font-medium block">
              Benchmark Accuracy
            </span>
            <div className="text-2xl font-extrabold text-stone-900 mt-0.5">
              {(details.final_accuracy * 100).toFixed(1)}%
            </div>
            <p className="text-[11px] text-stone-500 mt-1">
              Baseline: {(details.baseline_accuracy * 100).toFixed(1)}%
            </p>
          </div>

          <div className="bg-surface-50 border border-surface-border rounded-xl p-4">
            <span className="text-stone-400 text-xs uppercase font-medium block">
              Execution Time
            </span>
            <div className="text-2xl font-extrabold text-stone-900 mt-0.5">
              {details.elapsed_time_sec < 60
                ? `${details.elapsed_time_sec.toFixed(0)}s`
                : `${(details.elapsed_time_sec / 60).toFixed(1)}m`}
            </div>
            <p className="text-[11px] text-stone-500 mt-1">Device: {details.device}</p>
          </div>
        </div>
      </div>

      {/* Deployment & Checkpoint Card */}
      <div className="bg-white border border-surface-border rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-stone-900">Verified Hugging Face Checkpoint</h2>
            <p className="text-xs text-stone-500">
              Directly loadable with AutoModelForCausalLM without custom code
            </p>
          </div>
          <button
            onClick={handleCopyPath}
            className="inline-flex items-center gap-1.5 text-xs text-sage-800 bg-sage-100 hover:bg-sage-200 px-3 py-1.5 rounded-lg font-medium transition-colors"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>{copiedPath ? 'Path Copied!' : 'Copy Path'}</span>
          </button>
        </div>

        <div className="bg-surface-100 border border-surface-border rounded-xl p-3 font-mono text-xs text-stone-800 break-all select-all">
          {bestCheckpointPath}
        </div>

        {/* Python Snippet */}
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-stone-700">Python Loading Snippet</span>
            <button
              onClick={handleCopySnippet}
              className="text-xs text-sage-700 hover:text-sage-900 font-semibold inline-flex items-center gap-1"
            >
              <Copy className="w-3 h-3" />
              <span>{copiedSnippet ? 'Copied to Clipboard!' : 'Copy Snippet'}</span>
            </button>
          </div>
          <pre className="bg-stone-900 text-stone-100 p-4 rounded-xl text-xs font-mono overflow-x-auto leading-relaxed">
            {pythonSnippet}
          </pre>
        </div>
      </div>

      {/* Markdown Report Preview (if available) */}
      {reportMarkdown && (
        <div className="bg-white border border-surface-border rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-stone-900">Summary Report</h2>
            <button
              onClick={() => setShowMarkdownModal(!showMarkdownModal)}
              className="text-xs text-sage-700 font-semibold hover:text-sage-900"
            >
              {showMarkdownModal ? 'Hide Report Preview' : 'Show Report Preview'}
            </button>
          </div>
          {showMarkdownModal && (
            <div className="bg-surface-50 border border-surface-border rounded-xl p-5 text-xs text-stone-800 font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
              {reportMarkdown}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
