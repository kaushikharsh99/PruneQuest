import React, { useState, useEffect } from 'react';
import {
  Search,
  Filter,
  ArrowUpDown,
  X,
} from 'lucide-react';
import { RunSummaryItem } from '../types';
import { fetchRuns, resumeRun, compareRuns } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';

interface LibraryPageProps {
  onNavigate: (tab: string, runId?: string) => void;
}

export const LibraryPage: React.FC<LibraryPageProps> = ({ onNavigate }) => {
  const [runs, setRuns] = useState<RunSummaryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [search, setSearch] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sortKey, setSortKey] = useState<string>('newest');

  // Compare Runs State
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [comparisonResult, setComparisonResult] = useState<any | null>(null);
  const [comparing, setComparing] = useState<boolean>(false);

  const loadRuns = () => {
    setLoading(true);
    fetchRuns(statusFilter === 'all' ? undefined : statusFilter, undefined)
      .then((data) => {
        setRuns(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    loadRuns();
  }, [statusFilter]);

  const handleResume = async (runId: string) => {
    try {
      const res = await resumeRun(runId);
      onNavigate('running', res.run_id);
    } catch (e: any) {
      alert(`Could not resume run: ${e.message}`);
    }
  };

  const handleToggleCompare = (runId: string) => {
    if (selectedForCompare.includes(runId)) {
      setSelectedForCompare(selectedForCompare.filter((id) => id !== runId));
    } else {
      if (selectedForCompare.length >= 2) {
        setSelectedForCompare([selectedForCompare[1], runId]);
      } else {
        setSelectedForCompare([...selectedForCompare, runId]);
      }
    }
  };

  const handleRunComparison = async () => {
    if (selectedForCompare.length !== 2) return;
    setComparing(true);
    try {
      const res = await compareRuns(selectedForCompare[0], selectedForCompare[1]);
      setComparisonResult(res);
    } catch (e: any) {
      alert(`Comparison failed: ${e.message}`);
    } finally {
      setComparing(false);
    }
  };

  // Filter & Sort
  const filteredRuns = runs
    .filter(
      (r) =>
        r.model_id.toLowerCase().includes(search.toLowerCase()) ||
        r.run_id.toLowerCase().includes(search.toLowerCase()) ||
        r.benchmark.toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) => {
      if (sortKey === 'newest') return b.created_at.localeCompare(a.created_at);
      if (sortKey === 'oldest') return a.created_at.localeCompare(b.created_at);
      if (sortKey === 'compression') return b.compression_pct - a.compression_pct;
      if (sortKey === 'retention') return b.accuracy_retention_pct - a.accuracy_retention_pct;
      return 0;
    });

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-stone-900 tracking-tight">
            Experiment Library
          </h1>
          <p className="text-xs text-stone-500 mt-1">
            Search, compare, inspect, and resume previous optimization runs
          </p>
        </div>

        {/* Compare Action Bar */}
        {selectedForCompare.length > 0 && (
          <div className="flex items-center gap-3 bg-sage-50 border border-sage-200 px-4 py-2 rounded-xl text-xs">
            <span className="font-semibold text-sage-900">
              Selected for Compare ({selectedForCompare.length}/2)
            </span>
            {selectedForCompare.length === 2 && (
              <button
                onClick={handleRunComparison}
                disabled={comparing}
                className="bg-sage-700 hover:bg-sage-800 text-white font-semibold px-3 py-1 rounded-lg transition-colors"
              >
                {comparing ? 'Comparing...' : 'Compare Side-by-Side'}
              </button>
            )}
            <button
              onClick={() => setSelectedForCompare([])}
              className="text-stone-400 hover:text-stone-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Comparison Modal / Panel */}
      {comparisonResult && (
        <div className="bg-white border-2 border-sage-300 rounded-2xl p-6 shadow-md space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-stone-900">
              Side-by-Side Run Comparison: {comparisonResult.run_a.run_id} vs{' '}
              {comparisonResult.run_b.run_id}
            </h2>
            <button
              onClick={() => setComparisonResult(null)}
              className="text-stone-400 hover:text-stone-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4 text-xs">
            {/* Run A */}
            <div className="p-4 bg-surface-50 border border-surface-border rounded-xl space-y-2">
              <span className="font-bold text-stone-900 block font-mono text-sm">
                {comparisonResult.run_a.run_id}
              </span>
              <p className="text-stone-600">Model: {comparisonResult.run_a.model_id}</p>
              <p className="text-stone-600">Benchmark: {comparisonResult.run_a.benchmark}</p>
              <div className="pt-2 border-t border-surface-border space-y-1">
                <p>
                  Compression:{' '}
                  <span className="font-bold">
                    -{comparisonResult.run_a.compression_pct.toFixed(1)}%
                  </span>
                </p>
                <p>
                  Retention:{' '}
                  <span className="font-bold">
                    {comparisonResult.run_a.accuracy_retention_pct.toFixed(1)}%
                  </span>
                </p>
                <p>
                  Accuracy:{' '}
                  <span className="font-bold">
                    {(comparisonResult.run_a.final_accuracy * 100).toFixed(1)}%
                  </span>
                </p>
              </div>
            </div>

            {/* Run B */}
            <div className="p-4 bg-surface-50 border border-surface-border rounded-xl space-y-2">
              <span className="font-bold text-stone-900 block font-mono text-sm">
                {comparisonResult.run_b.run_id}
              </span>
              <p className="text-stone-600">Model: {comparisonResult.run_b.model_id}</p>
              <p className="text-stone-600">Benchmark: {comparisonResult.run_b.benchmark}</p>
              <div className="pt-2 border-t border-surface-border space-y-1">
                <p>
                  Compression:{' '}
                  <span className="font-bold">
                    -{comparisonResult.run_b.compression_pct.toFixed(1)}%
                  </span>
                </p>
                <p>
                  Retention:{' '}
                  <span className="font-bold">
                    {comparisonResult.run_b.accuracy_retention_pct.toFixed(1)}%
                  </span>
                </p>
                <p>
                  Accuracy:{' '}
                  <span className="font-bold">
                    {(comparisonResult.run_b.final_accuracy * 100).toFixed(1)}%
                  </span>
                </p>
              </div>
            </div>
          </div>

          <div className="bg-sage-50 border border-sage-200 p-3 rounded-xl text-xs text-sage-900 space-y-1">
            <span className="font-bold">Delta Summary:</span>
            <p>
              Compression Difference:{' '}
              <span className="font-semibold">{comparisonResult.compression_diff_pct}%</span> •
              Accuracy Retention Difference:{' '}
              <span className="font-semibold">{comparisonResult.retention_diff_pct}%</span>
            </p>
            {comparisonResult.compatibility_notes.length > 0 && (
              <p className="text-amber-800">
                Notice: {comparisonResult.compatibility_notes.join(' • ')}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="bg-white border border-surface-border rounded-xl p-4 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
        {/* Search */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-stone-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by model, run ID, or benchmark..."
            className="w-full pl-9 pr-4 py-2 rounded-lg border border-surface-border bg-surface-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-sage-500"
          />
        </div>

        {/* Status Filter & Sort Controls */}
        <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-stone-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-surface-border bg-surface-50 focus:outline-none"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="running">Running</option>
              <option value="interrupted">Interrupted</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <ArrowUpDown className="w-3.5 h-3.5 text-stone-400" />
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-surface-border bg-surface-50 focus:outline-none"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="compression">Highest Compression</option>
              <option value="retention">Highest Retention</option>
            </select>
          </div>
        </div>
      </div>

      {/* Runs Table */}
      <div className="bg-white border border-surface-border rounded-xl shadow-sm overflow-hidden">
        {loading ? (
          <div className="text-center py-16">
            <span className="w-6 h-6 border-2 border-sage-600 border-t-transparent rounded-full animate-spin inline-block" />
            <p className="text-xs text-stone-500 mt-2">Scanning run directories...</p>
          </div>
        ) : filteredRuns.length === 0 ? (
          <div className="text-center py-16 text-stone-500 text-xs">
            No optimization runs found matching current filter.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-stone-600">
              <thead className="bg-surface-100 text-stone-500 uppercase tracking-wider text-[11px] border-b border-surface-border">
                <tr>
                  <th className="py-3 px-3 w-8">
                    <span className="sr-only">Select</span>
                  </th>
                  <th className="py-3 px-3">Run ID</th>
                  <th className="py-3 px-3">Model</th>
                  <th className="py-3 px-3">Benchmark</th>
                  <th className="py-3 px-3">Strategy</th>
                  <th className="py-3 px-3">Compression</th>
                  <th className="py-3 px-3">Retention</th>
                  <th className="py-3 px-3">Status</th>
                  <th className="py-3 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {filteredRuns.map((run) => (
                  <tr key={run.run_id} className="hover:bg-surface-50 transition-colors">
                    <td className="py-3 px-3">
                      <input
                        type="checkbox"
                        checked={selectedForCompare.includes(run.run_id)}
                        onChange={() => handleToggleCompare(run.run_id)}
                        className="rounded accent-sage-700 cursor-pointer"
                      />
                    </td>
                    <td className="py-3 px-3 font-mono font-medium text-stone-900">
                      {run.run_id}
                    </td>
                    <td className="py-3 px-3 font-medium text-stone-800">{run.model_id}</td>
                    <td className="py-3 px-3">{run.benchmark}</td>
                    <td className="py-3 px-3 uppercase text-[10px] font-semibold text-stone-500">
                      {run.strategy}
                    </td>
                    <td className="py-3 px-3 font-semibold text-stone-900">
                      {run.compression_pct > 0 ? `-${run.compression_pct.toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3 px-3 font-semibold text-stone-900">
                      {run.accuracy_retention_pct > 0
                        ? `${run.accuracy_retention_pct.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-3 px-3">
                      <StatusBadge status={run.status} size="sm" />
                    </td>
                    <td className="py-3 px-3 text-right space-x-2">
                      {run.resumable && (
                        <button
                          onClick={() => handleResume(run.run_id)}
                          className="text-amber-700 hover:text-amber-900 font-semibold"
                          title="Resume optimization from last verified checkpoint"
                        >
                          Resume
                        </button>
                      )}
                      <button
                        onClick={() => onNavigate('results', run.run_id)}
                        className="text-sage-700 hover:text-sage-900 font-semibold"
                      >
                        Details →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
