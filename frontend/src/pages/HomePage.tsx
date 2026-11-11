import React from 'react';
import { Sparkles, ArrowRight, HardDrive, CheckCircle2, Zap, ShieldCheck } from 'lucide-react';
import { SystemInfo, RunSummaryItem } from '../types';
import { StatusBadge } from '../components/StatusBadge';

interface HomePageProps {
  systemInfo?: SystemInfo;
  recentRuns: RunSummaryItem[];
  activeJob: any;
  onNavigate: (tab: string, runId?: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({
  systemInfo,
  recentRuns,
  activeJob,
  onNavigate,
}) => {
  const completedRuns = recentRuns.filter((r) => r.status === 'completed');
  const bestComp = completedRuns.reduce(
    (max, r) => (r.compression_pct > max ? r.compression_pct : max),
    0
  );

  return (
    <div className="space-y-10">
      {/* Hero Section */}
      <div className="bg-gradient-to-b from-white to-surface-100 border border-surface-border rounded-2xl p-8 sm:p-10 shadow-sm relative overflow-hidden">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-sage-100 border border-sage-200 text-sage-900 rounded-full text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5 text-sage-700" />
            <span>Zero-YAML Autonomous Pruning</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-stone-900 tracking-tight leading-tight">
            Compress Language Models Effortlessly on Your Own Hardware.
          </h1>
          <p className="mt-3 text-base sm:text-lg text-stone-600 leading-relaxed">
            PruneQuest automates structured depth and width compression. Specify your Hugging Face model
            and minimum accuracy retention — PruneQuest discovers, validates, and exports the smallest
            verified checkpoint within your search budget.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              onClick={() => onNavigate('new')}
              className="inline-flex items-center gap-2 bg-sage-700 hover:bg-sage-800 text-white font-medium px-5 py-2.5 rounded-xl shadow-sm transition-all hover:shadow hover:-translate-y-0.5 active:translate-y-0"
            >
              <Sparkles className="w-4 h-4" />
              <span>Start New Optimization</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('library')}
              className="inline-flex items-center gap-2 bg-white hover:bg-surface-200 border border-surface-border text-stone-700 font-medium px-4 py-2.5 rounded-xl transition-colors"
            >
              <span>Browse Experiment Library</span>
            </button>
          </div>
        </div>
      </div>

      {/* Active Job Callout if running */}
      {activeJob?.active && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-emerald-500 animate-ping" />
            <div>
              <p className="text-sm font-semibold text-emerald-900">
                Active Optimization in Progress: {activeJob.job.model_id}
              </p>
              <p className="text-xs text-emerald-700">
                Benchmark: {activeJob.job.benchmark} • Run ID: {activeJob.job.run_id}
              </p>
            </div>
          </div>
          <button
            onClick={() => onNavigate('running', activeJob.job.run_id)}
            className="inline-flex items-center gap-1.5 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-medium px-4 py-2 rounded-lg transition-colors"
          >
            <span>View Live Dashboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Overview Stats & System Card */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        <div className="bg-white border border-surface-border rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-stone-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Completed Runs</span>
            <CheckCircle2 className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-3xl font-extrabold text-stone-900">{completedRuns.length}</div>
          <p className="text-xs text-stone-500 mt-1">Verified checkpoints stored</p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-stone-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Peak Compression</span>
            <Zap className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-3xl font-extrabold text-stone-900">
            {bestComp > 0 ? `-${bestComp.toFixed(1)}%` : '—'}
          </div>
          <p className="text-xs text-stone-500 mt-1">Maximum parameter reduction</p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-stone-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Target Retention</span>
            <ShieldCheck className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-3xl font-extrabold text-stone-900">≥ 99.0%</div>
          <p className="text-xs text-stone-500 mt-1">Strict quality constraint</p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-stone-500 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Target Device</span>
            <HardDrive className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-lg font-bold text-stone-900 truncate">
            {systemInfo?.cuda_available ? systemInfo.device_name : 'CPU'}
          </div>
          <p className="text-xs text-stone-500 mt-1">
            {systemInfo?.cuda_available
              ? `${systemInfo.vram_free_gb.toFixed(1)} GB VRAM Available`
              : `${systemInfo?.ram_free_gb.toFixed(1)} GB RAM Available`}
          </p>
        </div>
      </div>

      {/* Recent Experiments */}
      <div className="bg-white border border-surface-border rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-lg font-bold text-stone-900">Recent Optimizations</h2>
            <p className="text-xs text-stone-500">History of structured pruning experiments</p>
          </div>
          <button
            onClick={() => onNavigate('library')}
            className="text-xs font-semibold text-sage-700 hover:text-sage-900"
          >
            View All ({recentRuns.length}) →
          </button>
        </div>

        {recentRuns.length === 0 ? (
          <div className="text-center py-12 border border-dashed border-stone-200 rounded-xl bg-surface-50">
            <Sparkles className="w-8 h-8 text-stone-400 mx-auto mb-2" />
            <h3 className="text-sm font-semibold text-stone-800">No experiments yet</h3>
            <p className="text-xs text-stone-500 mt-1">
              Start your first automated pruning optimization to see results here.
            </p>
            <button
              onClick={() => onNavigate('new')}
              className="mt-4 inline-flex items-center gap-1.5 bg-sage-700 hover:bg-sage-800 text-white text-xs font-medium px-3.5 py-1.5 rounded-lg"
            >
              Start Optimization
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-stone-600">
              <thead className="bg-surface-100 text-stone-500 uppercase tracking-wider text-[11px] border-y border-surface-border">
                <tr>
                  <th className="py-2.5 px-3">Run ID</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Benchmark</th>
                  <th className="py-2.5 px-3">Compression</th>
                  <th className="py-2.5 px-3">Retention</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {recentRuns.slice(0, 5).map((run) => (
                  <tr key={run.run_id} className="hover:bg-surface-50 transition-colors">
                    <td className="py-3 px-3 font-mono font-medium text-stone-900">
                      {run.run_id}
                    </td>
                    <td className="py-3 px-3 font-medium text-stone-800">{run.model_id}</td>
                    <td className="py-3 px-3">{run.benchmark}</td>
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
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => onNavigate('results', run.run_id)}
                        className="text-sage-700 hover:text-sage-900 font-semibold text-xs"
                      >
                        Inspect →
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
