import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Square,
  ArrowRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Layers,
  Zap,
  ShieldCheck,
  RotateCcw,
} from 'lucide-react';
import { RunDetailItem, OptimizationEventPayload } from '../types';
import { fetchRunDetails, cancelRun } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';

interface RunningPageProps {
  runId: string;
  onNavigate: (tab: string, runId?: string) => void;
}

export const RunningPage: React.FC<RunningPageProps> = ({ runId, onNavigate }) => {
  const [runDetails, setRunDetails] = useState<RunDetailItem | null>(null);
  const [events, setEvents] = useState<OptimizationEventPayload[]>([]);
  const [connected, setConnected] = useState<boolean>(false);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const lastEventIdRef = useRef<number>(0);

  // Poll basic run details
  const loadDetails = async () => {
    try {
      const data = await fetchRunDetails(runId);
      setRunDetails(data);
    } catch (e: any) {
      setError(e.message);
    }
  };

  useEffect(() => {
    loadDetails();
    const interval = setInterval(loadDetails, 3000);
    return () => clearInterval(interval);
  }, [runId]);

  // Connect SSE for real-time streaming
  useEffect(() => {
    let active = true;

    const connectSSE = () => {
      const url = `/api/v1/runs/${encodeURIComponent(runId)}/events?last_event_id=${lastEventIdRef.current}`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onopen = () => {
        if (active) setConnected(true);
      };

      es.onmessage = (msg) => {
        if (!active) return;
        try {
          const evData: OptimizationEventPayload = JSON.parse(msg.data);
          if (msg.lastEventId) {
            lastEventIdRef.current = parseInt(msg.lastEventId);
          }
          setEvents((prev) => [...prev, evData]);
        } catch (e) {
          console.error('SSE parse error:', e);
        }
      };

      es.addEventListener('end', () => {
        if (active) {
          setConnected(false);
          loadDetails();
        }
      });

      es.onerror = () => {
        if (active) setConnected(false);
      };
    };

    connectSSE();

    return () => {
      active = false;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [runId]);

  const handleCancel = async () => {
    if (!window.confirm('Request graceful stop? PruneQuest will safely preserve the best verified checkpoint so far.')) {
      return;
    }
    setCancelling(true);
    try {
      await cancelRun(runId);
      loadDetails();
    } catch (e: any) {
      alert(`Could not cancel: ${e.message}`);
    } finally {
      setCancelling(false);
    }
  };

  const isTerminal =
    runDetails?.status === 'completed' ||
    runDetails?.status === 'interrupted' ||
    runDetails?.status === 'failed';

  // Extract latest metrics from events or details
  const lastEvent = events[events.length - 1];
  const measuredRetention =
    runDetails?.accuracy_retention_pct ??
    (lastEvent?.payload?.accuracy_retention_pct !== undefined
      ? lastEvent.payload.accuracy_retention_pct
      : (lastEvent?.payload?.accuracy_retention !== undefined
          ? (lastEvent.payload.accuracy_retention <= 1.0 ? lastEvent.payload.accuracy_retention * 100 : lastEvent.payload.accuracy_retention)
          : 100.0));
  const currentParams =
    runDetails?.final_parameters ||
    lastEvent?.payload?.current_params ||
    runDetails?.original_parameters ||
    0;
  const compressionPct =
    runDetails?.compression_pct ||
    lastEvent?.payload?.compression_pct ||
    lastEvent?.payload?.best_compression ||
    0.0;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 text-xs p-3 rounded-xl flex items-center justify-between">
          <span>Failed to refresh run details: {error}</span>
          <button onClick={() => setError(null)} className="font-semibold underline">Dismiss</button>
        </div>
      )}

      {/* Top Banner / Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white border border-surface-border rounded-2xl p-6 shadow-sm">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="font-mono text-xs text-stone-400">Run: {runId}</span>
            {runDetails && <StatusBadge status={runDetails.status} size="sm" />}
            <span
              className={`inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full ${
                connected
                  ? 'bg-emerald-50 text-emerald-700'
                  : 'bg-stone-100 text-stone-500'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  connected ? 'bg-emerald-500 animate-ping' : 'bg-stone-400'
                }`}
              />
              {connected ? 'Live Stream Active' : 'Disconnected'}
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-stone-900 tracking-tight">
            {runDetails?.model_id || 'Optimizing Model'}
          </h1>
          <p className="text-xs text-stone-500 mt-0.5">
            Benchmark: <span className="font-semibold text-stone-700">{runDetails?.benchmark}</span> •
            Preset: <span className="capitalize">{runDetails?.preset}</span> • Device:{' '}
            <span className="font-mono">{runDetails?.device}</span>
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          {!isTerminal && (
            <button
              onClick={handleCancel}
              disabled={cancelling || runDetails?.status === 'cancelling'}
              className="inline-flex items-center gap-1.5 bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 text-xs font-semibold px-4 py-2 rounded-xl transition-colors disabled:opacity-50"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>{cancelling ? 'Stopping...' : 'Stop Safely'}</span>
            </button>
          )}

          {isTerminal && (
            <button
              onClick={() => onNavigate('results', runId)}
              className="inline-flex items-center gap-1.5 bg-sage-700 hover:bg-sage-800 text-white text-xs font-semibold px-4 py-2 rounded-xl shadow-sm transition-colors"
            >
              <span>View Full Results</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white border border-surface-border rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-stone-400 text-xs uppercase mb-1">
            <span>Compression</span>
            <Zap className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-2xl font-extrabold text-stone-900">
            {compressionPct > 0 ? `-${compressionPct.toFixed(1)}%` : '0.0%'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">Total parameter reduction</p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-stone-400 text-xs uppercase mb-1">
            <span>Retention</span>
            <ShieldCheck className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-2xl font-extrabold text-stone-900">
            {measuredRetention > 0 ? `${measuredRetention.toFixed(1)}%` : '—'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">
            Target: ≥ {runDetails?.retention_target_pct || 99.0}%
          </p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-stone-400 text-xs uppercase mb-1">
            <span>Active Params</span>
            <Layers className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-2xl font-extrabold text-stone-900">
            {currentParams > 0 ? `${(currentParams / 1e9).toFixed(2)}B` : '—'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">
            Base:{' '}
            {runDetails?.original_parameters
              ? `${(runDetails.original_parameters / 1e9).toFixed(2)}B`
              : '—'}
          </p>
        </div>

        <div className="bg-white border border-surface-border rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-stone-400 text-xs uppercase mb-1">
            <span>Benchmark Score</span>
            <Activity className="w-4 h-4 text-sage-600" />
          </div>
          <div className="text-2xl font-extrabold text-stone-900">
            {runDetails?.final_accuracy
              ? `${(runDetails.final_accuracy * 100).toFixed(1)}%`
              : runDetails?.baseline_accuracy
              ? `${(runDetails.baseline_accuracy * 100).toFixed(1)}%`
              : 'Testing...'}
          </div>
          <p className="text-[11px] text-stone-500 mt-1">
            Baseline:{' '}
            {runDetails?.baseline_accuracy
              ? `${(runDetails.baseline_accuracy * 100).toFixed(1)}%`
              : 'Establishing'}
          </p>
        </div>
      </div>

      {/* Real-time Event Stream / Timeline */}
      <div className="bg-white border border-surface-border rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-stone-900">Optimization Lifecycle Events</h2>
            <p className="text-xs text-stone-500">Live SSE stream directly from the optimization engine</p>
          </div>
          <span className="text-xs text-stone-400">{events.length} events logged</span>
        </div>

        <div className="border border-surface-border rounded-xl bg-surface-50 p-4 max-h-96 overflow-y-auto space-y-2 font-mono text-xs">
          {events.length === 0 ? (
            <div className="text-center py-8 text-stone-400">
              <span className="w-3 h-3 rounded-full bg-sage-500 inline-block animate-ping mr-2" />
              Awaiting events from optimizer...
            </div>
          ) : (
            events.map((ev, idx) => {
              const eventName = (ev.event_type || ev.category || '').toLowerCase();
              const isAccepted = eventName.includes('accepted');
              const isRejected = eventName.includes('rejected');
              const isStepBack = eventName.includes('step_back') || eventName.includes('rollback');
              const isInterrupted = eventName.includes('interrupted') || eventName.includes('failed');

              let displayTime = '';
              if (ev.timestamp) {
                try {
                  displayTime = new Date(ev.timestamp).toLocaleTimeString();
                } catch {
                  displayTime = '';
                }
              }

              return (
                <div
                  key={ev.event_id || idx}
                  className={`p-2.5 rounded-lg border flex items-start gap-3 ${
                    isAccepted
                      ? 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
                      : isRejected
                      ? 'bg-stone-100 border-stone-200 text-stone-700'
                      : isStepBack
                      ? 'bg-amber-50/80 border-amber-200 text-amber-900'
                      : isInterrupted
                      ? 'bg-rose-100 border-rose-300 text-rose-900'
                      : 'bg-white border-surface-border text-stone-800'
                  }`}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {isAccepted && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
                    {isRejected && <XCircle className="w-4 h-4 text-stone-400" />}
                    {isStepBack && <RotateCcw className="w-4 h-4 text-amber-600" />}
                    {isInterrupted && <AlertTriangle className="w-4 h-4 text-rose-600" />}
                    {!isAccepted && !isRejected && !isStepBack && !isInterrupted && (
                      <span className="w-2 h-2 rounded-full bg-sage-600 inline-block m-1" />
                    )}
                  </div>
                  <div className="flex-1 overflow-x-auto">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-[11px] uppercase tracking-wider text-sage-800">
                        [{ev.phase || 'info'}]
                      </span>
                      <span className="font-semibold">{ev.event_type || ev.category || 'event'}</span>
                      {displayTime && (
                        <span className="text-[10px] text-stone-400 ml-auto">
                          {displayTime}
                        </span>
                      )}
                    </div>
                    {ev.payload && Object.keys(ev.payload).length > 0 && (
                      <div className="mt-1 text-[11px] text-stone-600 opacity-90 break-all">
                        {JSON.stringify(ev.payload)}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
