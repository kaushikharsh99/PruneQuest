import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { NewOptimizationPage } from './pages/NewOptimizationPage';
import { RunningPage } from './pages/RunningPage';
import { ResultsPage } from './pages/ResultsPage';
import { LibraryPage } from './pages/LibraryPage';
import { SystemInfo, RunSummaryItem } from './types';
import { fetchSystemInfo, fetchRuns, fetchActiveOptimization } from './services/api';
import { ErrorBoundary } from './components/ErrorBoundary';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('home');
  const [activeRunId, setActiveRunId] = useState<string | undefined>(undefined);
  const [selectedResultRunId, setSelectedResultRunId] = useState<string | undefined>(undefined);

  const [systemInfo, setSystemInfo] = useState<SystemInfo | undefined>(undefined);
  const [recentRuns, setRecentRuns] = useState<RunSummaryItem[]>([]);
  const [activeJob, setActiveJob] = useState<{ active: boolean; job: any }>({
    active: false,
    job: null,
  });

  const loadAppState = async () => {
    try {
      const [sys, runs, active] = await Promise.all([
        fetchSystemInfo().catch(() => undefined),
        fetchRuns(undefined, undefined).catch(() => []),
        fetchActiveOptimization().catch(() => ({ active: false, job: null })),
      ]);
      if (sys) setSystemInfo(sys);
      setRecentRuns(runs);
      setActiveJob(active);

      if (active.active && active.job) {
        setActiveRunId(active.job.run_id);
      }
    } catch (e) {
      console.error('Failed to load application state:', e);
    }
  };

  useEffect(() => {
    loadAppState();
    const interval = setInterval(loadAppState, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigate = (tab: string, runId?: string) => {
    if (tab === 'running') {
      if (runId) setActiveRunId(runId);
      setCurrentTab('running');
    } else if (tab === 'results') {
      if (runId) setSelectedResultRunId(runId);
      setCurrentTab('results');
    } else {
      setCurrentTab(tab);
    }
  };

  const handleStartJob = (runId: string) => {
    setActiveRunId(runId);
    setCurrentTab('running');
    loadAppState();
  };

  return (
    <div className="min-h-screen bg-surface-50 text-stone-800 flex flex-col font-sans selection:bg-sage-200">
      <Navbar
        currentTab={currentTab}
        onSelectTab={(tab) => handleNavigate(tab)}
        hasActiveJob={activeJob.active}
        activeRunId={activeRunId}
        systemInfo={systemInfo}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        <ErrorBoundary>
          {currentTab === 'home' && (
            <HomePage
              systemInfo={systemInfo}
              recentRuns={recentRuns}
              activeJob={activeJob}
              onNavigate={handleNavigate}
            />
          )}

          {currentTab === 'new' && (
            <NewOptimizationPage onStartJob={handleStartJob} />
          )}

          {currentTab === 'running' && (
            <RunningPage
              runId={activeRunId || recentRuns[0]?.run_id || 'demo_run'}
              onNavigate={handleNavigate}
            />
          )}

          {currentTab === 'results' && (
            <ResultsPage
              runId={selectedResultRunId || recentRuns[0]?.run_id || 'demo_run'}
              onNavigate={handleNavigate}
            />
          )}

          {currentTab === 'library' && (
            <LibraryPage onNavigate={handleNavigate} />
          )}
        </ErrorBoundary>
      </main>

      <footer className="border-t border-surface-border bg-surface-50/50 py-6 text-center text-xs text-stone-400">
        <p>PruneQuest Studio — Local-First Autonomous LLM Compression</p>
      </footer>
    </div>
  );
};

export default App;
