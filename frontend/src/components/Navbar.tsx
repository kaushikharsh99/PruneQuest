import React from 'react';
import { Sparkles, Library, Activity, Cpu } from 'lucide-react';
import { SystemInfo } from '../types';

interface NavbarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
  hasActiveJob: boolean;
  activeRunId?: string;
  systemInfo?: SystemInfo;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onSelectTab,
  hasActiveJob,
  activeRunId,
  systemInfo,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-surface-50/90 backdrop-blur-md border-b border-surface-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-8">
          <button
            onClick={() => onSelectTab('home')}
            className="flex items-center gap-2.5 group text-left"
          >
            <div className="w-8 h-8 rounded-lg bg-sage-700 flex items-center justify-center text-white font-bold text-sm shadow-sm group-hover:bg-sage-800 transition-colors">
              PQ
            </div>
            <div>
              <span className="font-semibold text-stone-900 tracking-tight text-base group-hover:text-sage-800 transition-colors">
                PruneQuest Studio
              </span>
              <span className="text-[10px] uppercase font-bold text-sage-600 block tracking-widest -mt-0.5">
                Local Pruning
              </span>
            </div>
          </button>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            <button
              onClick={() => onSelectTab('home')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'home'
                  ? 'bg-surface-200 text-stone-900'
                  : 'text-stone-600 hover:text-stone-900 hover:bg-surface-100'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => onSelectTab('new')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'new'
                  ? 'bg-surface-200 text-stone-900'
                  : 'text-stone-600 hover:text-stone-900 hover:bg-surface-100'
              }`}
            >
              <Sparkles className="w-4 h-4 text-sage-600" />
              New Optimization
            </button>
            <button
              onClick={() => onSelectTab('library')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'library'
                  ? 'bg-surface-200 text-stone-900'
                  : 'text-stone-600 hover:text-stone-900 hover:bg-surface-100'
              }`}
            >
              <Library className="w-4 h-4 text-stone-500" />
              Library
            </button>
          </nav>
        </div>

        {/* Right Status Section */}
        <div className="flex items-center gap-3">
          {/* Active Job Alert */}
          {hasActiveJob && (
            <button
              onClick={() => onSelectTab('running')}
              className="flex items-center gap-2 bg-sage-50 border border-sage-200 px-3 py-1 rounded-full text-xs font-semibold text-sage-900 hover:bg-sage-100 transition-colors"
            >
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
              <Activity className="w-3.5 h-3.5 text-sage-700" />
              <span>Optimizing {activeRunId ? `(${activeRunId.slice(0, 16)}...)` : ''}</span>
            </button>
          )}

          {/* Hardware Indicator */}
          {systemInfo && (
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 bg-surface-100 border border-surface-border rounded-lg text-xs text-stone-600">
              <Cpu className="w-3.5 h-3.5 text-sage-700" />
              <span className="font-medium text-stone-800">
                {systemInfo.cuda_available ? systemInfo.device_name : 'CPU Only'}
              </span>
              {systemInfo.cuda_available && (
                <span className="text-stone-400">
                  ({systemInfo.vram_free_gb.toFixed(1)} / {systemInfo.vram_total_gb.toFixed(1)} GB)
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
