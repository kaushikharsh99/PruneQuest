import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const norm = status?.toLowerCase() || 'unknown';

  let bg = 'bg-stone-100 text-stone-700 border-stone-200';
  let dot = 'bg-stone-400';
  let label = status;

  if (norm === 'completed') {
    bg = 'bg-emerald-50 text-emerald-800 border-emerald-200';
    dot = 'bg-emerald-500';
    label = 'Completed';
  } else if (norm === 'running') {
    bg = 'bg-sage-100 text-sage-900 border-sage-300 animate-pulse';
    dot = 'bg-sage-600';
    label = 'Running';
  } else if (norm === 'cancelling') {
    bg = 'bg-amber-50 text-amber-800 border-amber-200';
    dot = 'bg-amber-500 animate-ping';
    label = 'Cancelling...';
  } else if (norm === 'interrupted') {
    bg = 'bg-amber-50 text-amber-800 border-amber-200';
    dot = 'bg-amber-500';
    label = 'Interrupted';
  } else if (norm === 'failed') {
    bg = 'bg-rose-50 text-rose-800 border-rose-200';
    dot = 'bg-rose-500';
    label = 'Failed';
  } else if (norm === 'pending') {
    bg = 'bg-stone-100 text-stone-700 border-stone-200';
    dot = 'bg-stone-400';
    label = 'Pending';
  }

  const px = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border ${px} ${bg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
};
