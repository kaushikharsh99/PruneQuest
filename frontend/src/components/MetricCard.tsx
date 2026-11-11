import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  badge?: string;
  badgePositive?: boolean;
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  badge,
  badgePositive = true,
  icon,
}) => {
  return (
    <div className="bg-white border border-surface-border rounded-xl p-5 shadow-sm hover:border-sage-300 transition-colors">
      <div className="flex items-center justify-between text-stone-500 mb-2">
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
        {icon && <div className="text-sage-600">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-stone-900 tracking-tight">{value}</span>
        {badge && (
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
              badgePositive
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                : 'bg-stone-100 text-stone-600'
            }`}
          >
            {badge}
          </span>
        )}
      </div>
      {subtext && <p className="text-xs text-stone-500 mt-1.5">{subtext}</p>}
    </div>
  );
};
