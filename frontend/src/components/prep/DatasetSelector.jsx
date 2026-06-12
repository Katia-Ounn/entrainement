/**
 * DatasetSelector.jsx — Dropdown de sélection du dataset actif.
 * Style propre, var(--…) only.
 */
import { ChevronDown, Database } from 'lucide-react';

export default function DatasetSelector({ datasets, currentId, onChange, loading }) {
  if (loading) {
    return (
      <div className="px-3 py-2 rounded-lg border text-xs font-mono"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--border-default)',
          color:       'var(--text-tertiary)',
        }}>
        Chargement…
      </div>
    );
  }

  if (datasets.length === 0) {
    return (
      <div className="px-3 py-2 rounded-lg border text-xs font-mono"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--accent-orange)',
          color:       'var(--accent-orange)',
        }}>
        Aucun dataset disponible
      </div>
    );
  }

  return (
    <div className="relative inline-block">
      <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
        <Database size={13} style={{ color: 'var(--brand-primary)' }} />
      </div>
      <select
        value={currentId || ''}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="pl-8 pr-8 py-2 rounded-lg text-sm font-mono border outline-none appearance-none cursor-pointer"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--border-default)',
          color:       'var(--text-primary)',
          minWidth:    220,
        }}
      >
        {!currentId && <option value="">— sélectionner —</option>}
        {datasets.map(d => (
          <option key={d.id} value={d.id}>
            {d.name} · {d.status}
          </option>
        ))}
      </select>
      <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
        <ChevronDown size={13} style={{ color: 'var(--text-tertiary)' }} />
      </div>
    </div>
  );
}
