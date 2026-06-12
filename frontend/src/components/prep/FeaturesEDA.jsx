/**
 * FeaturesEDA.jsx — Sous-onglet 3 : EDA sur le Dataset_V1.
 *
 *  - Stats descriptives (table count/mean/std/min/max/quartiles)
 *  - Distribution RUL : histogramme (RUL>0)
 *  - ECDF (% sous le seuil d'alerte)
 *  - Camembert sain/alerte
 *  - Corrélations features ↔ RUL (bar horizontal)
 *
 *  POST /api/datasets/{id}/eda_features
 */
import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, LineChart, Line, PieChart, Pie, Legend, ReferenceLine,
} from 'recharts';
import {
  Play, Loader, AlertCircle, ChevronRight, TrendingUp,
  Activity, Sigma, AlertTriangle, CheckCircle2,
} from 'lucide-react';
import { useApp } from '../../AppContext';

const API = 'http://localhost:8000';

export default function FeaturesEDA({ datasetId, onCompleted }) {
  const { edaFeatResult, setEdaFeatResult, markPrepStep } = useApp();
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/datasets/${datasetId}/eda_features`, {
        method: 'POST',
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setEdaFeatResult(data);
      markPrepStep('features_eda');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!edaFeatResult) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <TrendingUp size={36} style={{ color: 'var(--brand-primary)' }}/>
        <h3 className="text-lg font-bold mt-3" style={{ color: 'var(--text-primary)' }}>
          EDA sur les features créées
        </h3>
        <p className="text-sm mt-1 mb-5 max-w-md text-center" style={{ color: 'var(--text-tertiary)' }}>
          Stats descriptives + distribution du RUL (histogramme, ECDF, sain/alerte)
          + matrice de corrélation features ↔ RUL.
        </p>
        <button onClick={run} disabled={loading}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background:  loading ? 'var(--bg-card-alt)' : 'var(--brand-primary)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--brand-primary)',
            cursor:      loading ? 'wait' : 'pointer',
          }}>
          {loading
            ? <><Loader size={14} className="animate-spin" /> Analyse…</>
            : <><Play size={14} /> Lancer EDA features</>}
        </button>
        {error && (
          <div className="mt-4 rounded-lg px-3 py-2 text-xs max-w-xl border"
            style={{
              background:  'var(--tint-error-bg)',
              color:       'var(--error)',
              borderColor: 'var(--error)',
            }}>
            <AlertCircle size={14} className="inline mr-1.5" />
            <span className="whitespace-pre-wrap font-mono">{error}</span>
          </div>
        )}
      </div>
    );
  }

  const r = edaFeatResult;

  // Préparation données histogramme RUL
  const histData = r.rul_distribution
    ? r.rul_distribution.bins.slice(0, -1).map((b, i) => ({
        bin: Math.round(b),
        count: r.rul_distribution.counts[i],
      }))
    : [];

  // ECDF
  const ecdfData = r.ecdf
    ? r.ecdf.rul.map((rul, i) => ({ rul, pct: r.ecdf.pct[i] }))
    : [];

  // Corrélations avec RUL (triées par valeur absolue desc)
  const corrEntries = r.corr_with_rul
    ? Object.entries(r.corr_with_rul)
        .map(([feat, val]) => ({ feat, val: Number(val) }))
        .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    : [];

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Section title="Overview Dataset_V1" icon={Activity}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Lignes"          value={r.overview?.n_rows} highlight />
          <StatCard label="Composants"      value={r.overview?.n_comp} />
          <StatCard label="Pannes (failure=1)" value={r.overview?.n_pannes} />
          <StatCard label="RUL = 0"         value={r.overview?.rul_zero}
                    sub={r.overview?.rul_zero_pct ? `${Number(r.overview.rul_zero_pct).toFixed(1)}%` : null} />
        </div>
      </Section>

      {/* Stats descriptives */}
      {r.stats && (
        <Section title="Statistiques descriptives" icon={Sigma}>
          <div className="overflow-x-auto rounded-lg border" style={{ borderColor: 'var(--border-subtle)' }}>
            <table className="text-xs font-mono w-full" style={{ color: 'var(--text-secondary)' }}>
              <thead style={{ background: 'var(--bg-card-alt)' }}>
                <tr>
                  <th className="px-2 py-1.5 text-left" style={{ color: 'var(--text-tertiary)' }}>feature</th>
                  {['count','mean','std','min','25%','50%','75%','max'].map((k) => (
                    <th key={k} className="px-2 py-1.5 text-right" style={{ color: 'var(--text-tertiary)' }}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(r.stats).map(([feat, stats], i) => (
                  <tr key={feat} style={{ background: i % 2 ? 'var(--bg-card)' : 'transparent' }}>
                    <td className="px-2 py-1 font-semibold"
                        style={{ color: feat === 'RUL' ? 'var(--accent-orange)' : 'var(--text-primary)' }}>
                      {feat}
                    </td>
                    {['count','mean','std','min','25%','50%','75%','max'].map((k) => (
                      <td key={k} className="px-2 py-1 text-right">
                        {typeof stats[k] === 'number' ? stats[k].toFixed(2) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Distribution RUL — 3 graphes côte à côte */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* 1. Histogramme */}
        <Section title="Histogramme RUL (jours)" icon={TrendingUp}>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                <XAxis dataKey="bin" tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }} />
                <YAxis             tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }} />
                <Tooltip contentStyle={{
                  background: 'var(--bg-elevated)',
                  border:     '1px solid var(--border-strong)',
                  borderRadius: 6,
                  color:      'var(--text-primary)',
                }}/>
                <ReferenceLine x={r.rul_distribution?.median} stroke="var(--accent-orange)"
                  label={{ value: 'médiane', fill: 'var(--accent-orange)', fontSize: 10 }} />
                <ReferenceLine x={r.alert_balance?.threshold} stroke="var(--error)"
                  label={{ value: `seuil ${r.alert_balance?.threshold}j`, fill: 'var(--error)', fontSize: 10 }} />
                <Bar dataKey="count" fill="var(--accent-blue)" radius={[2,2,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>

        {/* 2. ECDF */}
        <Section title="ECDF — % sous le seuil" icon={Activity}>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ecdfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                <XAxis dataKey="rul" tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
                       label={{ value: 'RUL (j)', fill: 'var(--text-tertiary)', fontSize: 10, position: 'insideBottom', offset: -5 }} />
                <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }} domain={[0,100]}
                       label={{ value: '% cumul', fill: 'var(--text-tertiary)', fontSize: 10, angle: -90, position: 'insideLeft' }} />
                <Tooltip contentStyle={{
                  background: 'var(--bg-elevated)',
                  border:     '1px solid var(--border-strong)',
                  borderRadius: 6,
                  color:      'var(--text-primary)',
                }}/>
                <Line type="monotone" dataKey="pct" stroke="var(--accent-purple)" strokeWidth={2} dot={false}/>
                <ReferenceLine x={r.alert_balance?.threshold} stroke="var(--error)" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Section>

        {/* 3. Camembert sain/alerte */}
        <Section title={`Sain vs Alerte (≤${r.alert_balance?.threshold ?? 10}j)`} icon={AlertTriangle}>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Sain',   value: Number(r.alert_balance?.healthy_pct ?? 0) },
                    { name: 'Alerte', value: Number(r.alert_balance?.alert_pct ?? 0) },
                  ]}
                  dataKey="value" cx="50%" cy="50%"
                  outerRadius={75} innerRadius={40} label
                >
                  <Cell fill="var(--success)" />
                  <Cell fill="var(--error)" />
                </Pie>
                <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-tertiary)' }}/>
                <Tooltip contentStyle={{
                  background: 'var(--bg-elevated)',
                  border:     '1px solid var(--border-strong)',
                  borderRadius: 6,
                  color:      'var(--text-primary)',
                }}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Section>
      </div>

      {/* 🆕 Tableau distribution RUL par catégorie (notebook cell 42 fin) */}
      {r.rul_categories?.length > 0 && (
        <Section title="Distribution RUL par catégorie" icon={Sigma}>
          <p className="text-xs mb-2" style={{ color: 'var(--text-tertiary)' }}>
            Découpage du RUL en 4 buckets pour comprendre la composition du dataset.
          </p>
          <div className="overflow-x-auto rounded-lg border"
               style={{ borderColor: 'var(--border-subtle)' }}>
            <table className="text-xs w-full" style={{ color: 'var(--text-secondary)' }}>
              <thead style={{ background: 'var(--bg-card-alt)' }}>
                <tr>
                  <th className="px-3 py-2 text-left"  style={{ color: 'var(--text-tertiary)' }}>Catégorie</th>
                  <th className="px-3 py-2 text-right" style={{ color: 'var(--text-tertiary)' }}>Nb lignes</th>
                  <th className="px-3 py-2 text-right" style={{ color: 'var(--text-tertiary)' }}>% du total</th>
                  <th className="px-3 py-2 text-left"  style={{ color: 'var(--text-tertiary)' }}>Distribution</th>
                </tr>
              </thead>
              <tbody>
                {r.rul_categories.map((cat, i) => {
                  const colorMap = {
                    'error':   'var(--error)',
                    'warning': 'var(--accent-orange)',
                    'info':    'var(--accent-blue)',
                    'success': 'var(--success)',
                  };
                  const color = colorMap[cat.color] || 'var(--text-secondary)';
                  return (
                    <tr key={i} style={{ background: i % 2 ? 'var(--bg-card-alt)' : 'transparent' }}>
                      <td className="px-3 py-2 font-mono font-semibold" style={{ color }}>
                        {cat.label}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">{cat.n.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right font-mono font-bold" style={{ color }}>{cat.pct}%</td>
                      <td className="px-3 py-2">
                        <div className="h-2 rounded" style={{
                          background: color,
                          width: `${Math.max(cat.pct, 1)}%`,
                          minWidth: 4,
                        }}/>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* 🆕 Heatmap matrice de corrélation complète (notebook cell 44) */}
      {r.corr_matrix && r.corr_features_order && (
        <Section title="Matrice de corrélation complète (15 features + RUL)" icon={TrendingUp}>
          <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
            Triangle inférieur masqué. Vert = positive · Rouge = négative · plus c'est sombre, plus c'est fort.
          </p>
          <CorrHeatmap matrix={r.corr_matrix} order={r.corr_features_order}/>
        </Section>
      )}

      {/* Corrélations */}
      {corrEntries.length > 0 && (
        <Section title="Corrélations features ↔ RUL (triées)" icon={TrendingUp}>
          <div style={{ height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={corrEntries} margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                <XAxis type="number" domain={[-1, 1]} tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}/>
                <YAxis type="category" dataKey="feat" width={120}
                       tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}/>
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-elevated)',
                    border:     '1px solid var(--border-strong)',
                    borderRadius: 6,
                    color:      'var(--text-primary)',
                  }}
                  formatter={(v) => Number(v).toFixed(3)}/>
                <ReferenceLine x={0} stroke="var(--text-muted)" />
                <Bar dataKey="val" radius={[0,3,3,0]}>
                  {corrEntries.map((e, i) => (
                    <Cell key={i} fill={e.val >= 0 ? 'var(--accent-green)' : 'var(--error)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
            Une corrélation positive (vert) = la feature augmente avec le RUL (composant sain).
            Négative (rouge) = la feature signale un état dégradé (RUL faible).
          </p>
        </Section>
      )}

      {/* CTA continuer */}
      <div className="flex justify-end pt-3 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
        <button onClick={onCompleted}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background: 'var(--success)',
            color:      'var(--bg-elevated)',
            border:     '1px solid var(--success)',
          }}>
          <CheckCircle2 size={14} /> Aller au Prétraitement <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}


// Section = accordion auto-managé (Lot C — design Azure-style)
function Section({ title, icon: Icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border overflow-hidden" style={{
      background:  'var(--bg-card)',
      borderColor: open ? 'var(--brand-primary)' : 'var(--border-default)',
    }}>
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 transition-all"
        style={{ background: open ? 'var(--bg-elevated)' : 'transparent' }}>
        <div className="flex items-center gap-2 text-left">
          {Icon && <Icon size={14} style={{ color: 'var(--brand-primary)' }} />}
          <p className="text-xs font-semibold uppercase tracking-widest"
             style={{ color: 'var(--text-tertiary)' }}>{title}</p>
        </div>
        <span style={{ color: open ? 'var(--brand-primary)' : 'var(--text-muted)' }}>
          {open ? '▴' : '▾'}
        </span>
      </button>
      {open && (
        <div className="px-4 py-4 border-t"
             style={{ borderColor: 'var(--border-subtle)' }}>
          {children}
        </div>
      )}
    </div>
  );
}

// 🆕 Heatmap corrélation — triangle inférieur, gradient rouge↔vert
function CorrHeatmap({ matrix, order }) {
  const cellSize = 36;
  const corrColor = (v) => {
    // v ∈ [-1, 1] : -1 = rouge sombre, 0 = neutre, +1 = vert sombre
    if (v === null || v === undefined || isNaN(v)) return 'transparent';
    const intensity = Math.min(1, Math.abs(v));
    if (v >= 0) {
      // vert
      return `color-mix(in srgb, var(--success) ${Math.round(intensity * 75)}%, var(--bg-card))`;
    }
    // rouge
    return `color-mix(in srgb, var(--error) ${Math.round(intensity * 75)}%, var(--bg-card))`;
  };
  const textColor = (v) =>
    Math.abs(v) > 0.5 ? '#fff' : 'var(--text-primary)';

  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] font-mono"
             style={{ color: 'var(--text-secondary)', borderCollapse: 'separate' }}>
        <thead>
          <tr>
            <th style={{ minWidth: 120 }}></th>
            {order.map(col => (
              <th key={col} className="px-1 py-1 text-center font-normal whitespace-nowrap"
                  style={{ color: 'var(--text-tertiary)',
                    minWidth: cellSize,
                    transform: 'rotate(-45deg)',
                    height: 90 }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {order.map((row, i) => (
            <tr key={row}>
              <td className="px-2 py-1 text-right whitespace-nowrap font-semibold"
                  style={{ color: row === 'RUL' ? 'var(--accent-orange)' : 'var(--text-secondary)' }}>
                {row}
              </td>
              {order.map((col, j) => {
                // Triangle inférieur uniquement (j <= i)
                if (j > i) return <td key={col} style={{ width: cellSize, height: cellSize }}/>;
                const v = matrix[row]?.[col];
                return (
                  <td key={col} className="text-center"
                      style={{
                        width:   cellSize,
                        height:  cellSize,
                        background: corrColor(v),
                        color:      textColor(v),
                        border:    '1px solid var(--bg-card-alt)',
                      }}>
                    {v != null ? v.toFixed(2) : ''}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


function StatCard({ label, value, sub, highlight }) {
  return (
    <div className="rounded-lg p-3 border"
      style={{
        background:  'var(--bg-card)',
        borderColor: highlight ? 'var(--brand-primary)' : 'var(--border-default)',
      }}>
      <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
      <p className="text-xl font-bold font-mono"
         style={{ color: highlight ? 'var(--brand-primary)' : 'var(--text-primary)' }}>
        {value == null ? '—' : Number(value).toLocaleString()}
      </p>
      {sub && <p className="text-[10px] mt-0.5 font-mono" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
    </div>
  );
}
