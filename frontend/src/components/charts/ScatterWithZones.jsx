/**
 * ScatterWithZones.jsx — Nuage RUL prédit vs RUL réel avec zones d'erreur.
 *
 *   - Axes adaptatifs sur [0, max_rul] (current_max_rul du modèle)
 *   - 3 zones de tolérance autour de la diagonale y=x :
 *       Vert   = |y_pred - y_true| ≤ 2 jours  → Excellent
 *       Orange = |y_pred - y_true| ≤ 5 jours  → OK
 *       Rouge  = |y_pred - y_true| > 5 jours  → À revoir
 *   - Ligne diagonale verte = perfection
 *   - Légende avec pourcentages par zone
 *
 *   Tout via vars CSS (theme-aware).
 */
import { useMemo } from 'react';
import {
  ResponsiveContainer, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';

const THRESHOLDS = {
  excellent: 2,   // |erreur| ≤ 2j
  ok:        5,   // |erreur| ≤ 5j
};

export default function ScatterWithZones({
  y_true = [],
  y_pred = [],
  max_rul = 30,
  height = 360,
}) {
  // Catégorisation
  const { greenPts, orangePts, redPts, points } = useMemo(() => {
    const pts = (y_true || []).map((t, i) => {
      const p   = (y_pred || [])[i] ?? 0;
      const err = Math.abs(p - t);
      let category;
      if      (err <= THRESHOLDS.excellent) category = 'green';
      else if (err <= THRESHOLDS.ok)        category = 'orange';
      else                                  category = 'red';
      return { y_true: t, y_pred: p, error: err, category };
    });
    return {
      points:    pts,
      greenPts:  pts.filter(p => p.category === 'green'),
      orangePts: pts.filter(p => p.category === 'orange'),
      redPts:    pts.filter(p => p.category === 'red'),
    };
  }, [y_true, y_pred]);

  const pct = (n) => points.length ? Math.round((n / points.length) * 100) : 0;

  if (!points.length) {
    return (
      <div className="rounded-xl border p-6 text-center text-xs"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--border-default)',
          color:       'var(--text-muted)',
        }}>
        Aucune donnée de prédictions (y_true / y_pred manquants).
      </div>
    );
  }

  return (
    <div className="rounded-xl border p-4"
      style={{
        background:  'var(--bg-elevated)',
        borderColor: 'var(--border-default)',
      }}>
      <p className="text-sm font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
        🎯 Nuage de corrélation — RUL Réel vs RUL Prédit (jours)
      </p>
      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
        Axes adaptatifs sur <code>[0, {max_rul}]</code> · {points.length} points de test ·
        Tolérance : <span style={{ color: 'var(--success)' }}>±2j (vert)</span> ·
        <span style={{ color: 'var(--accent-orange)' }}> ±5j (orange)</span> ·
        <span style={{ color: 'var(--error)' }}> au-delà (rouge)</span>
      </p>

      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 50, left: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>

          <XAxis type="number" dataKey="y_true" name="Réel"
                 domain={[0, max_rul]}
                 tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
                 axisLine={{ stroke: 'var(--border-default)' }}
                 label={{
                   value: 'RUL Réel (jours)',
                   position: 'insideBottom',
                   offset: -15,
                   fill:  'var(--text-tertiary)',
                   fontSize: 11,
                 }}/>

          <YAxis type="number" dataKey="y_pred" name="Prédit"
                 domain={[0, max_rul]}
                 tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
                 axisLine={{ stroke: 'var(--border-default)' }}
                 label={{
                   value: 'RUL Prédit (jours)',
                   angle: -90,
                   position: 'insideLeft',
                   offset: -10,
                   fill:  'var(--text-tertiary)',
                   fontSize: 11,
                 }}/>

          {/* Diagonale verte (perfection y = x) */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: max_rul, y: max_rul }]}
            stroke="var(--success)" strokeWidth={2.5} strokeDasharray="6 3"
            label={{ value: 'y = x', fill: 'var(--success)', fontSize: 10, position: 'insideTopLeft' }}/>

          {/* Bordures zones ±2j (vert) */}
          <ReferenceLine
            segment={[{ x: 0, y: -2 }, { x: max_rul, y: max_rul - 2 }]}
            stroke="var(--success)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}/>
          <ReferenceLine
            segment={[{ x: 0, y: 2 }, { x: max_rul, y: max_rul + 2 }]}
            stroke="var(--success)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}/>

          {/* Bordures zones ±5j (orange) */}
          <ReferenceLine
            segment={[{ x: 0, y: -5 }, { x: max_rul, y: max_rul - 5 }]}
            stroke="var(--accent-orange)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}/>
          <ReferenceLine
            segment={[{ x: 0, y: 5 }, { x: max_rul, y: max_rul + 5 }]}
            stroke="var(--accent-orange)" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.5}/>

          {/* Scatter par catégorie */}
          <Scatter name="Excellent (≤2j)" data={greenPts}  fill="var(--success)"       fillOpacity={0.75}/>
          <Scatter name="OK (≤5j)"        data={orangePts} fill="var(--accent-orange)" fillOpacity={0.75}/>
          <Scatter name="À revoir (>5j)"  data={redPts}    fill="var(--error)"         fillOpacity={0.75}/>

          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{
              background:  'var(--bg-elevated)',
              border:      '1px solid var(--border-strong)',
              borderRadius: 8,
              fontSize: 11,
              color: 'var(--text-primary)',
            }}
            formatter={(v, n) => [`${Number(v).toFixed(1)} j`, n]}/>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Légende avec pourcentages */}
      <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 mt-3 text-xs font-mono">
        <LegendBadge color="var(--success)"        label="Excellent ≤2j" n={greenPts.length}  pct={pct(greenPts.length)} />
        <LegendBadge color="var(--accent-orange)"  label="OK ≤5j"        n={orangePts.length} pct={pct(orangePts.length)} />
        <LegendBadge color="var(--error)"          label="À revoir >5j"  n={redPts.length}    pct={pct(redPts.length)} />
      </div>
    </div>
  );
}


function LegendBadge({ color, label, n, pct }) {
  return (
    <div className="flex items-center gap-1.5">
      <span style={{
        display: 'inline-block',
        width: 10, height: 10,
        borderRadius: '50%',
        background: color,
      }}/>
      <span style={{ color: 'var(--text-secondary)' }}>
        {label}
        <b className="ml-1" style={{ color }}>{n}</b>
        <span className="ml-1" style={{ color: 'var(--text-muted)' }}>· {pct}%</span>
      </span>
    </div>
  );
}
