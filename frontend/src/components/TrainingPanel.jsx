/**
 * TrainingPanel.jsx — Entraînement Cevital avec visualisation temps réel.
 *
 * Phase 3 (enrichi, PAS refonte) :
 *  ✅ Card 📦 DATASET en haut (sélecteur + récap preprocessing)
 *  ✅ Card ⚙️ PARAMÈTRES MODÈLE (embedding_dim 4/8/16/32)
 *  ✅ Architectures : LSTM/GRU enabled, RNN/Transformer grisés (🔒)
 *  ✅ Fenêtre temporelle en JOURS (presets 7/14/30/60/90)
 *  ✅ Champ Machine ID retiré
 *  ✅ ArchitectureVisualizer : 2 inputs (X_num + X_comp via Embedding)
 *  ✅ MAE en JOURS (pas heures)
 *  ✅ Payload API : dataset_id + embedding_dim
 */
import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import {
  BrainCircuit, Play, Loader, CheckCircle, RefreshCw, Info, Eye,
  Database, Cpu, Lock, AlertTriangle,
} from 'lucide-react';
import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ScatterChart, Scatter,
  BarChart, Bar, Cell, ReferenceLine
} from 'recharts';
import { useApp } from '../AppContext';
import ArchitectureVisualizer from './ArchitectureVisualizer';

const API    = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Architectures supportées (Cevital = LSTM/GRU). RNN/Transformer désactivés
// (pas dans le notebook Cevital ; on les garde pour la pédagogie côté Démo).
const ARCH_DEF = [
  { id:'LSTM',        color:'var(--accent-blue)',   enabled: true  },
  { id:'GRU',         color:'var(--accent-green)',  enabled: true  },
  { id:'RNN',         color:'var(--accent-orange)', enabled: false },
  { id:'Transformer', color:'var(--accent-purple)', enabled: false },
];
const ARCH_COLORS = Object.fromEntries(ARCH_DEF.map(a => [a.id, a.color]));
const EMBEDDING_OPTIONS = [4, 8, 16, 32];

// ── Boîte info ───────────────────────────────────────────────
function InfoBox({ text, color='var(--accent-blue)' }) {
  return (
    <div className="flex gap-2 rounded-lg px-3 py-2 border"
      style={{ background:color+'08', borderColor:color+'30' }}>
      <Info size={13} style={{ color, flexShrink:0, marginTop:1 }} />
      <p className="text-xs leading-relaxed" style={{ color:'var(--text-secondary)' }}>{text}</p>
    </div>
  );
}

// ── Stat card ────────────────────────────────────────────────
function Stat({ label, value, color='var(--accent-blue)', sub }) {
  return (
    <div className="rounded-lg px-3 py-2.5 border" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
      <p className="text-xs" style={{ color:'var(--text-tertiary)' }}>{label}</p>
      <p className="text-lg font-bold font-mono" style={{ color }}>{value ?? '—'}</p>
      {sub && <p className="text-xs mt-0.5" style={{ color:'var(--text-muted)' }}>{sub}</p>}
    </div>
  );
}

// ── Barre de progression style Keras ─────────────────────────
function KerasEpochBar({ epoch }) {
  if (!epoch) return null;
  const pct   = Math.min((epoch.epoch / epoch.total) * 100, 100);
  const filled = Math.round(pct / 5);   // 20 blocs max
  const bar   = '━'.repeat(filled) + '╌'.repeat(20 - filled);

  return (
    <div className="rounded-lg border p-3 font-mono text-xs space-y-2"
      style={{ background:'var(--bg-deep)', borderColor:'var(--border-default)' }}>
      {/* Ligne principale style terminal Keras */}
      <div style={{ color:'var(--text-primary)' }}>
        <span style={{ color:'var(--accent-blue)' }}>Epoch {epoch.epoch}/{epoch.total}</span>
        {' '}
        <span style={{ color:'var(--text-tertiary)' }}>207/207</span>
        {' '}
        <span style={{ color: pct < 100 ? 'var(--accent-orange)' : 'var(--success)' }}>{bar}</span>
        {' '}
        <span style={{ color:'var(--text-muted)' }}>{epoch.elapsed || '—'}s</span>
        {' — '}
        <span style={{ color:'#f06292' }}>loss: </span>
        <span style={{ color:'var(--text-primary)' }}>{epoch.loss?.toFixed(4)}</span>
        {' — '}
        <span style={{ color:'var(--accent-orange)' }}>mae: </span>
        <span style={{ color:'var(--text-primary)' }}>{epoch.mae?.toFixed(4)}</span>
        {' — '}
        <span style={{ color:'var(--accent-purple)' }}>val_loss: </span>
        <span style={{ color:'var(--text-primary)' }}>{epoch.val_loss?.toFixed(4)}</span>
        {' — '}
        <span style={{ color:'var(--accent-green)' }}>val_mae: </span>
        <span style={{ color:'var(--text-primary)' }}>{epoch.val_mae?.toFixed(4)}</span>
      </div>
      {/* Barre de progression visuelle */}
      <div className="w-full rounded-full overflow-hidden" style={{ height:4, background:'var(--bg-elevated)' }}>
        <div className="h-full rounded-full transition-all duration-300"
          style={{ width:`${pct}%`, background: pct<100?'var(--accent-blue)':'var(--success)' }} />
      </div>
    </div>
  );
}

// ── Historique époques scrollable ─────────────────────────────
function EpochHistory({ epochs }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [epochs]);
  if (!epochs.length) return null;
  return (
    <div>
      <p className="text-xs font-semibold mb-1.5" style={{ color:'var(--text-tertiary)' }}>Historique des époques</p>
      <div ref={ref} className="rounded-lg border overflow-auto font-mono text-xs"
        style={{ background:'var(--bg-deep)', borderColor:'var(--border-default)', maxHeight:180 }}>
        <table className="w-full" style={{ borderCollapse:'collapse' }}>
          <thead style={{ position:'sticky', top:0, background:'var(--bg-elevated)' }}>
            <tr>
              {['Époque','Loss','Val Loss','MAE','Val MAE'].map(h => (
                <th key={h} className="px-3 py-1.5 text-left whitespace-nowrap"
                  style={{ color:'var(--text-tertiary)', borderBottom:'1px solid var(--border-default)', fontSize:10 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {epochs.map((e, i) => {
              const isLast = i === epochs.length - 1;
              const isBest = e.val_loss === Math.min(...epochs.map(x => x.val_loss));
              return (
                <tr key={e.epoch} style={{
                  background: isLast ? '#1a3a5c40' : isBest ? 'var(--tint-success-bg)' : i%2===0 ? 'var(--bg-base)' : 'var(--bg-deep)',
                  borderBottom:'1px solid var(--bg-card-alt)',
                }}>
                  <td className="px-3 py-1" style={{ color: isBest?'var(--success)': isLast?'var(--accent-blue)':'var(--text-tertiary)' }}>
                    {e.epoch}{isBest?' 🏆':''}
                  </td>
                  <td className="px-3 py-1" style={{ color:'#f06292' }}>{e.loss?.toFixed(6)}</td>
                  <td className="px-3 py-1" style={{ color:'var(--accent-purple)' }}>{e.val_loss?.toFixed(6)}</td>
                  <td className="px-3 py-1" style={{ color:'var(--accent-orange)' }}>{e.mae?.toFixed(6)}</td>
                  <td className="px-3 py-1" style={{ color:'var(--accent-green)' }}>{e.val_mae?.toFixed(6)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Graphes loss + MAE live ───────────────────────────────────
function LiveCharts({ epochs }) {
  if (!epochs.length) return (
    <div className="flex items-center justify-center h-32 rounded-lg border"
      style={{ borderColor:'var(--border-default)', color:'var(--text-muted)' }}>
      <p className="text-sm">En attente des premières époques...</p>
    </div>
  );
  return (
    <div className="space-y-4">
      {/* Loss */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>
          Loss (MSE) — Train vs Validation
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={epochs} margin={{ top:5, right:10, bottom:5, left:0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
            <XAxis dataKey="epoch" tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} />
            <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} width={60} tickFormatter={v=>v.toFixed(4)} />
            <Tooltip contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8 }}
              formatter={(v,n) => [v?.toFixed(6), n]} labelFormatter={l=>`Époque ${l}`} />
            <Legend wrapperStyle={{ fontSize:11 }} />
            <Line type="monotone" dataKey="loss"     stroke="#f06292" dot={false} strokeWidth={2} name="Train Loss" />
            <Line type="monotone" dataKey="val_loss" stroke="var(--accent-purple)" dot={false} strokeWidth={2} name="Val Loss" strokeDasharray="5 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {/* MAE */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>
          MAE — Train vs Validation
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={epochs} margin={{ top:5, right:10, bottom:5, left:0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
            <XAxis dataKey="epoch" tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} />
            <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} width={60} tickFormatter={v=>v.toFixed(4)} />
            <Tooltip contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8 }}
              formatter={(v,n) => [v?.toFixed(6), n]} labelFormatter={l=>`Époque ${l}`} />
            <Legend wrapperStyle={{ fontSize:11 }} />
            <Line type="monotone" dataKey="mae"     stroke="var(--accent-orange)" dot={false} strokeWidth={2} name="Train MAE" />
            <Line type="monotone" dataKey="val_mae" stroke="var(--accent-green)" dot={false} strokeWidth={2} name="Val MAE" strokeDasharray="5 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── Graphe Trials AutoML ─────────────────────────────────────
function TrialsView({ trials, maxTrials }) {
  if (!trials.length) return null;
  const bestLoss = Math.min(...trials.map(t => t.avg_cv_loss));
  return (
    <div className="space-y-3">
      <InfoBox text="Optimisation Bayésienne : chaque essai teste une combinaison d'hyperparamètres différente. La CV Loss (TimeSeriesSplit 5 plis) mesure la généralisation. L'essai avec la CV Loss minimale sera retenu pour l'entraînement final." color="var(--accent-purple)" />
      <p className="text-xs font-semibold" style={{ color:'var(--text-tertiary)' }}>
        CV Loss par essai — {trials.length}/{maxTrials} terminés
        {trials.length > 0 && <span style={{ color:'var(--success)' }}> · Meilleur : essai #{trials.find(t=>t.avg_cv_loss===bestLoss)?.trial} ({bestLoss.toFixed(5)})</span>}
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={trials} margin={{ top:5, right:10, bottom:5, left:0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis dataKey="trial" tick={{ fill:'var(--text-tertiary)', fontSize:9 }}
            label={{ value:'Essai', position:'insideBottom', offset:-2, fill:'var(--text-muted)', fontSize:9 }} />
          <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} width={60} tickFormatter={v=>v.toFixed(4)} />
          <Tooltip contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8 }}
            formatter={(v) => [v?.toFixed(6), 'CV Loss']} labelFormatter={l=>`Essai #${l}`} />
          <Bar dataKey="avg_cv_loss" radius={[3,3,0,0]} maxBarSize={35}>
            {trials.map((t,i) => (
              <Cell key={i} fill={t.avg_cv_loss===bestLoss ? 'var(--success)' : 'var(--accent-blue)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {/* Tableau classement */}
      <div className="overflow-auto rounded-lg border" style={{ borderColor:'var(--border-default)', maxHeight:200 }}>
        <table className="w-full text-xs font-mono" style={{ borderCollapse:'collapse' }}>
          <thead style={{ position:'sticky', top:0, background:'var(--bg-elevated)' }}>
            <tr>
              {['Rang','Essai #','CV Loss moy.','Durée','Statut'].map(h => (
                <th key={h} className="px-3 py-1.5 text-left" style={{ color:'var(--text-tertiary)', borderBottom:'1px solid var(--border-default)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...trials].sort((a,b)=>a.avg_cv_loss-b.avg_cv_loss).map((t,i) => (
              <tr key={t.trial} style={{ background: i===0?'var(--tint-success-bg)': i%2===0?'var(--bg-card)':'#15172a', borderBottom:'1px solid var(--bg-card-alt)' }}>
                <td className="px-3 py-1.5" style={{ color: i===0?'var(--accent-orange)':'var(--text-muted)' }}>{i===0?'🥇':i===1?'🥈':i===2?'🥉':i+1}</td>
                <td className="px-3 py-1.5 font-bold" style={{ color: i===0?'var(--success)':'var(--text-primary)' }}>#{t.trial}</td>
                <td className="px-3 py-1.5" style={{ color: i===0?'var(--success)':'var(--text-tertiary)' }}>{t.avg_cv_loss?.toFixed(6)}</td>
                <td className="px-3 py-1.5" style={{ color:'var(--text-muted)' }}>{t.duration}s</td>
                <td className="px-3 py-1.5"><span className="px-1.5 py-0.5 rounded text-xs"
                  style={{ background: i===0?'var(--tint-success-bg)':'var(--bg-elevated)', color: i===0?'var(--success)':'var(--text-tertiary)' }}>
                  {i===0?'✓ Meilleur':'Terminé'}
                </span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Résultats finaux & prédictions ───────────────────────────
// Cevital : tout en JOURS (data.mae / data.rmse / data.errors). Backward-compat
// avec l'ancien format Azure (mae_hours/rmse_hours) au cas où.
function PredictionsView({ data, hyperparams }) {
  if (!data) return null;
  const { y_true, y_pred, errors } = data;
  const r2   = data.r2 ?? data.r2_score;
  const mae  = data.mae ?? data.mae_hours;
  const rmse = data.rmse ?? data.rmse_hours;
  const scatter = y_true.map((v,i) => ({ real:v, pred:y_pred[i], err:errors[i] }));

  // Bins erreurs (en JOURS)
  const errBins = [
    { range:'0–2 j',    count: errors.filter(e=>e<=2).length,            color:'var(--success)' },
    { range:'2–5 j',    count: errors.filter(e=>e>2&&e<=5).length,       color:'var(--accent-green)' },
    { range:'5–10 j',   count: errors.filter(e=>e>5&&e<=10).length,      color:'var(--accent-orange)' },
    { range:'10–30 j',  count: errors.filter(e=>e>10&&e<=30).length,     color:'var(--accent-pink)' },
    { range:'>30 j',    count: errors.filter(e=>e>30).length,            color:'var(--error)' },
  ];

  return (
    <div className="space-y-5">
      <InfoBox
        text="Résultats sur le jeu de test (split par composant — données jamais vues pendant l'entraînement). Les valeurs sont dénormalisées et clippées dans [0, current_max_rul] via predict_with_safety()."
        color="var(--success)"
      />

      {/* Métriques globales — en JOURS */}
      <div>
        <p className="text-xs font-semibold mb-2" style={{ color:'var(--text-tertiary)' }}>
          Métriques finales (jours)
        </p>
        <div className="grid grid-cols-4 gap-3">
          <Stat label="R² Score"     value={r2?.toFixed(4) ?? '—'}                        color="var(--success)"        sub="Plus proche de 1 = meilleur" />
          <Stat label="MAE (j)"      value={mae  != null ? `${mae.toFixed(2)} j`  : '—'}  color="var(--accent-blue)"    sub="Mean Abs. Error en jours" />
          <Stat label="RMSE (j)"     value={rmse != null ? `${rmse.toFixed(2)} j` : '—'}  color="var(--accent-orange)"  sub="Sensible aux grands écarts" />
          <Stat label="Échantillons" value={y_true.length}                                color="var(--accent-purple)"  sub="Points de test" />
        </div>
      </div>

      {/* Hyperparamètres retenus */}
      {hyperparams && (
        <div className="rounded-lg border p-3" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className="text-xs font-semibold mb-2" style={{ color:'var(--text-tertiary)' }}>Hyperparamètres du modèle retenu</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(hyperparams).map(([k,v]) => (
              <div key={k} className="text-xs font-mono px-2 py-1 rounded border"
                style={{ background:'var(--bg-elevated)', borderColor:'var(--border-strong)', color:'var(--accent-blue)' }}>
                <span style={{ color:'var(--text-tertiary)' }}>{k}: </span>
                <span>{JSON.stringify(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scatter : prédit vs réel */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>RUL Prédit vs RUL Réel (jours)</p>
        <InfoBox text="Chaque point = un sample du jeu de test. La ligne verte diagonale = prédiction parfaite. Plus les points s'en rapprochent, meilleur est le modèle." color="var(--success)" />
        <div className="mt-2">
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top:10, right:10, bottom:30, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
              <XAxis type="number" dataKey="real" name="Réel" tick={{ fill:'var(--text-tertiary)', fontSize:9 }}
                label={{ value:'RUL Réel (jours)', position:'insideBottom', offset:-15, fill:'var(--text-muted)', fontSize:10 }} />
              <YAxis type="number" dataKey="pred" name="Prédit" tick={{ fill:'var(--text-tertiary)', fontSize:9 }} width={55}
                label={{ value:'RUL Prédit (j)', angle:-90, position:'insideLeft', fill:'var(--text-muted)', fontSize:10 }} />
              <Tooltip cursor={{ strokeDasharray:'3 3' }}
                contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8, fontSize:11 }}
                formatter={(v,n) => [`${v?.toFixed(1)} j`, n]} />
              <ReferenceLine segment={[{x:0,y:0},{x:maxVal,y:maxVal}]} stroke="var(--success)" strokeWidth={1.5} strokeDasharray="4 2" label={{ value:'Parfait', fill:'var(--success)', fontSize:9 }} />
              <Scatter data={scatter.slice(0,300)} fill="var(--accent-blue)" fillOpacity={0.45} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Courbe temporelle */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>
          Évolution temporelle — RUL Réel vs Prédit (200 premiers samples)
        </p>
        <InfoBox text="Vue chronologique : les deux courbes devraient se superposer au maximum. Les écarts indiquent les zones où le modèle a plus de mal à prédire (ex: pic de RUL juste après une panne)." color="var(--accent-blue)" />
        <div className="mt-2">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={y_true.slice(0,200).map((v,i)=>({ i, real:v, pred:y_pred[i] }))}
              margin={{ top:5, right:10, bottom:5, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
              <XAxis dataKey="i" tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} />
              <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={{ stroke:'var(--border-default)' }} width={55} tickFormatter={v=>`${Math.round(v)}j`} />
              <Tooltip contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8 }}
                formatter={(v) => [`${v?.toFixed(1)} j`]} labelFormatter={l=>`Sample #${l}`} />
              <Legend wrapperStyle={{ fontSize:11 }} />
              <Line type="monotone" dataKey="real" stroke="var(--success)" dot={false} strokeWidth={2} name="RUL Réel" />
              <Line type="monotone" dataKey="pred" stroke="var(--accent-blue)" dot={false} strokeWidth={2} name="RUL Prédit" strokeDasharray="5 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Distribution des erreurs */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>Distribution des erreurs absolues par tranche</p>
        <InfoBox text="Idéalement, la majorité des prédictions doit être dans la tranche 0–10h d'erreur. Les erreurs >50h sont critiques pour la maintenance préventive." color="var(--accent-orange)" />
        <div className="mt-2 grid grid-cols-2 gap-4">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={errBins} margin={{ top:5, right:5, bottom:5, left:0 }}>
              <XAxis dataKey="range" tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={false} />
              <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:9 }} axisLine={false} />
              <Tooltip contentStyle={{ background:'var(--bg-elevated)', border:'1px solid var(--border-strong)', borderRadius:8 }}
                formatter={(v) => [v+' samples']} />
              <Bar dataKey="count" radius={[3,3,0,0]}>
                {errBins.map((b,i) => <Cell key={i} fill={b.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="space-y-1.5">
            {errBins.map(b => {
              const pct = Math.round((b.count / errors.length) * 100);
              return (
                <div key={b.range} className="space-y-0.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span style={{ color: b.color }}>{b.range}</span>
                    <span style={{ color:'var(--text-primary)' }}>{b.count} ({pct}%)</span>
                  </div>
                  <div className="w-full rounded-full" style={{ height:6, background:'var(--bg-elevated)' }}>
                    <div className="h-full rounded-full" style={{ width:`${pct}%`, background: b.color }} />
                  </div>
                </div>
              );
            })}
            <p className="text-xs pt-1" style={{ color:'var(--text-muted)' }}>Total : {errors.length} samples</p>
          </div>
        </div>
      </div>

      {/* Tableau détaillé 20 premiers */}
      <div>
        <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>Détail des prédictions — 20 premiers samples du jeu de test</p>
        <div className="overflow-auto rounded-lg border" style={{ borderColor:'var(--border-default)', maxHeight:250 }}>
          <table className="w-full text-xs font-mono" style={{ borderCollapse:'collapse' }}>
            <thead style={{ position:'sticky', top:0, background:'var(--bg-elevated)' }}>
              <tr>
                {['#','RUL Réel (h)','RUL Prédit (j)','Erreur abs. (h)','Erreur (%)','Qualité'].map(h => (
                  <th key={h} className="px-3 py-1.5 text-left whitespace-nowrap"
                    style={{ color:'var(--text-tertiary)', borderBottom:'1px solid var(--border-default)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {y_true.slice(0,50).map((real, i) => {
                const pred = y_pred[i];
                const err  = errors[i];
                const pct  = real > 0 ? Math.round((err/real)*100) : 0;
                const qual = err < 5 ? { label:'Excellent', color:'var(--success)' }
                           : err < 15? { label:'Bon',       color:'var(--accent-green)' }
                           : err < 30? { label:'Moyen',     color:'var(--accent-orange)' }
                           :           { label:'Faible',    color:'var(--error)' };
                return (
                  <tr key={i} style={{ background: i%2===0?'var(--bg-card)':'#15172a', borderBottom:'1px solid var(--bg-card-alt)' }}>
                    <td className="px-3 py-1.5" style={{ color:'var(--text-muted)' }}>{i+1}</td>
                    <td className="px-3 py-1.5 font-bold" style={{ color:'var(--success)' }}>{real.toFixed(1)} j</td>
                    <td className="px-3 py-1.5 font-bold" style={{ color:'var(--accent-blue)' }}>{pred.toFixed(1)} j</td>
                    <td className="px-3 py-1.5" style={{ color: err<10?'var(--accent-green)': err<30?'var(--accent-orange)':'var(--error)' }}>{err.toFixed(1)} j</td>
                    <td className="px-3 py-1.5" style={{ color:'var(--text-tertiary)' }}>{pct}%</td>
                    <td className="px-3 py-1.5">
                      <span className="px-1.5 py-0.5 rounded text-xs" style={{ background:qual.color+'20', color:qual.color }}>
                        {qual.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="text-xs mt-1" style={{ color:'var(--text-muted)' }}>Scroll pour voir plus · 50 lignes affichées</p>
      </div>
    </div>
  );
}

// ─── ArchitectureVisualizer extrait dans son propre fichier (Phase 3) ───
// Voir ./ArchitectureVisualizer.jsx

export default function TrainingPanel() {
  // Dataset Cevital partagé (depuis AppContext, sync avec PreparationPanel)
  const {
    currentDatasetId, selectDataset, preprocResult,
    pendingRetrain, consumeRetrain,
    goToTab, datasetVersion,
  } = useApp();

  const [mode,         setMode]         = useState('manual');
  const [arch,         setArch]         = useState('LSTM');
  const [expName,      setExpName]      = useState('Exp_LSTM_01');
  const [embeddingDim, setEmbeddingDim] = useState(8);
  // Lookback en JOURS (Cevital). Lecture seule depuis preprocResult si dispo.
  const lookbackFromPrep = preprocResult?.lookback ?? 30;
  const [lookback,  setLookback]  = useState(lookbackFromPrep);

  // Synchroniser lookback avec preproc si user a relancé le preprocessing
  useEffect(() => {
    if (preprocResult?.lookback) setLookback(preprocResult.lookback);
  }, [preprocResult?.lookback]);

  // 🔄 Pré-remplissage depuis le Leaderboard ("Re-entraîner")
  //
  // ✅ Lot B : on NE change PAS le dataset actif. L'utilisateur garde le
  // dataset qu'il a sélectionné dans la card 📦 DATASET. Il peut donc
  // appliquer les hyperparams d'un modèle entraîné sur dataset A à
  // n'importe quel dataset preprocessed (dataset B fusionné, par exemple).
  useEffect(() => {
    if (!pendingRetrain) return;
    const { hyperparams: hp, architecture, mode: m, name } = pendingRetrain;
    // ⚠️ dataset_id intentionnellement IGNORÉ — voir commentaire ci-dessus
    if (architecture) setArch(architecture);
    if (m) setMode(m);
    if (name) setExpName(name);
    if (hp) {
      if (hp.embedding_dim != null) setEmbeddingDim(hp.embedding_dim);
      if (hp.num_layers    != null) setNumLayers(hp.num_layers);
      if (Array.isArray(hp.units))         setUnits([...hp.units, 32, 32, 32].slice(0, 4));
      if (Array.isArray(hp.dropout_rates)) setDropouts([...hp.dropout_rates, 0.1, 0.1, 0.1].slice(0, 4));
      if (hp.learning_rate != null) setLr(hp.learning_rate);
      if (hp.epochs        != null) setEpochs(hp.epochs);
      if (hp.batch_size    != null) setBatchSize(hp.batch_size);
      if (hp.patience      != null) setPatience(hp.patience);
      if (hp.lookback      != null) setLookback(hp.lookback);
    }
    consumeRetrain();
    setTimeout(() => addLog(
      `✓ Hyperparamètres chargés depuis "${name || '…'}" — choisis le dataset à entraîner et lance.`
    ), 50);
  }, [pendingRetrain, consumeRetrain]);

  // Liste des datasets dispo (pour la card 📦 DATASET)
  // 🆕 Re-fetch quand `datasetVersion` change (après preprocessing OK p.ex.)
  // → garantit que `currentDataset.status` est à jour.
  const [datasets, setDatasets] = useState([]);
  useEffect(() => {
    fetch(`${API}/api/datasets`)
      .then(r => r.json())
      .then(d => setDatasets(Array.isArray(d) ? d : []))
      .catch(() => setDatasets([]));
  }, [datasetVersion]);
  const currentDataset = useMemo(
    () => datasets.find(d => d.id === currentDatasetId),
    [datasets, currentDatasetId],
  );
  const datasetReady = currentDataset?.status === 'preprocessed';
  const numClassesComp = preprocResult?.num_classes_comp
    ?? currentDataset?.n_composants
    ?? 100;

  // Manuel
  const [numLayers, setNumLayers] = useState(2);
  const [units,     setUnits]     = useState([64, 32, 32, 32]);
  const [dropouts,  setDropouts]  = useState([0.2, 0.1, 0.1, 0.1]);
  const [lr,        setLr]        = useState(0.001);
  const [epochs,    setEpochs]    = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [patience,  setPatience]  = useState(10);

  // AutoML
  // AutoML — bornes manuelles (défauts = notebook PFE_CHAMPION cell 56)
  const [layersMin,  setLayersMin]  = useState(1);
  const [layersMax,  setLayersMax]  = useState(2);
  const [unitsMin,   setUnitsMin]   = useState(64);
  const [unitsMax,   setUnitsMax]   = useState(128);
  const [unitsStep,  setUnitsStep]  = useState(32);
  const [dropMin,    setDropMin]    = useState(0.10);
  const [dropMax,    setDropMax]    = useState(0.25);
  const [lrChoices,  setLrChoices]  = useState([1e-3, 2e-3]);
  const [embSearch,  setEmbSearch]  = useState([4, 8, 16, 32]);
  const [maxTrials,  setMaxTrials]  = useState(10);
  const [trialEp,    setTrialEp]    = useState(12);    // notebook : 12
  const [finalEp,    setFinalEp]    = useState(60);    // notebook : 60
  const [cvFolds,    setCvFolds]    = useState(5);     // gardé pour compat, pas envoyé

  // État
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [epochData,   setEpochData]   = useState([]);
  const [trialData,   setTrialData]   = useState([]);
  const [currentEpoch,setCurrentEpoch]= useState(null);
  const [logs,        setLogs]        = useState([]);
  const [result,      setResult]      = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [hyperparams, setHyperparams] = useState(null);
  const [activeTab,   setActiveTab]   = useState('live');
  const [startTime,   setStartTime]   = useState(null);
  // ⚠️ Logique Azure : on N'a PAS de carte custom AutoML. La progression dans
  // un trial est montrée par `KerasEpochBar` (epoch courant) + onglet Trials
  // (essais terminés). Les events `trial_start` / `trial_end` ne servent qu'à
  // alimenter `trialData` (pour l'onglet Trials) et à logger dans la zone
  // "Logs en direct".

  const wsRef  = useRef(null);
  const logRef = useRef(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  const addLog = (msg) => setLogs(prev => [...prev.slice(-400), { ts: Date.now(), text: msg }]);

  const resetState = () => {
    setTrainingStatus('idle'); setEpochData([]); setTrialData([]);
    setCurrentEpoch(null); setLogs([]); setResult(null);
    setPredictions(null); setHyperparams(null); setActiveTab('live'); setStartTime(null);
  };

  const connectWS = (expId) => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`${WS_URL}/${expId}`);
    wsRef.current = ws;
    ws.onopen = () => addLog('🔌 Connexion WebSocket établie...');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch(data.type) {
          case 'log':
            addLog(data.message);
            break;
          case 'epoch':
            const elapsed = startTime ? Math.round((Date.now()-startTime)/1000) : null;
            const ep = { epoch:data.epoch, total:data.total, loss:data.loss, val_loss:data.val_loss, mae:data.mae, val_mae:data.val_mae, elapsed };
            setCurrentEpoch(ep);
            setEpochData(prev => [...prev, ep]);
            break;
          case 'trial_start':
            addLog(`\n🔍 Essai ${data.trial}/${data.total} en cours...`);
            setActiveTab('trials');
            break;
          case 'trial_end':
            setTrialData(prev => [...prev, { trial:data.trial, avg_cv_loss:data.avg_cv_loss, duration:data.duration }]);
            addLog(`  ✅ Essai ${data.trial} terminé — CV Loss: ${data.avg_cv_loss?.toFixed(5)}`);
            // Si c'était le dernier essai → on signale le re-train final dans les logs
            if (data.trial >= maxTrials) {
              addLog(`\n🏗️ Phase 2 — Re-train final sur le best model (${finalEp} époques)…`);
            }
            break;
          case 'result':
            setResult(data);
            if (data.hyperparameters) setHyperparams(data.hyperparameters);
            addLog(data.message);
            break;
          case 'completed':
            setTrainingStatus('completed');
            if (data.predictions) { setPredictions(data.predictions); setActiveTab('predictions'); }
            if (data.hyperparameters) setHyperparams(data.hyperparameters);
            addLog(`\n🏁 Entraînement terminé ! R²=${data.r2?.toFixed(4)} | MAE=${data.mae?.toFixed(2)}j`);
            break;
          case 'error':
            setTrainingStatus('error');
            addLog(`\n❌ Erreur : ${data.message}`);
            break;
        }
      } catch(_) { addLog(event.data); }
    };
    ws.onerror = () => addLog('❌ Erreur WebSocket');
    ws.onclose = () => addLog('🔌 Connexion fermée.');
  };

  const handleTrain = useCallback(async () => {
    if (!currentDatasetId) {
      addLog('❌ Aucun dataset sélectionné. Va dans "Préparation Données".');
      return;
    }
    if (!datasetReady) {
      addLog('❌ Le dataset n\'est pas encore prétraité. Lance le Prétraitement d\'abord.');
      return;
    }
    resetState();
    setTrainingStatus('running');
    setStartTime(Date.now());
    addLog(mode === 'auto'
      ? `🚀 Lancement AutoML Bayésien (${maxTrials} essais × ${trialEp} époques + re-train ${finalEp})…`
      : `🚀 Lancement entraînement Manuel (${epochs} époques)…`
    );
    const url = mode==='manual' ? `${API}/api/train/manual` : `${API}/api/train/auto`;
    const payload = mode==='manual'
      ? {
          dataset_id:    currentDatasetId,
          name:          expName,
          architecture:  arch,
          embedding_dim: embeddingDim,
          num_layers:    numLayers,
          units:         units.slice(0, numLayers),
          dropout_rates: dropouts.slice(0, numLayers),
          learning_rate: lr,
          epochs,
          batch_size:    batchSize,
          patience,
        }
      : {
          dataset_id:       currentDatasetId,
          name:             expName,
          architecture:     arch,
          max_trials:       maxTrials,
          epochs:           trialEp,         // = époques PAR ESSAI bayésien
          batch_size:       batchSize,
          patience,
          embedding_search: embSearch,
          units_min:        unitsMin,
          units_max:        unitsMax,
          units_step:       unitsStep,
          nb_layers_min:    layersMin,
          nb_layers_max:    layersMax,
          dropout_min:      dropMin,
          dropout_max:      dropMax,
          lr_choices:       lrChoices,
          final_epochs:     finalEp,         // = re-entraînement final du best model
        };
    try {
      const res  = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Erreur serveur');
      connectWS(data.experiment_id);
    } catch(e) {
      addLog(`❌ ${e.message}`);
      setTrainingStatus('error');
    }
  }, [currentDatasetId, datasetReady, mode, arch, expName, embeddingDim,
      numLayers, units, dropouts, lr, epochs, batchSize, patience,
      maxTrials, trialEp]);

  const TABS = [
    { id:'live',        label:'⚡ Live',          show: true },
    { id:'loss',        label:'📉 Loss/MAE',       show: epochData.length > 0 },
    { id:'trials',      label:'🔍 AutoML Trials',  show: trialData.length > 0 },
    { id:'predictions', label:'🎯 Prédictions',    show: !!predictions },
  ];

  const inputSty = { background:'var(--bg-elevated)', borderColor:'var(--border-strong)', color:'var(--text-primary)' };
  const inputCls = "w-full px-3 py-2 rounded-lg text-sm font-mono border outline-none";
  const lblCls   = "text-xs font-semibold uppercase tracking-widest block mb-1";

  const archColor = ARCH_COLORS[arch] || 'var(--accent-blue)';

  return (
    <div className="grid grid-cols-5 gap-6">

      {/* ── Colonne gauche : config ── */}
      <div className="col-span-2 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background:`linear-gradient(135deg,${archColor}20,${archColor}40)`, border:`1px solid ${archColor}40` }}>
            <BrainCircuit size={18} style={{ color:archColor }} />
          </div>
          <div>
            <h2 className="text-lg font-semibold" style={{ color:'var(--text-primary)' }}>Entraînement</h2>
            <p className="text-xs" style={{ color:'var(--text-tertiary)' }}>AutoML Bayésien · TimeSeriesSplit · Live</p>
          </div>
        </div>

        {/* ── 🆕 Card DATASET ── (Phase 3 — tout en haut) */}
        <DatasetCard
          datasets={datasets}
          currentDataset={currentDataset}
          currentDatasetId={currentDatasetId}
          onSelect={selectDataset}
        />

        {/* Mode */}
        <div className="rounded-xl border p-4" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className={lblCls} style={{ color:'var(--text-tertiary)' }}>Mode</p>
          <div className="flex gap-2">
            {[['manual','🔧 Manuel'],['auto','⚡ AutoML']].map(([m,label]) => (
              <button key={m} onClick={()=>setMode(m)}
                className="flex-1 py-2 rounded-lg text-sm font-semibold border transition-all"
                style={{
                  background: mode===m?(m==='manual'?'var(--tint-info-bg)':'var(--tint-success-bg)'):'var(--bg-elevated)',
                  borderColor: mode===m?(m==='manual'?'var(--accent-blue)':'var(--accent-green)'):'var(--border-default)',
                  color: mode===m?(m==='manual'?'var(--accent-blue)':'var(--accent-green)'):'var(--text-muted)',
                }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Architecture — LSTM/GRU enabled, RNN/Transformer grisés */}
        <div className="rounded-xl border p-4" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className={lblCls} style={{ color:'var(--text-tertiary)' }}>Architecture</p>
          <div className="grid grid-cols-2 gap-2">
            {ARCH_DEF.map(({ id:a, color, enabled }) => (
              <button key={a}
                onClick={() => { if (enabled) setArch(a); }}
                disabled={!enabled}
                title={enabled ? `Choisir ${a}` : 'Réservé aux démos pédagogiques (pas dans le pipeline Cevital)'}
                className="py-2 rounded-lg text-sm font-bold border transition-all flex items-center justify-center gap-1.5"
                style={{
                  background:  arch === a && enabled ? `color-mix(in srgb, ${color} 18%, var(--bg-card))` : 'var(--bg-elevated)',
                  borderColor: arch === a && enabled ? color : 'var(--border-default)',
                  color:       !enabled ? 'var(--text-muted)' : (arch === a ? color : 'var(--text-secondary)'),
                  cursor:      enabled ? 'pointer' : 'not-allowed',
                  opacity:     enabled ? 1 : 0.55,
                }}>
                {!enabled && <Lock size={11} />}
                {a}
              </button>
            ))}
          </div>
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
            💡 Cevital = LSTM ou GRU (avec Embedding composant). RNN/Transformer restent visibles
            dans l'onglet <b>Démo</b> à titre pédagogique.
          </p>
        </div>

        {/* 🆕 Card PARAMÈTRES MODÈLE */}
        <ModelParamsCard
          embeddingDim={embeddingDim}
          setEmbeddingDim={setEmbeddingDim}
          mode={mode}
        />

        {/* Nom de l'expérience + Fenêtre temporelle (en JOURS) */}
        <div className="rounded-xl border p-4 space-y-3" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <div>
            <label className={lblCls} style={{ color:'var(--text-tertiary)' }}>Nom de l'expérience</label>
            <input className={inputCls} style={inputSty} value={expName} onChange={e=>setExpName(e.target.value)} />
          </div>

          {/* ⏱ Fenêtre temporelle (lookback) — READ-ONLY, défini en Prétraitement */}
          <div className="rounded-lg border p-3"
            style={{
              background:  'var(--bg-elevated)',
              borderColor: 'var(--accent-blue)',
              borderStyle: 'dashed',
            }}>
            <div className="flex items-center justify-between gap-2">
              <div>
                <p className={lblCls} style={{ color:'var(--text-tertiary)' }}>
                  ⏱ Fenêtre temporelle (lookback)
                </p>
                <p className="text-2xl font-bold font-mono mt-1"
                   style={{ color: 'var(--accent-blue)' }}>
                  {lookback} jours
                </p>
              </div>
              <button onClick={() => goToTab?.('prep')}
                className="px-2 py-1 rounded text-xs font-semibold border whitespace-nowrap"
                style={{
                  background:  'var(--bg-card)',
                  borderColor: 'var(--accent-blue)',
                  color:       'var(--accent-blue)',
                }}>
                ✏️ Modifier dans Prétraitement →
              </button>
            </div>
            <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
              🔒 Cette valeur est <b>figée</b> par le Prétraitement (étape qui construit les
              séquences de shape <code>(N, {lookback}, n_features)</code>). Si tu veux changer
              le lookback, va sur l'onglet <b>Préparation Données → Prétraitement</b>, lance
              un nouveau prétraitement, et reviens entraîner.
            </p>
          </div>
        </div>

        {/* Hyperparamètres */}
        <div className="rounded-xl border p-4 space-y-3" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className={lblCls} style={{ color:'var(--text-tertiary)' }}>
            {mode==='manual'?'Hyperparamètres fixes':'Espace de recherche AutoML'}
          </p>

          {mode==='manual' ? (
            <>
              <div>
                <label className="text-xs mb-1 block" style={{ color:'var(--text-tertiary)' }}>Nombre de couches</label>
                <div className="flex gap-2">
                  {[1,2,3,4].map(n => (
                    <button key={n} onClick={()=>setNumLayers(n)}
                      className="flex-1 py-1.5 rounded text-sm font-mono border transition-all"
                      style={{ background:numLayers===n?'#1a3a5c':'var(--bg-elevated)', borderColor:numLayers===n?'var(--accent-blue)':'var(--border-default)', color:numLayers===n?'var(--accent-blue)':'var(--text-muted)' }}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              {Array.from({length:numLayers},(_,i) => (
                <div key={i} className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs mb-1 block" style={{ color:'var(--text-tertiary)' }}>Couche {i+1} — Unités</label>
                    <input className={inputCls} style={inputSty} type="number"
                      value={units[i]||32} min={8} max={512} step={8}
                      onChange={e=>setUnits(u=>{const a=[...u];a[i]=Number(e.target.value);return a;})} />
                  </div>
                  <div>
                    <label className="text-xs mb-1 block" style={{ color:'var(--text-tertiary)' }}>Couche {i+1} — Dropout</label>
                    <input className={inputCls} style={inputSty} type="number"
                      value={dropouts[i]||0.2} min={0} max={0.9} step={0.05}
                      onChange={e=>setDropouts(d=>{const a=[...d];a[i]=Number(e.target.value);return a;})} />
                  </div>
                </div>
              ))}
              <div className="grid grid-cols-2 gap-2">
                {[
                  {label:'Learning Rate', el:<select className={inputCls} style={inputSty} value={lr} onChange={e=>setLr(Number(e.target.value))}>
                    {[0.01,0.001,0.0001].map(v=><option key={v} value={v}>{v}</option>)}
                  </select>},
                  {label:'Batch Size', el:<select className={inputCls} style={inputSty} value={batchSize} onChange={e=>setBatchSize(Number(e.target.value))}>
                    {[16,32,64,128].map(v=><option key={v} value={v}>{v}</option>)}
                  </select>},
                  {label:`Époques max`, el:<input className={inputCls} style={inputSty} type="number" value={epochs} onChange={e=>setEpochs(Number(e.target.value))} min={5} max={500}/>},
                  {label:'Early stopping', el:<input className={inputCls} style={inputSty} type="number" value={patience} onChange={e=>setPatience(Number(e.target.value))} min={3} max={50}/>},
                ].map(f=>(
                  <div key={f.label}><label className="text-xs mb-1 block" style={{color:'var(--text-tertiary)'}}>{f.label}</label>{f.el}</div>
                ))}
              </div>
            </>
          ) : (
            <>
              {/* AutoML — Section 1 : paramètres généraux */}
              <div className="rounded-lg p-2 mb-2 text-[11px]"
                style={{ background:'var(--tint-success-bg)',
                         borderLeft:'3px solid var(--accent-green)',
                         color:'var(--text-secondary)' }}>
                ⚙️ <b>Paramètres généraux AutoML</b>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  {label:'Essais Bayésiens',       val:maxTrials, set:setMaxTrials, min:2, max:30},
                  {label:'Époques / essai',        val:trialEp,   set:setTrialEp,   min:3, max:100},
                  {label:'Époques finales (best)', val:finalEp,   set:setFinalEp,   min:5, max:500},
                  {label:'Batch size',             val:batchSize, set:setBatchSize, min:8, max:512},
                  {label:'Early stopping',         val:patience,  set:setPatience,  min:3, max:50},
                ].map(f=>(
                  <div key={f.label}>
                    <label className="text-xs mb-1 block" style={{color:'var(--text-tertiary)'}}>{f.label}</label>
                    <input className={inputCls} style={inputSty} type="number"
                      value={f.val} onChange={e=>f.set(Number(e.target.value))} min={f.min} max={f.max}/>
                  </div>
                ))}
              </div>

              {/* AutoML — Section 2 : bornes Bayésien (CHOISISSABLES manuellement) */}
              <div className="rounded-lg p-2 mt-3 mb-2 text-[11px]"
                style={{ background:'var(--tint-info-bg)',
                         borderLeft:'3px solid var(--accent-blue)',
                         color:'var(--text-secondary)' }}>
                🎛️ <b>Bornes de recherche Bayésienne</b> (défauts = notebook PFE_CHAMPION)
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  {label:'nb_layers min', val:layersMin, set:setLayersMin, min:1, max:4},
                  {label:'nb_layers max', val:layersMax, set:setLayersMax, min:1, max:4},
                  {label:'units min',     val:unitsMin,  set:setUnitsMin,  min:8, max:512},
                  {label:'units max',     val:unitsMax,  set:setUnitsMax,  min:8, max:512},
                  {label:'units step',    val:unitsStep, set:setUnitsStep, min:8, max:128},
                  {label:'dropout min',   val:dropMin,   set:setDropMin,   min:0, max:0.9, step:0.05},
                  {label:'dropout max',   val:dropMax,   set:setDropMax,   min:0, max:0.9, step:0.05},
                ].map(f=>(
                  <div key={f.label}>
                    <label className="text-xs mb-1 block" style={{color:'var(--text-tertiary)'}}>{f.label}</label>
                    <input className={inputCls} style={inputSty} type="number"
                      value={f.val} onChange={e=>f.set(Number(e.target.value))}
                      min={f.min} max={f.max} step={f.step || 1}/>
                  </div>
                ))}
              </div>

              {/* Embedding dim choices */}
              <div className="mt-2">
                <label className="text-xs mb-1 block" style={{color:'var(--text-tertiary)'}}>
                  embedding_dim — valeurs candidate
                </label>
                <div className="flex gap-1.5 flex-wrap">
                  {[2, 4, 8, 16, 32, 64].map(v => {
                    const active = embSearch.includes(v);
                    return (
                      <button key={v} type="button"
                        onClick={() => setEmbSearch(prev =>
                          prev.includes(v) ? prev.filter(x => x !== v) : [...prev, v].sort((a,b)=>a-b)
                        )}
                        className="px-2 py-1 rounded text-xs font-mono border"
                        style={{
                          background:  active ? 'var(--accent-purple)' : 'var(--bg-elevated)',
                          borderColor: active ? 'var(--accent-purple)' : 'var(--border-default)',
                          color:       active ? 'var(--bg-elevated)' : 'var(--text-tertiary)',
                        }}>
                        {active && '✓ '}{v}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Learning rates choices */}
              <div className="mt-2">
                <label className="text-xs mb-1 block" style={{color:'var(--text-tertiary)'}}>
                  learning_rate — valeurs candidate
                </label>
                <div className="flex gap-1.5 flex-wrap">
                  {[1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2].map(v => {
                    const active = lrChoices.includes(v);
                    return (
                      <button key={v} type="button"
                        onClick={() => setLrChoices(prev =>
                          prev.includes(v) ? prev.filter(x => x !== v) : [...prev, v].sort((a,b)=>a-b)
                        )}
                        className="px-2 py-1 rounded text-xs font-mono border"
                        style={{
                          background:  active ? 'var(--accent-orange)' : 'var(--bg-elevated)',
                          borderColor: active ? 'var(--accent-orange)' : 'var(--border-default)',
                          color:       active ? 'var(--bg-elevated)' : 'var(--text-tertiary)',
                        }}>
                        {active && '✓ '}{v}
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Mode AutoML : message d'info — total époques de recherche */}
        {mode === 'auto' && trainingStatus === 'idle' && (
          <div className="rounded-xl border p-4"
            style={{ background:'var(--bg-base)', borderColor:'var(--accent-green)' }}>
            <div className="flex items-start gap-2">
              <BrainCircuit size={18} style={{ color:'var(--accent-green)', flexShrink:0, marginTop:2 }}/>
              <div>
                <p className="text-sm font-semibold" style={{ color:'var(--accent-green)' }}>
                  Mode AutoML — Recherche bayésienne
                </p>
                <p className="text-xs mt-1" style={{ color:'var(--text-tertiary)' }}>
                  <b>Phase 1 — Recherche</b> : {maxTrials} essais × max {trialEp} époques chacun
                  = <span style={{color:'var(--accent-green)'}}>jusqu'à {maxTrials * trialEp} époques</span> pour explorer<br/>
                  <b>Phase 2 — Entraînement final</b> : best model ré-entraîné sur
                  <span style={{color:'var(--accent-orange)'}}> {finalEp} époques</span> (notebook PFE = 60)<br/>
                  Early stopping après <b>{patience}</b> époques sans amélioration ·
                  Logs LIVE à chaque essai dans la zone monitoring.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Bouton lancer */}
        <button onClick={handleTrain} disabled={trainingStatus==='running'}
          className="w-full py-3.5 rounded-xl font-bold flex items-center justify-center gap-2 transition-all"
          style={{
            background: trainingStatus==='running'?'var(--bg-elevated)':`linear-gradient(135deg,${archColor}15,${archColor}30)`,
            color: trainingStatus==='running'?'var(--text-muted)':archColor,
            border:`1px solid ${trainingStatus==='running'?'var(--border-default)':archColor+'60'}`,
            cursor: trainingStatus==='running'?'not-allowed':'pointer',
          }}>
          {trainingStatus==='running'
            ? <><Loader size={16} className="animate-spin"/> Entraînement en cours...</>
            : <><Play size={16}/> Lancer {arch}</>
          }
        </button>

        {trainingStatus!=='idle' && (
          <button onClick={resetState} className="w-full py-2 rounded-lg text-xs border flex items-center justify-center gap-1"
            style={{ borderColor:'var(--border-default)', color:'var(--text-tertiary)', background:'var(--bg-card)' }}>
            <RefreshCw size={12}/> Réinitialiser
          </button>
        )}
      </div>

      {/* ── Colonne droite : monitoring ── */}
      <div className="col-span-3 space-y-4">

        {/* 🎨 Architecture en direct — affiché à droite des hyperparamètres, même niveau */}
        {mode === 'manual' && trainingStatus === 'idle' && (
          <ArchitectureVisualizer
            arch={arch}
            numLayers={numLayers}
            units={units}
            dropouts={dropouts}
            batchSize={batchSize}
            lookback={lookback}
            embeddingDim={embeddingDim}
            numClassesComp={numClassesComp}
          />
        )}

        {/* Statut */}
        <div className="rounded-xl border p-4 flex items-center justify-between"
          style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full" style={{
              background: trainingStatus==='running'?'var(--accent-blue)': trainingStatus==='completed'?'var(--success)': trainingStatus==='error'?'var(--error)':'var(--text-muted)',
              boxShadow: trainingStatus==='running'?'0 0 8px var(--accent-blue)':'none',
              animation: trainingStatus==='running'?'pulse-dot 1.5s ease-in-out infinite':undefined,
            }}/>
            <span className="text-sm font-semibold" style={{ color:'var(--text-primary)' }}>Statut de l'entraînement</span>
          </div>
          <span className="text-sm font-mono px-3 py-1 rounded-full" style={{
            background: trainingStatus==='running'?'#0d2a3a': trainingStatus==='completed'?'var(--tint-success-bg)': trainingStatus==='error'?'var(--tint-error-bg)':'var(--bg-elevated)',
            color: trainingStatus==='running'?'var(--accent-blue)': trainingStatus==='completed'?'var(--success)': trainingStatus==='error'?'var(--error)':'var(--text-muted)',
          }}>
            {trainingStatus==='running'?'⟳ En cours': trainingStatus==='completed'?'✓ Terminé': trainingStatus==='error'?'✗ Erreur':'◦ En attente'}
          </span>
        </div>

        {/* Métriques finales — en JOURS pour Cevital */}
        {result && (
          <div className="grid grid-cols-4 gap-3">
            <Stat label="R² Score"  value={result.r2?.toFixed(4) || predictions?.r2_score?.toFixed(4)}  color="var(--success)"        sub="Coefficient de détermination" />
            <Stat label="MAE (j)"   value={result.mae != null ? `${result.mae.toFixed(2)} j` : '—'}       color="var(--accent-blue)"    sub="Mean Abs. Error en jours" />
            <Stat label="RMSE (j)"  value={result.rmse != null ? `${result.rmse.toFixed(2)} j` : '—'}     color="var(--accent-orange)"  sub="Root MSE en jours" />
            <Stat label="Durée"     value={result.duration ? `${Math.round(result.duration)}s` : '—'}    color="var(--accent-purple)"  sub="Temps d'entraînement" />
          </div>
        )}

        {/* Barre Keras style — époque courante (manuel ET dans chaque trial AutoML) */}
        {trainingStatus==='running' && currentEpoch && (
          <KerasEpochBar epoch={currentEpoch} />
        )}

        {/* Tabs */}
        {trainingStatus !== 'idle' && (
          <>
            <div className="flex gap-1 p-1 rounded-xl border" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
              {TABS.filter(t=>t.show).map(t => (
                <button key={t.id} onClick={()=>setActiveTab(t.id)}
                  className="flex-1 py-2 rounded-lg text-xs font-semibold transition-all"
                  style={{
                    background: activeTab===t.id?'var(--bg-elevated)':'transparent',
                    color: activeTab===t.id?'var(--text-primary)':'var(--text-muted)',
                    border: activeTab===t.id?'1px solid var(--border-strong)':'1px solid transparent',
                  }}>
                  {t.label}
                </button>
              ))}
            </div>

            {/* Tab Live */}
            {activeTab==='live' && (
              <div className="space-y-3">
                <div className="rounded-xl border" style={{ background:'var(--bg-deep)', borderColor:'var(--border-default)' }}>
                  <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor:'var(--border-default)' }}>
                    <span className="text-xs font-semibold uppercase tracking-widest" style={{ color:'var(--text-tertiary)' }}>Logs en direct</span>
                    {trainingStatus==='running' && <span className="text-xs font-mono" style={{ color:'var(--accent-blue)' }}>● LIVE</span>}
                  </div>
                  <div ref={logRef} className="p-3 font-mono text-xs space-y-0.5 overflow-y-auto" style={{ maxHeight:200 }}>
                    {logs.length===0
                      ? <p style={{ color:'var(--text-muted)' }}>En attente du lancement...</p>
                      : logs.map((l,i) => (
                        <p key={i} style={{
                          color: l.text.includes('❌')?'var(--error)': l.text.includes('✅')||l.text.includes('🏁')?'var(--success)': l.text.includes('🔍')?'var(--accent-orange)':'var(--text-tertiary)',
                          whiteSpace:'pre-wrap',
                        }}>{l.text}</p>
                      ))
                    }
                  </div>
                </div>
                <EpochHistory epochs={epochData} />
              </div>
            )}

            {/* Tab Loss/MAE */}
            {activeTab==='loss' && (
              <div className="rounded-xl border p-4" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
                <p className="text-xs font-semibold mb-1" style={{ color:'var(--text-tertiary)' }}>
                  Courbes d'apprentissage — {epochData.length} époque{epochData.length>1?'s':''}
                </p>
                <InfoBox text="La courbe de validation (pointillée) doit suivre la courbe d'entraînement. Si val_loss remonte alors que loss continue de baisser → surapprentissage (overfitting). L'early stopping stoppe automatiquement dans ce cas." color="var(--accent-blue)" />
                <div className="mt-3">
                  <LiveCharts epochs={epochData} />
                </div>
              </div>
            )}

            {/* Tab Trials */}
            {activeTab==='trials' && (
              <div className="rounded-xl border p-4" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
                <p className="text-xs font-semibold mb-3" style={{ color:'var(--text-tertiary)' }}>
                  Optimisation Bayésienne — {trialData.length}/{maxTrials} essais
                </p>
                <TrialsView trials={trialData} maxTrials={maxTrials} />
              </div>
            )}

            {/* Tab Prédictions */}
            {activeTab==='predictions' && (
              <div className="rounded-xl border p-4" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
                <p className="text-xs font-semibold mb-3" style={{ color:'var(--text-tertiary)' }}>
                  Résultats de prédiction — Jeu de test (dénormalisé)
                </p>
                <PredictionsView data={predictions} hyperparams={hyperparams} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════
// Sous-composants Phase 3 — DATASET card + PARAMÈTRES MODÈLE card
// ════════════════════════════════════════════════════════════════

function DatasetCard({ datasets, currentDataset, currentDatasetId, onSelect }) {
  const ds = currentDataset;
  const preprocessed = ds?.status === 'preprocessed';
  const eligible = datasets.filter(d => d.status === 'preprocessed' || d.id === currentDatasetId);

  return (
    <div className="rounded-xl border p-4 space-y-3"
      style={{
        background:  'var(--bg-card)',
        borderColor: preprocessed ? 'var(--brand-primary)' : 'var(--border-default)',
      }}>
      <div className="flex items-center gap-2">
        <Database size={14} style={{ color: 'var(--brand-primary)' }} />
        <p className="text-xs font-semibold uppercase tracking-widest"
           style={{ color: 'var(--text-tertiary)' }}>
          Dataset
        </p>
      </div>

      <select
        value={currentDatasetId || ''}
        onChange={(e) => onSelect(parseInt(e.target.value, 10))}
        className="w-full px-3 py-2 rounded-lg text-sm font-mono border outline-none"
        style={{
          background:  'var(--bg-elevated)',
          borderColor: 'var(--border-default)',
          color:       'var(--text-primary)',
        }}>
        {datasets.length === 0 && <option value="">Aucun dataset</option>}
        {!currentDatasetId && datasets.length > 0 && <option value="">— sélectionner —</option>}
        {datasets.map(d => (
          <option key={d.id} value={d.id}>
            {d.name} · {d.status}
          </option>
        ))}
      </select>

      {ds && (
        <>
          <div className="space-y-0.5 text-xs font-mono"
               style={{ color: 'var(--text-secondary)' }}>
            {ds.n_rows > 0 && (
              <div>📊 {ds.n_rows.toLocaleString()} lignes · {ds.n_composants} composants</div>
            )}
            {ds.n_failures > 0 && (
              <div>💥 {ds.n_failures} pannes · 🔧 {ds.n_maintenances} maint.</div>
            )}
            {ds.period_start && (
              <div>📅 {String(ds.period_start).slice(0,10)} → {String(ds.period_end).slice(0,10)}</div>
            )}
          </div>

          {ds.preproc_config && (
            <div className="rounded-lg p-2 border"
              style={{
                background:  'var(--bg-elevated)',
                borderColor: 'var(--border-subtle)',
              }}>
              <p className="text-[10px] mb-1.5"
                 style={{ color: 'var(--text-tertiary)' }}>
                ⚙️ Config Prétraitement (lecture seule)
              </p>
              <div className="text-xs font-mono space-y-0.5"
                   style={{ color: 'var(--text-secondary)' }}>
                <div>• Lookback : <b>{ds.preproc_config.lookback} jours</b></div>
                <div>• MAX RUL : <b>{ds.preproc_config.current_max_rul} jours</b></div>
                <div>• Poids RUL faibles : <b>×{ds.preproc_config.weight_factor}</b></div>
                <div>• Test ratio : <b>{Math.round((ds.preproc_config.test_ratio ?? 0.2) * 100)}%</b></div>
              </div>
            </div>
          )}

          {!preprocessed && (
            <div className="rounded-lg p-2.5 border flex items-start gap-2"
              style={{
                background:  'var(--tint-error-bg)',
                borderColor: 'var(--accent-orange)',
              }}>
              <AlertTriangle size={14} style={{ color: 'var(--accent-orange)', flexShrink: 0 }}/>
              <p className="text-xs leading-relaxed"
                 style={{ color: 'var(--accent-orange)' }}>
                Ce dataset n'a pas encore été <b>prétraité</b>. Va dans
                <b> Préparation Données → Prétraitement</b> avant de lancer l'entraînement.
              </p>
            </div>
          )}
        </>
      )}

      {!ds && datasets.length === 0 && (
        <div className="rounded-lg p-3 text-center border"
          style={{
            background:  'var(--bg-elevated)',
            borderColor: 'var(--accent-orange)',
          }}>
          <p className="text-xs" style={{ color: 'var(--accent-orange)' }}>
            ⚠️ Aucun dataset disponible
          </p>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Crée-en un dans <b>Préparation Données</b>.
          </p>
        </div>
      )}
    </div>
  );
}


function ModelParamsCard({ embeddingDim, setEmbeddingDim, mode }) {
  return (
    <div className="rounded-xl border p-4"
      style={{
        background:  'var(--bg-card)',
        borderColor: 'var(--border-default)',
      }}>
      <div className="flex items-center gap-2 mb-3">
        <Cpu size={14} style={{ color: 'var(--accent-purple)' }}/>
        <p className="text-xs font-semibold uppercase tracking-widest"
           style={{ color: 'var(--text-tertiary)' }}>
          Paramètres Modèle
        </p>
      </div>

      <label className="text-xs mb-1 block"
             style={{ color: 'var(--text-secondary)' }}>
        Embedding Composant
        {mode === 'auto' && (
          <span className="ml-2 text-[10px]"
                style={{ color: 'var(--accent-orange)' }}>
            (AutoML cherche aussi cette valeur)
          </span>
        )}
      </label>

      <div className="flex gap-2">
        {[4, 8, 16, 32].map(val => {
          const active = embeddingDim === val;
          return (
            <button key={val}
              onClick={() => setEmbeddingDim(val)}
              disabled={mode === 'auto'}
              className="flex-1 py-1.5 rounded text-sm font-mono border transition-all"
              style={{
                background:  active ? 'color-mix(in srgb, var(--accent-purple) 22%, var(--bg-card))' : 'var(--bg-elevated)',
                borderColor: active ? 'var(--accent-purple)' : 'var(--border-default)',
                color:       active ? 'var(--accent-purple)' : 'var(--text-tertiary)',
                cursor:      mode === 'auto' ? 'not-allowed' : 'pointer',
                opacity:     mode === 'auto' ? 0.6 : 1,
              }}>
              {val}
            </button>
          );
        })}
      </div>

      <p className="text-xs mt-3 leading-relaxed"
         style={{ color: 'var(--text-muted)' }}>
        💡 Chaque composant est transformé en vecteur de cette taille. Plus c'est grand,
        plus le modèle peut "personnaliser" sa prédiction par composant. Combiné aux
        9 features numériques via <code>Concatenate</code>.
      </p>
    </div>
  );
}
