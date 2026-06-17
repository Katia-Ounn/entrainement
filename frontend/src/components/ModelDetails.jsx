/**
 * ModelDetails.jsx — Card détails d'une expérience avec 5 mini-onglets.
 *
 *   📈 Régression       — 4 cards métriques + ScatterWithZones + Timeline
 *   🎯 Classification    — slider seuil + 4 cards + matrice confusion + camembert
 *   📊 Apprentissage     — courbes Loss + MAE + tableau époques collapsible
 *   🔮 Prédictions       — filtre composant + tableau panne réelle vs prédite
 *   ⚙️ Config            — hyperparams + dataset + fichiers + bouton copier JSON
 *
 *   Sticky bar en bas : Re-entraîner / Télécharger ZIP / Supprimer.
 *   Fetch /api/experiments/{id}/details au montage.
 */
import { useState, useEffect, useMemo } from 'react';
import {
  TrendingUp, Target, Activity, Calendar, Settings,
  RefreshCw, Download, Trash2, X, Copy, Check,
  AlertTriangle, ChevronDown, ChevronUp, Loader,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell,
  ScatterChart, Scatter, ReferenceLine,
  ComposedChart, Bar, Area,
} from 'recharts';
import ScatterWithZones from './charts/ScatterWithZones';

const API = 'http://localhost:8000';


export default function ModelDetails({
  experiment,
  onClose,
  onDelete,
  onRetrain,
}) {
  const [activeTab, setActiveTab] = useState('regression');
  const [details, setDetails]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error,   setError]       = useState(null);

  // Fetch détails complets
  useEffect(() => {
    if (!experiment?.id) return;
    let cancelled = false;
    setLoading(true); setError(null);
    fetch(`${API}/api/experiments/${experiment.id}/details`)
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(d => { if (!cancelled) setDetails(d); })
      .catch(e => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [experiment?.id]);

  const exp = details || experiment;
  const isFull = exp.mode === 'full';   // modèle de déploiement (réentraîné sur 100% des données)
  // Modèle full → on masque Classification + Prédictions (évaluation sur données
  // déjà vues = trompeur) et on montre un onglet "Prédiction panne future".
  const tabs = isFull
    ? [
        { id: 'next_failure', label: '🔮 Prédiction panne future', icon: Calendar },
        { id: 'training',     label: '📊 Apprentissage',           icon: Activity },
        { id: 'config',       label: '⚙️ Config',                  icon: Settings },
      ]
    : [
        { id: 'regression',     label: '📈 Régression',      icon: TrendingUp },
        { id: 'classification', label: '🎯 Classification',  icon: Target },
        { id: 'training',       label: '📊 Apprentissage',   icon: Activity },
        { id: 'predictions',    label: '🔮 Prédictions',     icon: Calendar },
        { id: 'config',         label: '⚙️ Config',          icon: Settings },
      ];
  // Si l'onglet actif n'existe pas pour ce type de modèle → on retombe sur le 1er.
  const safeTab = tabs.some(t => t.id === activeTab) ? activeTab : tabs[0].id;

  const handleDownloadZip = () => {
    window.open(`${API}/api/experiments/${experiment.id}/export`, '_blank');
  };

  return (
    <div className="rounded-2xl border mt-4 relative"
      style={{
        background:  'var(--bg-elevated)',
        borderColor: 'var(--border-strong)',
      }}>
      {/* Header */}
      <div className="flex items-start justify-between px-6 pt-5 pb-3">
        <div>
          <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
            {exp.name}
          </h3>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
            {exp.architecture} · {exp.mode}
            {exp.dataset_name && ` · Dataset "${exp.dataset_name}"`}
            {exp.created_at && ' · ' + new Date(exp.created_at).toLocaleString('fr-FR')}
          </p>
        </div>
        <button onClick={onClose}
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ color: 'var(--text-muted)' }}>
          <X size={16}/>
        </button>
      </div>

      {/* Onglets */}
      <div className="px-6">
        <div className="flex gap-1 p-1 rounded-lg border"
          style={{
            background:  'var(--bg-base)',
            borderColor: 'var(--border-default)',
          }}>
          {tabs.map(tab => {
            const Icon = tab.icon;
            const isActive = safeTab === tab.id;
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded text-xs font-semibold transition-all"
                style={{
                  background:  isActive ? 'var(--bg-elevated)' : 'transparent',
                  color:       isActive ? 'var(--brand-primary)' : 'var(--text-muted)',
                  border:      isActive ? '1px solid var(--brand-primary)' : '1px solid transparent',
                }}>
                <Icon size={13}/>
                <span className="hidden md:inline">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Contenu onglet (padding-bottom = espace pour sticky bar) */}
      <div className="px-6 pt-4 pb-24">
        {loading && (
          <div className="flex items-center gap-2 text-sm py-6"
               style={{ color: 'var(--text-tertiary)' }}>
            <Loader size={14} className="animate-spin"/> Chargement détails…
          </div>
        )}
        {error && (
          <div className="rounded-lg p-3 text-xs border"
            style={{
              background:  'var(--tint-error-bg)',
              borderColor: 'var(--error)',
              color:       'var(--error)',
            }}>
            <AlertTriangle size={14} className="inline mr-1.5"/>
            {error}
          </div>
        )}
        {!loading && !error && (
          <>
            {safeTab === 'regression'     && <RegressionTab exp={exp}/>}
            {safeTab === 'classification' && <ClassificationTab exp={exp} experimentId={experiment.id}/>}
            {safeTab === 'training'       && <TrainingTab exp={exp}/>}
            {safeTab === 'predictions'    && <PredictionsTab exp={exp}/>}
            {safeTab === 'next_failure'   && (
              <div>
                <p className="text-xs mb-3" style={{ color:'var(--text-tertiary)' }}>
                  La prochaine panne <b>à venir</b> pour chaque composant (modèle déployé sur 100% des données).
                  C'est le futur → il n'y a <b>pas</b> de valeur réelle à comparer.
                </p>
                <NextFailuresView expId={experiment.id} />
              </div>
            )}
            {safeTab === 'config'         && <ConfigTab exp={exp}/>}
          </>
        )}
      </div>

      {/* Sticky action bar */}
      <div className="sticky bottom-0 px-6 py-3 flex flex-wrap items-center justify-end gap-2 border-t"
        style={{
          background:  'var(--bg-elevated)',
          borderColor: 'var(--border-default)',
          backdropFilter: 'blur(8px)',
          borderRadius: '0 0 1rem 1rem',
        }}>
        <button onClick={() => onRetrain?.(experiment.id)}
          className="px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 border"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--accent-blue)',
            color:       'var(--accent-blue)',
          }}>
          <RefreshCw size={13}/> Re-entraîner
        </button>
        <button onClick={handleDownloadZip}
          className="px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 border"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--success)',
            color:       'var(--success)',
          }}>
          <Download size={13}/> Télécharger ZIP
        </button>
        <button onClick={() => onDelete?.(experiment.id)}
          className="px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 border"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--error)',
            color:       'var(--error)',
          }}>
          <Trash2 size={13}/> Supprimer
        </button>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// 📈 Onglet Régression
// ═══════════════════════════════════════════════════════════════
function RegressionTab({ exp }) {
  const preds = exp.predictions;
  const maxRul = preds?.current_max_rul ?? exp.hyperparams?.current_max_rul ?? 30;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="R² Score" value={exp.r2?.toFixed(4)}             color="var(--success)"        sub="↑ proche de 1"/>
        <MetricCard label="MAE (j)"  value={fmt(exp.mae, 'j')}              color="var(--accent-blue)"    sub="Mean Abs. Error"/>
        <MetricCard label="RMSE (j)" value={fmt(exp.rmse, 'j')}             color="var(--accent-orange)"  sub="Root MSE"/>
        <MetricCard label="MAPE"     value={exp.mape != null ? `${exp.mape.toFixed(1)}%` : '—'} color="var(--accent-purple)" sub="% Mean Abs. Error"/>
      </div>

      {preds?.y_true?.length > 0 ? (
        <>
          <ScatterWithZones y_true={preds.y_true} y_pred={preds.y_pred} max_rul={maxRul}/>

          <div className="rounded-xl border p-4"
            style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
            <p className="text-xs font-semibold uppercase tracking-widest mb-2"
               style={{ color:'var(--text-tertiary)' }}>
              Timeline RUL — Réel vs Prédit (150 premiers samples)
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={preds.y_true.slice(0, 150).map((v, i) => ({
                  i, real: v, pred: preds.y_pred[i],
                }))} margin={{ top:5, right:10, bottom:5, left:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                <XAxis dataKey="i" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }} domain={[0, maxRul]}
                       tickFormatter={v => `${Math.round(v)}j`}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius:8,
                  color:'var(--text-primary)',
                }} formatter={(v,n) => [`${Number(v).toFixed(1)} j`, n]}/>
                <Legend wrapperStyle={{ fontSize: 11 }}/>
                <Line type="monotone" dataKey="real" name="RUL Réel"   stroke="var(--success)"     dot={false} strokeWidth={2}/>
                <Line type="monotone" dataKey="pred" name="RUL Prédit" stroke="var(--accent-blue)" dot={false} strokeWidth={2} strokeDasharray="5 3"/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <EmptyHint text="Pas de prédictions sauvegardées pour ce modèle."/>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// 🎯 Onglet Classification (avec slider seuil)
// ═══════════════════════════════════════════════════════════════
function ClassificationTab({ exp, experimentId }) {
  const [threshold, setThreshold] = useState(10);
  const [recomputed, setRecomputed] = useState(null);
  const [loading, setLoading]     = useState(false);

  // Recalcule via API quand on change le seuil
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`${API}/api/experiments/${experimentId}/recompute_classification?threshold=${threshold}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (!cancelled && d) setRecomputed(d); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [experimentId, threshold]);

  const m = recomputed || {
    accuracy:  exp.accuracy,
    precision: exp.precision,
    recall:    exp.recall,
    f1:        exp.f1_score,
    threshold: 10,
    confusion_matrix: null,
  };

  // Camembert sain/alerte
  const preds = exp.predictions;
  const pieData = useMemo(() => {
    if (!preds?.y_true) return null;
    const alert  = preds.y_true.filter(v => v <= threshold).length;
    const sain   = preds.y_true.length - alert;
    return [
      { name: `Sain (>${threshold}j)`,    value: sain },
      { name: `Alerte (≤${threshold}j)`,  value: alert },
    ];
  }, [preds, threshold]);

  return (
    <div className="space-y-5">
      {/* Slider seuil */}
      <div className="rounded-xl border p-4"
        style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-semibold uppercase tracking-widest"
                 style={{ color:'var(--text-tertiary)' }}>
            Seuil d'alerte
          </label>
          <span className="text-sm font-mono font-bold"
                style={{ color: 'var(--accent-orange)' }}>
            {threshold} jours
          </span>
        </div>
        <input type="range" min={1} max={60} step={1} value={threshold}
          onChange={e => setThreshold(Number(e.target.value))}
          className="w-full"
          style={{ accentColor: 'var(--accent-orange)' }}/>
        <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
          🎚️ Recalcule en temps réel : RUL ≤ seuil = composant en <b>alerte</b>.
          {loading && <span className="ml-2"><Loader size={10} className="inline animate-spin"/> Recalcul…</span>}
        </p>
      </div>

      {/* 4 cards métriques */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Accuracy"  value={fmtPct(m.accuracy)}  color={accColor(m.accuracy, 0.85)} sub="(TP+TN)/total"/>
        <MetricCard label="Precision" value={fmtPct(m.precision)} color={accColor(m.precision, 0.80)} sub="TP/(TP+FP)"/>
        <MetricCard label="Recall"    value={fmtPct(m.recall)}    color={accColor(m.recall, 0.80)}   sub="TP/(TP+FN)"/>
        <MetricCard label="F1 Score"  value={fmtPct(m.f1)}        color={accColor(m.f1, 0.80)}       sub="moy. harmonique"/>
      </div>

      {/* Matrice confusion + camembert */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {m.confusion_matrix && (
          <div className="rounded-xl border p-4"
            style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
            <p className="text-xs font-semibold uppercase tracking-widest mb-3"
               style={{ color:'var(--text-tertiary)' }}>
              Matrice de confusion
            </p>
            <ConfusionMatrix matrix={m.confusion_matrix} labels={m.labels || ['sain', 'alerte']}/>
          </div>
        )}

        {pieData && (
          <div className="rounded-xl border p-4"
            style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
            <p className="text-xs font-semibold uppercase tracking-widest mb-3"
               style={{ color:'var(--text-tertiary)' }}>
              Répartition Sain / Alerte (jeu de test)
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} dataKey="value" cx="50%" cy="50%"
                     outerRadius={70} innerRadius={36} label>
                  <Cell fill="var(--success)"/>
                  <Cell fill="var(--error)"/>
                </Pie>
                <Legend wrapperStyle={{ fontSize: 11, color:'var(--text-tertiary)' }}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius:8,
                  color:'var(--text-primary)',
                }}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}


function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || matrix.length !== 2) return null;
  const [[tn, fp], [fn, tp]] = matrix;
  const total = tn + fp + fn + tp || 1;
  const Cell_ = ({ v, color, sub }) => (
    <div className="rounded-lg p-3 text-center"
      style={{
        background: `color-mix(in srgb, ${color} 20%, var(--bg-card))`,
        border:     `1px solid ${color}`,
      }}>
      <div className="text-2xl font-bold font-mono" style={{ color }}>{v}</div>
      <div className="text-[10px] mt-0.5" style={{ color:'var(--text-muted)' }}>{sub}</div>
      <div className="text-[10px]" style={{ color:'var(--text-tertiary)' }}>{Math.round(100*v/total)}%</div>
    </div>
  );

  return (
    <div>
      <div className="grid grid-cols-3 gap-1 text-[10px] font-mono"
           style={{ color: 'var(--text-tertiary)' }}>
        <div></div>
        <div className="text-center">Préd. {labels[0]}</div>
        <div className="text-center">Préd. {labels[1]}</div>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-1">
        <div className="text-[10px] font-mono flex items-center justify-end pr-1"
             style={{ color: 'var(--text-tertiary)' }}>
          Réel {labels[0]}
        </div>
        <Cell_ v={tn} color="var(--success)"       sub="TN"/>
        <Cell_ v={fp} color="var(--accent-orange)" sub="FP"/>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-2">
        <div className="text-[10px] font-mono flex items-center justify-end pr-1"
             style={{ color: 'var(--text-tertiary)' }}>
          Réel {labels[1]}
        </div>
        <Cell_ v={fn} color="var(--error)"   sub="FN"/>
        <Cell_ v={tp} color="var(--success)" sub="TP"/>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// 📊 Onglet Apprentissage (Loss + MAE par époque)
// ═══════════════════════════════════════════════════════════════
function TrainingTab({ exp }) {
  const history = exp.training_history;
  const [showTable, setShowTable] = useState(false);

  if (!history || !history.length) {
    return <EmptyHint text="Pas d'historique d'apprentissage sauvegardé."/>;
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border p-4"
        style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
        <p className="text-xs font-semibold uppercase tracking-widest mb-2"
           style={{ color:'var(--text-tertiary)' }}>
          Loss (MSE)
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={history} margin={{ top:5, right:10, bottom:5, left:0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
            <XAxis dataKey="epoch" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
            <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}
                   tickFormatter={v => v.toFixed(4)}/>
            <Tooltip contentStyle={{
              background:'var(--bg-elevated)',
              border:'1px solid var(--border-strong)',
              borderRadius:8,
              color:'var(--text-primary)',
            }} formatter={(v,n)=>[Number(v).toFixed(6), n]} labelFormatter={l=>`Époque ${l}`}/>
            <Legend wrapperStyle={{ fontSize: 11 }}/>
            <Line type="monotone" dataKey="loss"     name="Train Loss" stroke="var(--accent-pink)"   dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="val_loss" name="Val Loss"   stroke="var(--accent-purple)" dot={false} strokeWidth={2} strokeDasharray="5 3"/>
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-xl border p-4"
        style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
        <p className="text-xs font-semibold uppercase tracking-widest mb-2"
           style={{ color:'var(--text-tertiary)' }}>
          MAE
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={history} margin={{ top:5, right:10, bottom:5, left:0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
            <XAxis dataKey="epoch" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
            <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}
                   tickFormatter={v => v.toFixed(4)}/>
            <Tooltip contentStyle={{
              background:'var(--bg-elevated)',
              border:'1px solid var(--border-strong)',
              borderRadius:8,
              color:'var(--text-primary)',
            }} formatter={(v,n)=>[Number(v).toFixed(6), n]} labelFormatter={l=>`Époque ${l}`}/>
            <Legend wrapperStyle={{ fontSize: 11 }}/>
            <Line type="monotone" dataKey="mae"     name="Train MAE" stroke="var(--accent-orange)" dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="val_mae" name="Val MAE"   stroke="var(--accent-green)"  dot={false} strokeWidth={2} strokeDasharray="5 3"/>
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Collapsible table */}
      <button onClick={() => setShowTable(v => !v)}
        className="w-full px-3 py-2 rounded-lg text-xs font-semibold border flex items-center justify-between"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--border-default)',
          color:       'var(--text-tertiary)',
        }}>
        <span>📋 Voir les {history.length} époques en détail</span>
        {showTable ? <ChevronUp size={13}/> : <ChevronDown size={13}/>}
      </button>
      {showTable && (
        <div className="rounded-xl border overflow-hidden"
          style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <div className="overflow-x-auto" style={{ maxHeight: 280 }}>
            <table className="text-xs font-mono w-full"
                   style={{ color:'var(--text-secondary)' }}>
              <thead style={{ background:'var(--bg-card-alt)', position:'sticky', top:0 }}>
                <tr>
                  {['Époque','Loss','Val Loss','MAE','Val MAE'].map(h => (
                    <th key={h} className="px-2 py-1.5 text-right"
                        style={{ color:'var(--text-tertiary)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((e, i) => (
                  <tr key={i} style={{ background: i % 2 ? 'var(--bg-card-alt)' : 'transparent' }}>
                    <td className="px-2 py-1 text-right" style={{ color:'var(--text-primary)' }}>{e.epoch}</td>
                    <td className="px-2 py-1 text-right">{e.loss?.toFixed(6)}</td>
                    <td className="px-2 py-1 text-right">{e.val_loss?.toFixed(6)}</td>
                    <td className="px-2 py-1 text-right">{e.mae?.toFixed(6)}</td>
                    <td className="px-2 py-1 text-right">{e.val_mae?.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// 🔮 Onglet Prédictions (filtre composant + table)
// ═══════════════════════════════════════════════════════════════
function PredictionsTab({ exp }) {
  return (
    <div className="space-y-6">

      {/* ── SECTION 1 : prédiction du FUTUR (pas de valeur réelle) ── */}
      <div>
        <h3 className="text-sm font-bold mb-1" style={{ color:'var(--text-primary)' }}>
          🔮 Prédiction du futur — Prochaine panne
        </h3>
        <p className="text-xs mb-3" style={{ color:'var(--text-tertiary)' }}>
          La prochaine panne <b>à venir</b> pour chaque composant. C'est le futur → il n'y a <b>pas</b> de valeur réelle à comparer.
        </p>
        <NextFailuresView expId={exp.id} />
      </div>

      {/* ── SECTION 2 : évaluation sur le PASSÉ (vérité connue) ── */}
      <div>
        <h3 className="text-sm font-bold mb-1" style={{ color:'var(--text-primary)' }}>
          📊 Évaluation du modèle — sur données passées
        </h3>
        <p className="text-xs mb-3" style={{ color:'var(--text-tertiary)' }}>
          Compare le <b>RUL réel</b> (connu, car historique) au <b>RUL prédit</b> pour mesurer la précision —
          ce n'est <b>pas</b> une prédiction du futur. ⚠️ Ce modèle étant entraîné sur ces données, les scores sont optimistes.
        </p>
        <PredictionsTable exp={exp} />
      </div>

    </div>
  );
}

// Prédiction de la PROCHAINE panne par composant (inférence de déploiement)
function NextFailuresView({ expId }) {
  const [data, setData]         = useState(null);
  const [err, setErr]           = useState(null);
  const [loading, setLoading]   = useState(true);
  const [selected, setSelected] = useState('');

  useEffect(() => {
    let alive = true;
    setLoading(true); setErr(null);
    fetch(`${API}/api/experiments/${expId}/next_failures`)
      .then(r => r.json().then(b => (r.ok ? b : Promise.reject(b.detail || 'Erreur'))))
      .then(d => { if (!alive) return; setData(d); setSelected(d.components?.[0]?.comp || ''); })
      .catch(e => { if (alive) setErr(typeof e === 'string' ? e : 'Erreur de prédiction'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [expId]);

  if (loading) return (
    <div className="rounded-xl border p-4 flex items-center gap-2"
      style={{ background:'var(--bg-card)', borderColor:'var(--border-default)', color:'var(--text-tertiary)' }}>
      <Loader size={16} className="animate-spin"/> Calcul des prédictions de prochaine panne…
    </div>
  );
  if (err)  return <EmptyHint text={`Prédiction de prochaine panne indisponible : ${err}`} />;
  if (!data?.components?.length) return <EmptyHint text="Aucune prédiction de prochaine panne disponible." />;

  const comps   = data.components;
  const sel     = comps.find(c => c.comp === selected) || comps[0];
  const todayTs = new Date(data.data_end).getTime();
  const urgency = (rul) => rul <= 7 ? 'var(--error)' : rul <= 21 ? 'var(--accent-orange)' : 'var(--success)';

  // Données du graphe combiné : barres = pannes · courbe (aire) = RUL restant
  const maxRul = data.max_rul || 30;
  const predTs = new Date(sel.predicted_next_failure).getTime();
  const chartData = [
    ...(sel.past_failures || [])
      .map(d => ({ x: new Date(d).getTime(), failPast: maxRul, date: d, kind: 'Panne passée' }))
      .filter(p => !Number.isNaN(p.x)),
    { x: todayTs, rul: sel.predicted_rul, date: data.data_end, kind: "Aujourd'hui" },
    { x: predTs,  failPred: maxRul, rul: 0, date: sel.predicted_next_failure, kind: 'Prochaine panne (prédite)' },
  ].sort((a, b) => a.x - b.x);

  const xsAll = chartData.map(p => p.x).filter(v => !Number.isNaN(v));
  const xmin  = Math.min(...xsAll), xmax = Math.max(...xsAll);
  const pad   = (xmax - xmin) * 0.05 || 86400000;

  return (
    <div className="rounded-xl border p-4 space-y-3"
      style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-sm font-semibold" style={{ color:'var(--text-primary)' }}>
          🔮 Prochaine panne prédite par composant
        </p>
        <span className="text-[11px] font-mono" style={{ color:'var(--text-muted)' }}>
          {data.n} composants · jusqu'au {data.data_end} · RUL ≤ {data.max_rul}j
        </span>
      </div>

      {/* Sélecteur composant (trié par urgence) */}
      <div className="flex items-center gap-2 flex-wrap">
        <label className="text-xs font-semibold" style={{ color:'var(--text-tertiary)' }}>Composant :</label>
        <select value={sel.comp} onChange={e => setSelected(e.target.value)}
          className="px-2 py-1 rounded text-xs font-mono border outline-none"
          style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-primary)' }}>
          {comps.map(c => (
            <option key={c.comp} value={c.comp}>{c.comp} — dans {c.predicted_rul}j</option>
          ))}
        </select>
      </div>

      {/* Stat prochaine panne */}
      <div className="flex items-center gap-6 flex-wrap">
        <div>
          <p className="text-[11px]" style={{ color:'var(--text-tertiary)' }}>Prochaine panne prédite</p>
          <p className="text-lg font-bold" style={{ color: urgency(sel.predicted_rul) }}>
            {sel.predicted_next_failure}
            <span className="text-xs font-mono ml-1">(dans {sel.predicted_rul} j)</span>
          </p>
        </div>
        <div>
          <p className="text-[11px]" style={{ color:'var(--text-tertiary)' }}>Pannes passées</p>
          <p className="text-sm font-mono" style={{ color:'var(--text-secondary)' }}>{sel.n_failures}</p>
        </div>
      </div>

      {/* Graphe combiné : barres = pannes (gris passées / rouge prédite) · courbe = RUL restant */}
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={chartData} margin={{ top: 12, right: 24, bottom: 18, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis type="number" dataKey="x" scale="time" domain={[xmin - pad, xmax + pad]}
            tickFormatter={(t) => new Date(t).toISOString().slice(0, 10)}
            tick={{ fontSize: 10, fill: 'var(--text-tertiary)' }} />
          <YAxis domain={[0, maxRul]} allowDecimals={false}
            tick={{ fontSize: 10, fill: 'var(--text-tertiary)' }}
            label={{ value: 'RUL (j)', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-tertiary)' }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload;
              return (
                <div style={{ background:'var(--bg-elevated)', border:'1px solid var(--border-default)', borderRadius:6, padding:'6px 8px', fontSize:11 }}>
                  <div style={{ color:'var(--text-primary)', fontWeight:600 }}>{p.date}</div>
                  <div style={{ color:'var(--text-tertiary)' }}>{p.kind}</div>
                  {p.rul != null && <div style={{ color:'var(--accent-orange)' }}>RUL : {p.rul} j</div>}
                </div>
              );
            }} />
          <ReferenceLine x={todayTs} stroke="var(--text-muted)" strokeDasharray="4 4"
            label={{ value: "aujourd'hui", fontSize: 9, fill: 'var(--text-muted)', position: 'insideTopRight' }} />
          <Bar dataKey="failPast" name="Pannes passées"   fill="var(--text-secondary)" barSize={12} radius={[2,2,0,0]} />
          <Bar dataKey="failPred" name="Prochaine panne"  fill="var(--error)"          barSize={12} radius={[2,2,0,0]} />
          <Area type="monotone" dataKey="rul" name="RUL restant (j)"
            stroke="var(--accent-orange)" fill="var(--accent-orange)" fillOpacity={0.18}
            strokeWidth={2} connectNulls dot={{ r: 3, fill: 'var(--accent-orange)' }} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 text-[11px] flex-wrap" style={{ color:'var(--text-tertiary)' }}>
        <span style={{ color:'var(--text-secondary)' }}>▮ Pannes passées</span>
        <span style={{ color:'var(--error)' }}>▮ Prochaine panne prédite</span>
        <span style={{ color:'var(--accent-orange)' }}>▬ RUL restant (courbe)</span>
        <span>┊ aujourd'hui ({data.data_end})</span>
      </div>
      <p className="text-[11px]" style={{ color:'var(--text-muted)' }}>
        ℹ️ Le RUL est plafonné à {data.max_rul} jours → les prédictions ne dépassent pas {data.max_rul}j.
      </p>
    </div>
  );
}

// Tableau d'évaluation réel vs prédit (sur le jeu de test)
function PredictionsTable({ exp }) {
  const preds = exp.predictions;
  const [filter, setFilter] = useState('all');

  if (!preds?.y_true?.length) {
    return <EmptyHint text="Pas de fichier predictions.csv pour ce modèle."/>;
  }

  // Liste composants distincts
  const components = useMemo(() => {
    const set = new Set(preds.comp || []);
    return ['all', ...Array.from(set).sort()];
  }, [preds.comp]);

  // Lignes filtrées
  const rows = useMemo(() => {
    const out = [];
    const len = Math.min(preds.y_true.length, preds.y_pred.length);
    for (let i = 0; i < len; i++) {
      const c = preds.comp?.[i] ?? '';
      if (filter !== 'all' && c !== filter) continue;
      const err = Math.abs(preds.y_pred[i] - preds.y_true[i]);
      out.push({
        date:  preds.dates?.[i],
        comp:  c,
        y_true: preds.y_true[i],
        y_pred: preds.y_pred[i],
        err,
      });
    }
    return out.slice(0, 300);   // limite affichage
  }, [preds, filter]);

  const qualOf = (err) => err <= 2
    ? { label: '✅ Excellent', color: 'var(--success)' }
    : err <= 5
    ? { label: '🟠 OK',       color: 'var(--accent-orange)' }
    : { label: '❌ À revoir',  color: 'var(--error)' };

  return (
    <div className="space-y-3">
      {/* Filtre */}
      <div className="flex items-center gap-2">
        <label className="text-xs font-semibold"
               style={{ color: 'var(--text-tertiary)' }}>
          Filtre composant :
        </label>
        <select value={filter} onChange={e => setFilter(e.target.value)}
          className="px-2 py-1 rounded text-xs font-mono border outline-none"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--border-default)',
            color:       'var(--text-primary)',
          }}>
          {components.map(c => (
            <option key={c} value={c}>{c === 'all' ? `Tous (${preds.y_true.length} samples)` : c}</option>
          ))}
        </select>
        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          {rows.length} ligne{rows.length > 1 ? 's' : ''} affichée{rows.length > 1 ? 's' : ''}
        </span>
      </div>

      {/* Table */}
      <div className="rounded-xl border overflow-hidden"
        style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
        <div className="overflow-x-auto" style={{ maxHeight: 480 }}>
          <table className="text-xs font-mono w-full"
                 style={{ color:'var(--text-secondary)' }}>
            <thead style={{ background:'var(--bg-card-alt)', position:'sticky', top:0 }}>
              <tr>
                <th className="px-3 py-2 text-left"  style={{ color:'var(--text-tertiary)' }}>Date</th>
                <th className="px-3 py-2 text-left"  style={{ color:'var(--text-tertiary)' }}>Composant</th>
                <th className="px-3 py-2 text-right" style={{ color:'var(--text-tertiary)' }}>RUL Réel (j)</th>
                <th className="px-3 py-2 text-right" style={{ color:'var(--text-tertiary)' }}>RUL Prédit (j)</th>
                <th className="px-3 py-2 text-right" style={{ color:'var(--text-tertiary)' }}>Écart (j)</th>
                <th className="px-3 py-2 text-center" style={{ color:'var(--text-tertiary)' }}>Évaluation</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const q = qualOf(r.err);
                return (
                  <tr key={i} style={{ background: i % 2 ? 'var(--bg-card-alt)' : 'transparent' }}>
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-tertiary)' }}>
                      {String(r.date || '').slice(0, 10)}
                    </td>
                    <td className="px-3 py-1.5 truncate" style={{ color: 'var(--text-primary)' }}>{r.comp}</td>
                    <td className="px-3 py-1.5 text-right font-bold" style={{ color: 'var(--success)' }}>{Number(r.y_true).toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-right font-bold" style={{ color: 'var(--accent-blue)' }}>{Number(r.y_pred).toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-right" style={{ color: q.color }}>{Number(r.err).toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-center" style={{ color: q.color }}>{q.label}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {preds.y_true.length > rows.length && (
        <p className="text-[11px]" style={{ color:'var(--text-muted)' }}>
          ℹ️ Limité à 300 lignes affichées (total : {preds.y_true.length}).
        </p>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// ⚙️ Onglet Config (hyperparams + dataset + fichiers + copy JSON)
// ═══════════════════════════════════════════════════════════════
function ConfigTab({ exp }) {
  const [copied, setCopied] = useState(false);

  const copyJson = async () => {
    try {
      const blob = JSON.stringify({
        experiment_id: exp.id,
        name:          exp.name,
        architecture:  exp.architecture,
        mode:          exp.mode,
        hyperparams:   exp.hyperparams,
        dataset_id:    exp.dataset_id,
      }, null, 2);
      await navigator.clipboard.writeText(blob);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) {}
  };

  return (
    <div className="space-y-4">
      {/* Hyperparams */}
      {exp.hyperparams && Object.keys(exp.hyperparams).length > 0 ? (
        <div className="rounded-xl border p-4"
          style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold uppercase tracking-widest"
               style={{ color:'var(--text-tertiary)' }}>
              Hyperparamètres utilisés
            </p>
            <button onClick={copyJson}
              className="px-2 py-1 rounded text-[11px] font-semibold flex items-center gap-1 border"
              style={{
                background:  'var(--bg-elevated)',
                borderColor: 'var(--border-default)',
                color:       copied ? 'var(--success)' : 'var(--text-tertiary)',
              }}>
              {copied ? <><Check size={11}/> Copié !</> : <><Copy size={11}/> Copier JSON</>}
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(exp.hyperparams).map(([k, v]) => (
              <div key={k} className="text-xs font-mono px-2 py-1 rounded border"
                style={{
                  background:  'var(--bg-elevated)',
                  borderColor: 'var(--border-strong)',
                  color:       'var(--accent-blue)',
                }}>
                <span style={{ color: 'var(--text-tertiary)' }}>{k}:</span>{' '}
                <span>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <EmptyHint text="Pas d'hyperparamètres sauvegardés."/>
      )}

      {/* Dataset */}
      {exp.dataset_name && (
        <div className="rounded-xl border p-4"
          style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2"
             style={{ color:'var(--text-tertiary)' }}>
            Dataset utilisé
          </p>
          <p className="text-sm font-mono"
             style={{ color:'var(--brand-primary)' }}>
            📦 {exp.dataset_name} <span style={{ color:'var(--text-muted)' }}>(id #{exp.dataset_id})</span>
          </p>
        </div>
      )}

      {/* Fichiers sauvegardés */}
      {exp.files && Object.keys(exp.files).length > 0 && (
        <div className="rounded-xl border p-4"
          style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-3"
             style={{ color:'var(--text-tertiary)' }}>
            Fichiers sauvegardés ({exp.model_dir})
          </p>
          <div className="space-y-1 text-xs font-mono">
            {Object.values(exp.files).map(f => (
              <div key={f.name} className="flex justify-between items-baseline">
                <span style={{ color:'var(--accent-blue)' }}>{f.name}</span>
                <span style={{ color:'var(--text-muted)' }}>{f.size_kb} ko</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Helpers UI
// ═══════════════════════════════════════════════════════════════
function MetricCard({ label, value, color, sub }) {
  return (
    <div className="rounded-xl border p-3"
      style={{
        background:  'var(--bg-card)',
        borderColor: color,
      }}>
      <p className="text-[11px]" style={{ color:'var(--text-tertiary)' }}>{label}</p>
      <p className="text-xl font-bold font-mono" style={{ color }}>
        {value ?? '—'}
      </p>
      {sub && <p className="text-[10px] mt-0.5" style={{ color:'var(--text-muted)' }}>{sub}</p>}
    </div>
  );
}

function EmptyHint({ text }) {
  return (
    <div className="rounded-xl border p-6 text-center text-xs"
      style={{
        background:  'var(--bg-card)',
        borderColor: 'var(--border-default)',
        color:       'var(--text-muted)',
      }}>
      {text}
    </div>
  );
}

function fmt(v, unit = '') {
  if (v == null || !isFinite(v)) return '—';
  return `${Number(v).toFixed(2)}${unit ? ' ' + unit : ''}`;
}

function fmtPct(v) {
  if (v == null || !isFinite(v)) return '—';
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function accColor(v, good = 0.8) {
  if (v == null) return 'var(--text-muted)';
  if (v >= good)         return 'var(--success)';
  if (v >= good - 0.15)  return 'var(--accent-orange)';
  return 'var(--error)';
}
