/**
 * Leaderboard.jsx — Tableau comparatif des expériences Cevital.
 *
 *  Phase 4 (enrichi, PAS refonte) :
 *   ✅ Header / InfoBox SQLite / Graphe R² comparaison conservés
 *   ✅ Colonnes tableau enrichies : F1 + MAPE en plus, MAE en JOURS
 *   ✅ Card détails déléguée à <ModelDetails> (5 mini-onglets + sticky bar)
 *   ✅ Bouton Re-entraîner : pré-remplit le formulaire d'entraînement
 *      via AppContext.requestRetrain()
 */
import { useState, useEffect, useCallback } from 'react';
import {
  Trophy, RefreshCw, Trash2, ChevronUp, ChevronDown, Eye, Database,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { useApp } from '../AppContext';
import ModelDetails from './ModelDetails';
import toast from 'react-hot-toast';

const API = 'http://localhost:8000';

const ARCH_COLORS = {
  LSTM:        'var(--accent-blue)',
  GRU:         'var(--accent-green)',
  RNN:         'var(--accent-orange)',
  Transformer: 'var(--accent-purple)',
};

function StatusBadge({ status }) {
  const map = {
    completed: { bg: 'var(--tint-success-bg)', color: 'var(--success)'       },
    running:   { bg: 'var(--tint-info-bg)',    color: 'var(--accent-blue)'   },
    failed:    { bg: 'var(--tint-error-bg)',   color: 'var(--error)'         },
    pending:   { bg: 'var(--bg-card)',         color: 'var(--text-tertiary)' },
  };
  const s = map[status] || map.pending;
  return (
    <span className="px-2 py-0.5 rounded-full text-xs font-mono"
      style={{ background: s.bg, color: s.color }}>
      {status}
    </span>
  );
}

// Couleurs sémantiques F1 / MAPE
const f1Color = (v) => v == null   ? 'var(--text-muted)'
                     : v >= 0.8    ? 'var(--success)'
                     : v >= 0.6    ? 'var(--accent-orange)'
                     :               'var(--error)';
const mapeColor = (v) => v == null ? 'var(--text-muted)'
                      : v < 10    ? 'var(--success)'
                      : v < 20    ? 'var(--accent-orange)'
                      :              'var(--error)';


export default function Leaderboard() {
  const { requestRetrain } = useApp();

  const [experiments, setExperiments] = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [sortKey,     setSortKey]     = useState('r2');
  const [sortAsc,     setSortAsc]     = useState(false);
  const [selectedId,  setSelectedId]  = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // ─── Fetch ─────────────────────────────────────────────────
  const fetchExperiments = useCallback(async () => {
    setLoading(true);
    try {
      const res  = await fetch(`${API}/api/experiments`);
      const data = await res.json();
      setExperiments(Array.isArray(data) ? data : []);
    } catch (_) {
      setExperiments([]);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { fetchExperiments(); }, [fetchExperiments]);
  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(fetchExperiments, 4000);
    return () => clearInterval(id);
  }, [autoRefresh, fetchExperiments]);

  // ─── Actions ───────────────────────────────────────────────
  const handleDelete = async (id) => {
    if (!confirm('Supprimer définitivement cette expérience et ses artefacts ?')) return;
    await fetch(`${API}/api/experiments/${id}`, { method: 'DELETE' });
    if (selectedId === id) setSelectedId(null);
    fetchExperiments();
  };

  const handleRetrain = async (id) => {
    try {
      const res  = await fetch(`${API}/api/experiments/${id}/retrain`, { method: 'POST' });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      // Pousse dans AppContext → App.jsx écoute et bascule sur Entraînement
      requestRetrain({
        hyperparams:  payload.hyperparams,
        dataset_id:   payload.dataset_id,
        architecture: payload.architecture,
        mode:         payload.mode,
        name:         payload.name,
        source_experiment_id: id,   // 🆕 pour le réentraînement "toutes les données" (meilleure époque)
      });
      toast.success(`Hyperparamètres chargés depuis "${payload.name || 'expérience'}"`);
    } catch (e) {
      toast.error(`Erreur Re-entraîner : ${e.message}`);
    }
  };

  // ─── Sort ──────────────────────────────────────────────────
  const handleSort = (key) => {
    if (sortKey === key) setSortAsc(v => !v);
    else { setSortKey(key); setSortAsc(false); }
  };

  const sorted = [...experiments].sort((a, b) => {
    const av = a[sortKey] ?? -Infinity;
    const bv = b[sortKey] ?? -Infinity;
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  });

  const completed = experiments.filter(e => e.status === 'completed' && e.r2 != null);
  const selected  = experiments.find(e => e.id === selectedId) || null;

  const SortIcon = ({ k }) => sortKey === k
    ? (sortAsc ? <ChevronUp size={11}/> : <ChevronDown size={11}/>)
    : null;

  const thCls = "px-3 py-2 text-left text-xs font-semibold uppercase tracking-widest cursor-pointer select-none whitespace-nowrap";

  return (
    <div className="space-y-6">

      {/* ── Header ───────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              background:  'color-mix(in srgb, var(--accent-orange) 25%, var(--bg-card))',
              border:      '1px solid var(--accent-orange)',
            }}>
            <Trophy size={18} style={{ color: 'var(--accent-orange)' }}/>
          </div>
          <div>
            <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Leaderboard
            </h2>
            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
              {experiments.length} expérience{experiments.length > 1 ? 's' : ''} ·
              {' '}{completed.length} terminée{completed.length > 1 ? 's' : ''} ·
              {' '}Stockées dans SQLite
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded-lg border"
            style={{
              background:  'var(--bg-card)',
              borderColor: 'var(--border-default)',
              color:       'var(--text-muted)',
            }}>
            <Database size={11}/> pdm_experiments.db
          </div>
          <button onClick={() => setAutoRefresh(v => !v)}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all"
            style={{
              background:  autoRefresh ? 'var(--tint-success-bg)' : 'var(--bg-card)',
              color:       autoRefresh ? 'var(--success)'         : 'var(--text-tertiary)',
              borderColor: autoRefresh ? 'var(--success)'         : 'var(--border-default)',
            }}>
            {autoRefresh ? '● Auto-refresh' : 'Auto-refresh'}
          </button>
          <button onClick={fetchExperiments}
            className="p-2 rounded-lg border"
            style={{
              background:  'var(--bg-card)',
              borderColor: 'var(--border-default)',
              color:       'var(--text-tertiary)',
            }}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''}/>
          </button>
        </div>
      </div>

      {/* ── InfoBox SQLite ───────────────────────────────────── */}
      <div className="rounded-xl border p-3 flex items-start gap-2"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
        <Database size={13} style={{ color: 'var(--accent-blue)', flexShrink: 0, marginTop: 1 }}/>
        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
          Chaque entraînement est automatiquement sauvegardé dans
          <span style={{ color:'var(--accent-blue)', fontFamily:'monospace' }}> pdm_experiments.db</span>{' '}
          (SQLite). Les résultats persistent entre les sessions — tu retrouves ici toutes tes
          expériences passées, même après redémarrage du serveur. Le modèle
          <span style={{ color:'var(--accent-orange)', fontFamily:'monospace' }}> .keras</span>,
          les scalers <span style={{ color:'var(--accent-orange)', fontFamily:'monospace' }}>.pkl</span>,
          et <code>predictions.csv</code> sont sauvegardés dans
          <span style={{ color:'var(--accent-blue)', fontFamily:'monospace' }}> backend/models/{'{exp_id}'}/</span>.
        </p>
      </div>

      {/* ── Graphe comparatif R² ─────────────────────────────── */}
      {completed.length > 1 && (
        <div className="rounded-xl border p-4"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-3"
             style={{ color: 'var(--text-tertiary)' }}>
            Comparaison R² Score — tous les modèles entraînés
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={completed.slice(0, 15)} margin={{ top: 5, right: 10, bottom: 30, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
              <XAxis dataKey="name" tick={{ fill: 'var(--text-tertiary)', fontSize: 9 }}
                     angle={-25} textAnchor="end"
                     axisLine={{ stroke: 'var(--border-default)' }}/>
              <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-tertiary)', fontSize: 9 }}
                     axisLine={{ stroke: 'var(--border-default)' }}/>
              <Tooltip contentStyle={{
                background:  'var(--bg-elevated)',
                border:      '1px solid var(--border-strong)',
                borderRadius: 8,
                color:       'var(--text-primary)',
              }} formatter={(v) => [v?.toFixed(4), 'R²']}/>
              <Bar dataKey="r2" radius={[4, 4, 0, 0]} maxBarSize={40}>
                {completed.slice(0, 15).map((e, i) => (
                  <Cell key={i} fill={ARCH_COLORS[e.architecture] || 'var(--accent-blue)'}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Tableau ──────────────────────────────────────────── */}
      <div className="rounded-xl border overflow-hidden"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{
                background:    'var(--bg-elevated)',
                borderBottom:  '1px solid var(--border-default)',
              }}>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }}>#</th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }} onClick={() => handleSort('name')}>
                  Nom <SortIcon k="name"/>
                </th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }}>Arch.</th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }}>Mode</th>
                <th className={thCls} style={{ color: 'var(--accent-orange)' }} onClick={() => handleSort('r2')}>
                  R² <SortIcon k="r2"/>
                </th>
                <th className={thCls} style={{ color: 'var(--accent-blue)' }} onClick={() => handleSort('mae')}>
                  MAE (j) <SortIcon k="mae"/>
                </th>
                <th className={thCls} style={{ color: 'var(--accent-green)' }} onClick={() => handleSort('f1_score')}>
                  F1 <SortIcon k="f1_score"/>
                </th>
                <th className={thCls} style={{ color: 'var(--accent-purple)' }} onClick={() => handleSort('mape')}>
                  MAPE <SortIcon k="mape"/>
                </th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }} onClick={() => handleSort('duration_sec')}>
                  Durée <SortIcon k="duration_sec"/>
                </th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }} onClick={() => handleSort('created_at')}>
                  Date <SortIcon k="created_at"/>
                </th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }}>Statut</th>
                <th className={thCls} style={{ color: 'var(--text-tertiary)' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={12} className="px-4 py-10 text-center text-sm"
                      style={{ color: 'var(--text-muted)' }}>
                    Aucune expérience. Lance un entraînement depuis l'onglet
                    <b style={{ color: 'var(--brand-primary)' }}> Entraînement</b> !
                  </td>
                </tr>
              ) : sorted.map((exp, idx) => {
                const isFirst = idx === 0 && exp.status === 'completed';
                const isSel   = selectedId === exp.id;
                return (
                  <tr key={exp.id}
                    className="transition-colors cursor-pointer"
                    style={{
                      borderBottom: '1px solid var(--border-default)',
                      background:   isSel ? 'var(--bg-elevated)' : 'transparent',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-alt)'}
                    onMouseLeave={e => e.currentTarget.style.background = isSel ? 'var(--bg-elevated)' : 'transparent'}
                    onClick={() => setSelectedId(isSel ? null : exp.id)}>
                    <td className="px-3 py-3 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                      {isFirst ? '🥇' : idx + 1}
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {exp.name}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="px-2 py-0.5 rounded text-xs font-mono font-bold"
                        style={{
                          background: `color-mix(in srgb, ${ARCH_COLORS[exp.architecture] || 'var(--text-tertiary)'} 18%, transparent)`,
                          color:      ARCH_COLORS[exp.architecture] || 'var(--text-tertiary)',
                        }}>
                        {exp.architecture}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
                        {exp.mode}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-sm font-bold font-mono"
                        style={{
                          color: exp.r2 == null
                                 ? 'var(--text-muted)'
                                 : (exp.r2 > 0.8
                                     ? 'var(--success)'
                                     : (exp.r2 > 0.5
                                         ? 'var(--accent-orange)'
                                         : 'var(--error)')),
                        }}>
                        {exp.r2 != null ? exp.r2.toFixed(4) : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-sm font-mono" style={{ color: 'var(--accent-blue)' }}>
                        {exp.mae != null ? `${exp.mae.toFixed(2)} j` : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-sm font-mono font-bold" style={{ color: f1Color(exp.f1_score) }}>
                        {exp.f1_score != null ? exp.f1_score.toFixed(3) : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-sm font-mono" style={{ color: mapeColor(exp.mape) }}>
                        {exp.mape != null ? `${exp.mape.toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
                        {exp.duration_sec != null ? `${Math.round(exp.duration_sec)}s` : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                        {exp.created_at
                          ? new Date(exp.created_at).toLocaleDateString('fr-FR', {
                              day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
                            })
                          : '—'}
                      </span>
                    </td>
                    <td className="px-3 py-3"><StatusBadge status={exp.status}/></td>
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-1">
                        <button onClick={e => { e.stopPropagation(); setSelectedId(isSel ? null : exp.id); }}
                          className="p-1.5 rounded transition-colors"
                          style={{ color: 'var(--text-tertiary)' }}
                          title="Voir détails">
                          <Eye size={14}/>
                        </button>
                        <button onClick={e => { e.stopPropagation(); handleDelete(exp.id); }}
                          className="p-1.5 rounded transition-colors"
                          style={{ color: 'var(--text-muted)' }}
                          title="Supprimer">
                          <Trash2 size={14}/>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Card détails (5 onglets) ────────────────────────── */}
      {selected && (
        <ModelDetails
          experiment={selected}
          onClose={() => setSelectedId(null)}
          onDelete={handleDelete}
          onRetrain={handleRetrain}
        />
      )}
    </div>
  );
}
