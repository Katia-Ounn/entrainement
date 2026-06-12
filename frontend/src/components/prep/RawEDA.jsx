/**
 * RawEDA.jsx — Sous-onglet 1 : EDA Brute Phase 1 du notebook.
 *
 *   6 sections accordion exactement conformes au notebook :
 *     Section 1 — Qualité des données       (cell 6)
 *     Section 2 — Distribution niveau hiérarchique (cell 8)
 *     Section 3 — Distribution temporelle    (cell 10)
 *     Section 4 — Distribution par composant (cell 12)
 *     Section 5 — Maintenance / durée / coût (cell 14)
 *     Section 6 — Résumé EDA Phase 1 (PFE)   (cell 16)
 *
 *   Design : accordion Azure-style. Toutes les couleurs via var(--…).
 */
import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, AreaChart, Area, PieChart, Pie, Legend, LineChart, Line, ReferenceLine,
} from 'recharts';
import {
  Play, Loader, AlertCircle, ChevronRight, BarChart3,
  Activity, TrendingUp, Cpu, Wrench, ShieldCheck, FileText,
} from 'lucide-react';
import { useApp } from '../../AppContext';

const API = 'http://localhost:8000';

const MONTHS  = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc'];
const WEEKDAYS = ['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'];

// Palette niveau hiérarchique
const NIVEAU_COLORS = {
  '1.0': 'var(--accent-blue)',
  '2.0': 'var(--accent-green)',
  '3.0': 'var(--accent-orange)',
  '4.0': 'var(--error)',
  'NA':  'var(--text-muted)',
};


export default function RawEDA({ datasetId, onCompleted }) {
  const { edaRawResult, setEdaRawResult, markPrepStep } = useApp();
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const runEDA = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/api/datasets/${datasetId}/eda_raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setEdaRawResult(data);
      markPrepStep('raw_eda');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ─── État initial : pas encore lancé ───
  if (!edaRawResult) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <BarChart3 size={36} style={{ color: 'var(--brand-primary)' }} />
        <h3 className="text-lg font-bold mt-3" style={{ color: 'var(--text-primary)' }}>
          EDA Brute · Phase 1 du pipeline (6 sections)
        </h3>
        <p className="text-sm mt-1 mb-5 text-center max-w-md" style={{ color: 'var(--text-tertiary)' }}>
          Analyse exploratoire sur <code>failure1.csv</code> reproduisant exactement
          les 6 sections du notebook PFE :
          qualité, niveaux, temporel, composants, maintenance, résumé.
        </p>
        <button
          onClick={runEDA} disabled={loading}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background:  loading ? 'var(--bg-card-alt)' : 'var(--brand-primary)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--brand-primary)',
            cursor:      loading ? 'wait' : 'pointer',
          }}>
          {loading
            ? <><Loader size={14} className="animate-spin" /> Analyse en cours…</>
            : <><Play size={14} /> Lancer EDA brute</>}
        </button>

        {error && (
          <div className="mt-4 rounded-lg px-3 py-2 text-xs max-w-xl flex items-start gap-2 border"
            style={{
              background:  'var(--tint-error-bg)',
              color:       'var(--error)',
              borderColor: 'var(--error)',
            }}>
            <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
            <span className="whitespace-pre-wrap font-mono">{error}</span>
          </div>
        )}
      </div>
    );
  }

  // ─── Résultats — 6 accordion sections ───
  const r = edaRawResult;
  const year = r._year_used || r.resume_pfe?.ot_year ? new Date().getFullYear() : null;

  return (
    <div className="space-y-3">
      {/* ─── Overview header ─── */}
      <div className="rounded-xl border p-4"
        style={{
          background:  'color-mix(in srgb, var(--brand-primary) 8%, var(--bg-elevated))',
          borderColor: 'var(--brand-primary)',
        }}>
        <div className="flex items-center gap-2 mb-3">
          <Activity size={14} style={{ color: 'var(--brand-primary)' }}/>
          <p className="text-xs font-semibold uppercase tracking-widest"
             style={{ color: 'var(--text-tertiary)' }}>
            Vue d'ensemble · year={r._year_used || '?'}
          </p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
          <StatCard label="Total OT"           value={r.overview?.total_ot}            icon="📋"/>
          <StatCard label={`OT ${r._year_used ?? ''}`} value={r.overview?.total_ot_year}      icon="📅"/>
          <StatCard label="OT Niveaux 3+4"     value={r.overview?.total_ot_niveaux_34} icon="🔩" highlight/>
          <StatCard label="Composants uniques" value={r.overview?.composants_uniques}  icon="⚙️" highlight/>
          <StatCard
            label={r.overview?.min_failures_seuil != null
              ? `≥${r.overview.min_failures_seuil} pannes (modélisables)`
              : 'Modélisables (≥N pannes)'}
            value={r.overview?.comp_modelisables}
            icon="🎯"
            highlight/>
          <StatCard label="Machines mères"     value={r.overview?.machines_meres}      icon="🏭"/>
        </div>

        {/* 🆕 Explication PÉDAGOGIQUE : pourquoi OT 3+4 ≠ Composants ≠ Modélisables */}
        {r.overview && (
          <div className="mt-3 rounded-lg border p-3 text-xs"
               style={{
                 background:  'var(--bg-card-alt)',
                 borderColor: 'var(--border-default)',
               }}>
            <p className="font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
              💡 Lis les chiffres comme ça :
            </p>
            <div className="space-y-1 font-mono" style={{ color: 'var(--text-tertiary)' }}>
              <p>
                <span style={{ color: 'var(--accent-orange)' }}>🔩 <b>OT Niveaux 3+4</b> = <b>{r.overview.total_ot_niveaux_34?.toLocaleString()}</b></span>
                {' '}→ nombre de <b>lignes</b> (ordres de travail / pannes) sur les composants
                de niveau 3 ou 4. <u>Une ligne = un incident.</u>
              </p>
              <p>
                <span style={{ color: 'var(--brand-primary)' }}>⚙️ <b>Composants uniques</b> = <b>{r.overview.composants_uniques?.toLocaleString()}</b></span>
                {' '}→ nombre de <b>composants physiques distincts</b> qui apparaissent dans
                ces {r.overview.total_ot_niveaux_34?.toLocaleString()} lignes. Un même composant peut tomber
                en panne plusieurs fois.
              </p>
              {r.overview.composants_uniques > 0 && (
                <p style={{ color: 'var(--text-muted)' }}>
                  → Soit en moyenne <b style={{ color: 'var(--accent-blue)' }}>
                    {(r.overview.total_ot_niveaux_34 / r.overview.composants_uniques).toFixed(2)}{' '}
                    pannes par composant
                  </b> ({r.overview.total_ot_niveaux_34} ÷ {r.overview.composants_uniques}).
                </p>
              )}
              <p>
                <span style={{ color: 'var(--success)' }}>🎯 <b>Modélisables</b> = <b>{r.overview.comp_modelisables?.toLocaleString() ?? '?'}</b></span>
                {' '}→ parmi ces <b>{r.overview.composants_uniques?.toLocaleString()}</b> composants,
                seuls ceux qui ont <b>≥ {r.overview.min_failures_seuil ?? '?'} panne(s)</b> sont
                gardés pour entraîner le modèle (un composant avec 1 seule panne ne donne
                pas assez de signal pour apprendre).
              </p>
              <p>
                <span style={{ color: 'var(--accent-purple)' }}>🏭 <b>Machines mères</b> = <b>{r.overview.machines_meres?.toLocaleString()}</b></span>
                {' '}→ nombre de <b>machines de niveau 2</b> auxquelles ces composants
                appartiennent (un composant niveau 3/4 est rattaché à une machine niveau 2).
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ═══════ SECTION 1 — Qualité des données ═══════ */}
      <Accordion title="📊 Section 1 — Qualité des données" icon={ShieldCheck} color="var(--accent-green)">
        <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
          Diagnostic des 8 colonnes essentielles : % manquantes + complétude.
        </p>

        {/* Interprétation (notebook cell 6 print lines) */}
        <div className="rounded-lg p-3 mb-4 text-xs"
          style={{ background: 'var(--bg-card-alt)', borderLeft: '3px solid var(--accent-green)' }}>
          <p className="font-semibold mb-1" style={{ color: 'var(--text-secondary)' }}>Interprétation :</p>
          <p style={{ color: 'var(--error)' }}>🔴 <code>failure_parent_code/level</code> → manquants pour niveaux 1+2 (normal ✅)</p>
          <p style={{ color: 'var(--accent-orange)' }}>🟠 <code>WOWO_JOB_CLASS</code> → manquant pour données 2024 inférées</p>
          <p style={{ color: 'var(--success)' }}>✅ Dates + Coût → complets</p>
        </div>

        {/* % missing par colonne */}
        {r.quality?.missing_pct && (
          <div className="space-y-1.5 mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-widest"
               style={{ color: 'var(--text-tertiary)' }}>
              Taux de valeurs manquantes
            </p>
            {Object.entries(r.quality.missing_pct).map(([col, pct]) => {
              const label = r.cols_utiles_labels?.[col] || col;
              const color = pct === 0 ? 'var(--success)'
                          : pct < 20  ? 'var(--accent-orange)'
                          : pct < 50  ? 'var(--error)'
                          :              'var(--error)';
              const status = pct === 0 ? '✅ Complet' : `${r.quality.missing[col]?.toLocaleString()} (${pct}%)`;
              return (
                <div key={col} className="flex items-center gap-2 text-xs">
                  <span className="w-48 truncate" style={{ color: 'var(--text-secondary)' }}>
                    {label} <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{col}</span>
                  </span>
                  <div className="flex-1 h-3 rounded overflow-hidden"
                       style={{ background: 'var(--bg-card-alt)' }}>
                    <div className="h-full transition-all" style={{
                      background: color, width: `${Math.min(100, pct)}%`,
                    }}/>
                  </div>
                  <span className="text-xs font-mono w-32 text-right"
                        style={{ color, fontWeight: 'bold' }}>{status}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Stacked Renseigné / Manquant */}
        {r.quality?.missing && (
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-widest mb-2"
               style={{ color: 'var(--text-tertiary)' }}>
              Complétude (renseigné vs manquant) — sur {r.quality.total_rows?.toLocaleString()} OT
            </p>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart layout="vertical"
                  data={Object.entries(r.quality.missing).map(([col, miss]) => ({
                    col: r.cols_utiles_labels?.[col] || col,
                    renseigne: r.quality.total_rows - miss,
                    manquant:  miss,
                  }))}
                  margin={{ left: 100 }}
                  stackOffset="expand">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis type="number" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}
                         tickFormatter={v => `${Math.round(v * 100)}%`}/>
                  <YAxis type="category" dataKey="col" width={120}
                         tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6,
                    color: 'var(--text-primary)',
                  }} formatter={(v) => v?.toLocaleString()}/>
                  <Legend wrapperStyle={{ fontSize: 10 }}/>
                  <Bar dataKey="renseigne" stackId="a" fill="var(--success)" name="Renseigné"/>
                  <Bar dataKey="manquant"  stackId="a" fill="var(--error)"   name="Manquant"/>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </Accordion>

      {/* ═══════ SECTION 2 — Distribution niveau hiérarchique ═══════ */}
      <Accordion title="📊 Section 2 — Distribution par niveau hiérarchique" icon={TrendingUp} color="var(--accent-blue)">
        <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
          3 vues : (a) pie tout équipement · (b) zoom niveaux 3+4 (composants modélisés) · (c) niveau 3 vs 4 par mois.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* 2.a Pie tous niveaux */}
          {r.niveau_distribution_all && (
            <ChartBox subtitle="2.a · Répartition OT par niveau">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={[
                    ...Object.entries(r.niveau_distribution_all).map(([k, v]) => ({
                      name: `Niveau ${parseInt(k)}`, value: v,
                    })),
                    ...(r.niveau_na > 0 ? [{ name: 'Non renseigné', value: r.niveau_na }] : []),
                  ]} dataKey="value" cx="50%" cy="50%" outerRadius={70} label={({percent}) => `${(percent * 100).toFixed(0)}%`}>
                    {Object.keys(r.niveau_distribution_all).map((k, i) => (
                      <Cell key={k} fill={NIVEAU_COLORS[k] || NIVEAU_COLORS['NA']}/>
                    ))}
                    {r.niveau_na > 0 && <Cell fill={NIVEAU_COLORS['NA']}/>}
                  </Pie>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <Legend wrapperStyle={{ fontSize: 9 }}/>
                </PieChart>
              </ResponsiveContainer>
            </ChartBox>
          )}

          {/* 2.b Bar niveaux 3+4 */}
          <ChartBox subtitle="2.b · OT niveaux 3+4 (composants modélisés)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={Object.entries(r.niveau_distribution || {})
                .filter(([k]) => ['3.0', '4.0'].includes(k))
                .map(([k, v]) => ({ niveau: `Niveau ${parseInt(k)}`, count: v }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                <XAxis dataKey="niveau" tick={{ fill:'var(--text-tertiary)', fontSize:11 }}/>
                <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:11 }}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius: 6, color: 'var(--text-primary)',
                }}/>
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  <Cell fill="var(--accent-orange)"/>
                  <Cell fill="var(--error)"/>
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartBox>

          {/* 2.c Lines niveau 3 vs 4 par mois */}
          <ChartBox subtitle="2.c · Niveau 3 vs Niveau 4 par mois">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={(r.pannes_mensuel_niv3 || []).map((v, i) => ({
                  mois: MONTHS[i], niv3: v, niv4: (r.pannes_mensuel_niv4 || [])[i] || 0,
                }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                <XAxis dataKey="mois" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius: 6, color: 'var(--text-primary)',
                }}/>
                <Legend wrapperStyle={{ fontSize: 10 }}/>
                <Line type="monotone" dataKey="niv3" stroke="var(--accent-orange)" name="Niveau 3" strokeWidth={2}/>
                <Line type="monotone" dataKey="niv4" stroke="var(--error)" name="Niveau 4" strokeWidth={2}/>
              </LineChart>
            </ResponsiveContainer>
          </ChartBox>
        </div>
      </Accordion>

      {/* ═══════ SECTION 3 — Distribution temporelle ═══════ */}
      <Accordion title="📊 Section 3 — Distribution temporelle des pannes" icon={Activity} color="var(--accent-orange)">
        <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
          4 vues : pannes mensuelles · empilé par niveau · cumulées · jour de semaine.
        </p>

        {/* Stats mois max/min */}
        <div className="rounded-lg p-3 mb-3 grid grid-cols-2 gap-3"
          style={{ background: 'var(--bg-card-alt)' }}>
          <KV k="Mois le plus chargé"
              v={`${MONTHS[r.resume_pfe?.mois_max - 1]} (${r.resume_pfe?.mois_max_n} pannes)`}/>
          <KV k="Mois le plus calme"
              v={`${MONTHS[(r.mois_min ?? 1) - 1]} (${r.mois_min_n ?? 0} pannes)`}/>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* 3.a Pannes par mois */}
          <ChartBox subtitle="3.a · Pannes par mois (niveaux 3+4)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(r.pannes_mensuel || []).map((v, i) => ({ mois: MONTHS[i], pannes: v }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                <XAxis dataKey="mois" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius: 6, color: 'var(--text-primary)',
                }}/>
                <ReferenceLine y={(r.pannes_mensuel || []).reduce((a,b)=>a+b,0) / 12}
                  stroke="var(--text-muted)" strokeDasharray="3 3"
                  label={{ value: 'moy.', fill:'var(--text-muted)', fontSize:9 }}/>
                <Bar dataKey="pannes" radius={[4,4,0,0]} fill="var(--accent-orange)"/>
              </BarChart>
            </ResponsiveContainer>
          </ChartBox>

          {/* 3.b Empilé niveau 3 + niveau 4 */}
          <ChartBox subtitle="3.b · Pannes par mois — niveau 3 + niveau 4 (empilé)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(r.pannes_mensuel_niv3 || []).map((v, i) => ({
                mois: MONTHS[i], niv3: v, niv4: (r.pannes_mensuel_niv4 || [])[i] || 0,
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                <XAxis dataKey="mois" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                <Tooltip contentStyle={{
                  background:'var(--bg-elevated)',
                  border:'1px solid var(--border-strong)',
                  borderRadius: 6, color: 'var(--text-primary)',
                }}/>
                <Legend wrapperStyle={{ fontSize: 10 }}/>
                <Bar dataKey="niv3" stackId="a" fill="var(--accent-orange)" name="Niveau 3"/>
                <Bar dataKey="niv4" stackId="a" fill="var(--error)" name="Niveau 4"/>
              </BarChart>
            </ResponsiveContainer>
          </ChartBox>

          {/* 3.c Cumulées */}
          {r.pannes_cumulees?.dates?.length > 0 && (
            <ChartBox subtitle="3.c · Pannes cumulées dans le temps">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={r.pannes_cumulees.dates.map((d, i) => ({
                    date: d, cumul: r.pannes_cumulees.values[i],
                  }))}>
                  <defs>
                    <linearGradient id="cumulGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="var(--error)" stopOpacity={0.6}/>
                      <stop offset="100%" stopColor="var(--error)" stopOpacity={0.05}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis dataKey="date" tick={{ fill:'var(--text-tertiary)', fontSize:9 }}
                         interval={Math.floor(r.pannes_cumulees.dates.length / 8)}/>
                  <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <Area type="monotone" dataKey="cumul" stroke="var(--error)" strokeWidth={2}
                        fill="url(#cumulGrad)"/>
                </AreaChart>
              </ResponsiveContainer>
            </ChartBox>
          )}

          {/* 3.d Pannes par jour de semaine */}
          {r.pannes_jour_sem && (
            <ChartBox subtitle="3.d · Pannes par jour de la semaine">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={r.pannes_jour_sem.map((v, i) => ({ jour: WEEKDAYS[i], pannes: v }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis dataKey="jour" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <Bar dataKey="pannes" radius={[4,4,0,0]} fill="var(--accent-purple)"/>
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          )}
        </div>
      </Accordion>

      {/* ═══════ SECTION 4 — Distribution par composant ═══════ */}
      <Accordion title="📊 Section 4 — Distribution des pannes par composant" icon={Cpu} color="var(--accent-purple)">
        <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
          3 vues : histogramme · catégories (1 / 2 / 3-4 / ≥5 pannes) · Top 15.
        </p>

        {/* Stats détaillées par catégorie (notebook cell 12 prints) */}
        {r.composants_categories && (
          <div className="rounded-lg p-3 mb-4 text-xs space-y-1 font-mono"
            style={{ background: 'var(--bg-card-alt)', borderLeft: '3px solid var(--accent-purple)' }}>
            <p className="font-semibold mb-1.5" style={{ color: 'var(--text-secondary)', fontFamily: 'inherit' }}>
              Détails par catégorie :
            </p>
            <p style={{ color: 'var(--text-secondary)' }}>
              Total composants (3+4) :
              <b className="ml-1">{r.overview?.composants_uniques}</b>
            </p>
            <p style={{ color: 'var(--error)' }}>
              1 seule panne : <b>{r.composants_categories['1']}</b>
              ({Math.round(r.composants_categories['1'] / r.overview?.composants_uniques * 100)}%)
              ← <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>exclus du modèle</span>
            </p>
            <p style={{ color: 'var(--accent-orange)' }}>
              2 pannes : <b>{r.composants_categories['2']}</b>
              ({Math.round(r.composants_categories['2'] / r.overview?.composants_uniques * 100)}%)
              ← <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>inclus (MIN_FAILURES={r.overview?.min_failures_seuil})</span>
            </p>
            <p style={{ color: 'var(--accent-blue)' }}>
              3-4 pannes : <b>{r.composants_categories['3_4']}</b>
              ({Math.round(r.composants_categories['3_4'] / r.overview?.composants_uniques * 100)}%)
            </p>
            <p style={{ color: 'var(--success)' }}>
              ≥5 pannes : <b>{r.composants_categories['5_plus']}</b>
              ({Math.round(r.composants_categories['5_plus'] / r.overview?.composants_uniques * 100)}%)
              ← <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>les plus fiables</span>
            </p>
            {r.composant_max_pannes && (
              <p style={{ color: 'var(--brand-primary)' }}>
                Max pannes : <b>{r.composant_max_n}</b> ({r.composant_max_pannes})
              </p>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* 4.a Histogramme */}
          {r.composants_hist?.counts?.length > 0 && (
            <ChartBox subtitle={`4.a · Histogramme · médiane=${r.pannes_per_comp_stats?.median?.toFixed(0)} · moy=${r.pannes_per_comp_stats?.mean?.toFixed(1)}`}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={r.composants_hist.bins.slice(0, -1).map((b, i) => ({
                    bin: `${b.toFixed(0)}-${r.composants_hist.bins[i+1].toFixed(0)}`,
                    count: r.composants_hist.counts[i],
                  }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis dataKey="bin" tick={{ fill:'var(--text-tertiary)', fontSize:8 }}/>
                  <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <ReferenceLine x={r.pannes_per_comp_stats?.median?.toString()} stroke="var(--error)" strokeDasharray="3 3"/>
                  <Bar dataKey="count" fill="var(--accent-blue)"/>
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          )}

          {/* 4.b Pie catégories */}
          {r.composants_categories && (() => {
            // 🔎 Données EXACTES depuis le backend (df34.groupby pannes count)
            const c        = r.composants_categories;
            const total    = (c['1'] || 0) + (c['2'] || 0) + (c['3_4'] || 0) + (c['5_plus'] || 0);
            // Modélisables = ceux avec ≥ min_failures pannes (calculé côté pipeline)
            const modelSum = (c['2'] || 0) + (c['3_4'] || 0) + (c['5_plus'] || 0);
            const modelBackend = r.overview?.comp_modelisables;   // référence backend
            // Vérification d'invariant : total === composants_uniques
            const checkOk  = total === (r.overview?.composants_uniques ?? total);
            const modelOk  = modelBackend == null || modelBackend === modelSum;
            const pct = (v) => total > 0 ? ((v / total) * 100).toFixed(1) : '0.0';
            return (
            <ChartBox subtitle="4.b · Catégories selon nb pannes">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={[
                    { name: `1 panne (${c['1']})`,      value: c['1'] || 0 },
                    { name: `2 pannes (${c['2']})`,     value: c['2'] || 0 },
                    { name: `3-4 pannes (${c['3_4']})`, value: c['3_4'] || 0 },
                    { name: `≥5 pannes (${c['5_plus']})`, value: c['5_plus'] || 0 },
                  ]} dataKey="value" cx="50%" cy="50%" outerRadius={70}
                       /* Affiche valeur ABSOLUE + % sur CHAQUE slice (pas juste les grosses) */
                       label={({ value, percent }) =>
                         value > 0 ? `${value} (${(percent * 100).toFixed(0)}%)` : ''}>
                    <Cell fill="var(--error)"/>
                    <Cell fill="var(--accent-orange)"/>
                    <Cell fill="var(--accent-blue)"/>
                    <Cell fill="var(--success)"/>
                  </Pie>
                  <Tooltip
                    formatter={(value, name) => [`${value} composants`, name]}
                    contentStyle={{
                      background:'var(--bg-elevated)',
                      border:'1px solid var(--border-strong)',
                      borderRadius: 6, color: 'var(--text-primary)',
                    }}/>
                  <Legend wrapperStyle={{ fontSize: 9 }}/>
                </PieChart>
              </ResponsiveContainer>

              {/* 🔎 Tableau de contrôle EXACT (le user demande des nombres réels, pas que des %) */}
              <div className="rounded-lg border mt-2 px-2 py-1.5 text-[11px] font-mono"
                style={{
                  background:  'var(--bg-card-alt)',
                  borderColor: checkOk && modelOk ? 'var(--border-default)' : 'var(--error)',
                }}>
                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                  <span style={{ color: 'var(--error)' }}>
                    1 panne : <b>{c['1']}</b> ({pct(c['1'])}%)
                  </span>
                  <span style={{ color: 'var(--accent-orange)' }}>
                    2 pannes : <b>{c['2']}</b> ({pct(c['2'])}%)
                  </span>
                  <span style={{ color: 'var(--accent-blue)' }}>
                    3-4 pannes : <b>{c['3_4']}</b> ({pct(c['3_4'])}%)
                  </span>
                  <span style={{ color: 'var(--success)' }}>
                    ≥5 pannes : <b>{c['5_plus']}</b> ({pct(c['5_plus'])}%)
                  </span>
                </div>
                <div className="mt-1.5 pt-1.5 border-t"
                     style={{ borderColor: 'var(--border-subtle)' }}>
                  <p style={{ color: checkOk ? 'var(--text-secondary)' : 'var(--error)' }}>
                    Σ catégories = <b>{total}</b>
                    {' '}{checkOk ? '✓' : '✗'} (composants_uniques = {r.overview?.composants_uniques})
                  </p>
                  <p style={{ color: modelOk ? 'var(--text-secondary)' : 'var(--error)' }}>
                    🎯 Modélisables (≥{r.overview?.min_failures_seuil}) =
                    {' '}2+3-4+≥5 = <b>{modelSum}</b>
                    {modelBackend != null && modelBackend !== modelSum && (
                      <span style={{ color: 'var(--error)' }}>
                        {' '}⚠ backend={modelBackend}
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </ChartBox>
            );
          })()}

          {/* 4.c Top 15 */}
          {r.top_composants && Object.keys(r.top_composants).length > 0 && (
            <ChartBox subtitle="4.c · Top 15 composants">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart layout="vertical"
                  data={Object.entries(r.top_composants).slice(0, 15).map(([n, v]) => ({ comp: n, n: v }))}
                  margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis type="number" tick={{ fill:'var(--text-tertiary)', fontSize:9 }}/>
                  <YAxis type="category" dataKey="comp" width={130}
                         tick={{ fill:'var(--text-tertiary)', fontSize:8 }}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <Bar dataKey="n" radius={[0,3,3,0]} fill="var(--error)"/>
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          )}
        </div>
      </Accordion>

      {/* ═══════ SECTION 5 — Maintenance + durée + coût ═══════ */}
      <Accordion title="📊 Section 5 — Maintenance / durée / coût" icon={Wrench} color="var(--accent-green)">
        <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
          3 vues : type maintenance (pie) · stats durée · coût par mois (barres).
        </p>

        {/* Stats du notebook cell 14 print lines */}
        <div className="rounded-lg p-3 mb-4 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs"
          style={{ background: 'var(--bg-card-alt)', borderLeft: '3px solid var(--accent-green)' }}>
          <p style={{ color: 'var(--text-secondary)' }}>
            <b>Type dominant</b> :
            <span className="font-mono ml-1">{r.resume_pfe?.type_dominant} ({r.resume_pfe?.type_dominant_n} OT — {Math.round(r.resume_pfe?.type_dominant_n / r.overview?.total_ot_niveaux_34 * 100)}%)</span>
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <b>Préventif (PREVEN)</b> :
            <span className="font-mono ml-1">{r.preventif_n ?? 0} OT ({r.preventif_pct ?? 0}%)</span>
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <b>Réparés le jour même</b> :
            <span className="font-mono ml-1">{r.resume_pfe?.meme_jour_pct}%</span>
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <b>Durée médiane</b> :
            <span className="font-mono ml-1">{r.duree_stats?.median?.toFixed(0) ?? '—'} jours</span>
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <b>Coût total {r._year_used ?? ''}</b> :
            <span className="font-mono ml-1">{r.resume_pfe?.cout_total_MDA} M DA</span>
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* 5.a Pie types */}
          {r.job_class_dist && Object.keys(r.job_class_dist).length > 0 && (
            <ChartBox subtitle="5.a · Type de maintenance">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={Object.entries(r.job_class_dist).map(([k, v]) => ({ name: k, value: v }))}
                       dataKey="value" cx="50%" cy="50%" outerRadius={70} label>
                    {Object.keys(r.job_class_dist).map((_, i) => (
                      <Cell key={i} fill={[
                        'var(--accent-blue)', 'var(--accent-green)',
                        'var(--accent-orange)', 'var(--accent-purple)',
                        'var(--accent-pink)',
                      ][i % 5]}/>
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }}/>
                  <Legend wrapperStyle={{ fontSize: 9 }}/>
                </PieChart>
              </ResponsiveContainer>
            </ChartBox>
          )}

          {/* 5.b Stats durée */}
          {r.duree_stats?.count > 0 && (
            <ChartBox subtitle="5.b · Statistiques durée réparation">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <KV k="Moyenne" v={`${r.duree_stats.mean?.toFixed(1)} j`}/>
                <KV k="Médiane" v={`${r.duree_stats.median?.toFixed(0)} j`}/>
                <KV k="Max"     v={`${r.duree_stats.max?.toFixed(0)} j`}/>
                <KV k="P95"     v={`${r.duree_stats.p95?.toFixed(0)} j`}/>
                <KV k="N"       v={r.duree_stats.count?.toLocaleString()}/>
                <KV k="Coût tot." v={`${(r.cout_total_year / 1e6).toFixed(1)} M DA`}/>
              </div>
            </ChartBox>
          )}

          {/* 5.c Coût par mois */}
          {r.cout_par_mois && (
            <ChartBox subtitle="5.c · Coût total par mois (M DA)">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={r.cout_par_mois.map((v, i) => ({
                  mois: MONTHS[i], cout: Math.round(v / 1e6 * 100) / 100,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                  <XAxis dataKey="mois" tick={{ fill:'var(--text-tertiary)', fontSize:10 }}/>
                  <YAxis tick={{ fill:'var(--text-tertiary)', fontSize:10 }}
                         tickFormatter={v => `${v}M`}/>
                  <Tooltip contentStyle={{
                    background:'var(--bg-elevated)',
                    border:'1px solid var(--border-strong)',
                    borderRadius: 6, color: 'var(--text-primary)',
                  }} formatter={(v) => `${v} M DA`}/>
                  <Bar dataKey="cout" radius={[4,4,0,0]} fill="var(--success)"/>
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          )}
        </div>
      </Accordion>

      {/* ═══════ SECTION 6 — Résumé final PFE ═══════ */}
      {r.resume_pfe && (
        <Accordion title="📋 Section 6 — Résumé EDA Phase 1 (rapport PFE)" icon={FileText} color="var(--brand-primary)" defaultOpen={true}>
          <p className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
            Synthèse des chiffres-clés pour le rapport — à copier-coller tel quel.
          </p>
          <div className="rounded-lg p-4 font-mono text-sm"
            style={{
              background:  'var(--bg-card-alt)',
              border:      '1px solid var(--brand-primary)',
              color:       'var(--text-primary)',
            }}>
            <ResumeRow k="OT totaux (année)"             v={r.resume_pfe.ot_year?.toLocaleString()}/>
            <ResumeRow k="OT niveaux 3+4 modélisés"      v={r.resume_pfe.ot_niveaux_34?.toLocaleString()}/>
            <ResumeRow k="Composants uniques (3+4)"      v={r.resume_pfe.composants_uniques}/>
            <ResumeRow k={`Composants modélisables (≥${r.overview?.min_failures_seuil})`} v={`${r.resume_pfe.comp_modelisables}  ← MIN_FAILURES=${r.overview?.min_failures_seuil}`} highlight/>
            <ResumeRow k="Machines mères"                v={r.resume_pfe.machines_meres}/>
            <ResumeRow k="Mois le plus chargé"           v={`${MONTHS[r.resume_pfe.mois_max - 1]} (${r.resume_pfe.mois_max_n} pannes)`}/>
            <ResumeRow k="Type dominant"                 v={`${r.resume_pfe.type_dominant} (${r.resume_pfe.type_dominant_n} OT)`}/>
            <ResumeRow k="Coût total année (3+4)"        v={`${r.resume_pfe.cout_total_MDA} M DA`}/>
            <ResumeRow k="Réparés le jour même"          v={`${r.resume_pfe.meme_jour_pct}%`}/>
          </div>
        </Accordion>
      )}

      {/* CTA continuer */}
      <div className="flex justify-end pt-3 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
        <button onClick={onCompleted}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background:  'var(--success)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--success)',
          }}>
          ✓ Validé → Feature Engineering <ChevronRight size={14}/>
        </button>
      </div>
    </div>
  );
}


// ─── Helpers ──────────────────────────────────────────────
function Accordion({ title, icon: Icon, color, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border overflow-hidden" style={{
      background:  'var(--bg-card)',
      borderColor: open ? color : 'var(--border-default)',
    }}>
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 transition-all"
        style={{ background: open ? 'var(--bg-elevated)' : 'transparent' }}>
        <div className="flex items-center gap-2">
          {Icon && <Icon size={15} style={{ color }}/>}
          <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{title}</p>
        </div>
        <span style={{ color: open ? color : 'var(--text-muted)' }}>{open ? '▴' : '▾'}</span>
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


function ChartBox({ subtitle, children }) {
  return (
    <div className="rounded-lg border p-3"
      style={{ background: 'var(--bg-card-alt)', borderColor: 'var(--border-default)' }}>
      {subtitle && (
        <p className="text-[11px] font-semibold mb-2"
           style={{ color: 'var(--text-tertiary)' }}>{subtitle}</p>
      )}
      {children}
    </div>
  );
}


function StatCard({ label, value, icon, highlight }) {
  return (
    <div className="rounded-lg p-3 border"
      style={{
        background:  'var(--bg-card)',
        borderColor: highlight ? 'var(--brand-primary)' : 'var(--border-default)',
      }}>
      <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>
        <span>{icon}</span> {label}
      </p>
      <p className="text-xl font-bold font-mono"
         style={{ color: highlight ? 'var(--brand-primary)' : 'var(--text-primary)' }}>
        {value == null ? '—' : Number(value).toLocaleString()}
      </p>
    </div>
  );
}


function KV({ k, v }) {
  return (
    <div className="flex items-baseline justify-between rounded px-2 py-1.5"
      style={{ background: 'var(--bg-elevated)' }}>
      <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>{k}</span>
      <span className="text-sm font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{v}</span>
    </div>
  );
}


function ResumeRow({ k, v, highlight }) {
  return (
    <div className="flex items-baseline justify-between py-0.5"
         style={{ color: highlight ? 'var(--brand-primary)' : 'var(--text-secondary)' }}>
      <span className="text-xs">{k}</span>
      <span className="text-sm font-bold">{v}</span>
    </div>
  );
}
