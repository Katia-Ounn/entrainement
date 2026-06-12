/**
 * FeatureEngineering.jsx — Sous-onglet 2 (refonte Lot C : design Azure-style).
 *
 *   - Accordion en sections : chaque étape du pipeline est dépliable
 *   - Grand tableau Dataset_V1 avec TOUTES les 23 colonnes (MTBF inclus)
 *   - Row selector 5/10/20/50 lignes
 *   - 🆕 Feature toggle UI : user choisit quelles features partent au modèle
 *
 *   Backend route : POST /api/datasets/{id}/features (inchangé)
 *   Les features sélectionnées sont stockées dans AppContext.selectedFeatures
 *   et envoyées au backend au moment du Preprocessing.
 */
import { useState, useMemo } from 'react';
import {
  Play, Loader, AlertCircle, ChevronRight, Cog, Download,
  CheckCircle2, ChevronDown, ChevronUp, Sliders, Database,
  Layers, FileText, Filter, Info,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { useApp } from '../../AppContext';

const API = 'http://localhost:8000';

// Les 8 étapes du pipeline (correspondant aux étapes notebook Phase 2)
// ⚠️ Ordre respecte l'enchaînement notebook : panel + RUL AVANT rolling features
const ETAPES = [
  { n: 1, id: 'filter_year',  label: 'Filtrage année',                       desc: 'On garde uniquement les pannes de l\'année courante' },
  { n: 2, id: 'hierarchy',    label: 'Hiérarchie machineID',                 desc: 'On remonte la hiérarchie pour identifier la machine mère (niveau 2) de chaque composant' },
  { n: 3, id: 'select_comps', label: 'Sélection composants modélisables',    desc: 'On garde uniquement les composants ayant ≥ min_failures pannes' },
  { n: 4, id: 'lookup',       label: 'Lookup maintenance',                   desc: 'On indexe les dates de maintenance par composant' },
  { n: 5, id: 'panel',        label: 'Panel composant × jour',               desc: 'Une ligne par (composant, jour) avec failure/maintenance binaires' },
  { n: 6, id: 'rul',          label: 'Calcul du RUL',                        desc: 'RUL = jours jusqu\'à la prochaine panne (sinon fin d\'année)' },
  { n: 7, id: 'rolling',      label: 'Features rolling (pannes_*, DSLF, …)', desc: 'pannes_7j/30j/90j, maint_*, DSLF, DSLM, MTBF, saisonnalité' },
  { n: 8, id: 'export',       label: 'Export Dataset_V1.csv',                desc: '23 colonnes finales prêtes pour le prétraitement' },
];

// Catégorisation des colonnes pour le toggle UI
const FEATURE_CATEGORIES = [
  {
    title: '🆔 Identifiants & métadonnées',
    color: 'var(--text-muted)',
    note:  'Non-features. Conservés pour traçabilité.',
    fixed: true,
    cols:  ['date', 'machineID', 'machineID_num', 'machineID_level',
            'comp_num', 'failure_comp'],
  },
  {
    title: '🎯 Cible & événements (binaires)',
    color: 'var(--accent-orange)',
    note:  'failure/maintenance = sources du calcul. RUL = cible du modèle.',
    fixed: true,
    cols:  ['failure', 'maintenance', 'RUL'],
  },
  {
    title: '🔢 Niveau composant',
    color: 'var(--accent-purple)',
    note:  'Niveau hiérarchique (3 ou 4). Notebook l\'utilise.',
    cols:  ['comp_level'],
  },
  {
    title: '📊 Fenêtres roulantes',
    color: 'var(--accent-blue)',
    note:  'Nombre de pannes/maintenances dans les N derniers jours. 6 features → notebook utilise les 6.',
    cols:  ['pannes_7j', 'pannes_30j', 'pannes_90j', 'maint_7j', 'maint_30j', 'maint_90j'],
  },
  {
    title: '⏱ Temps depuis dernier événement',
    color: 'var(--accent-green)',
    note:  'DSLF = Days Since Last Failure · DSLM = Days Since Last Maintenance.',
    cols:  ['DSLF', 'DSLM'],
  },
  {
    title: '📈 MTBF / saisonnalité (avancé)',
    color: 'var(--accent-purple)',
    note:  '⚠️ Non utilisés par le notebook par défaut. Tu peux les activer pour expérimenter.',
    cols:  ['MTBF_rolling', 'has_mtbf', 'month_sin', 'month_cos', 'dslf_mtbf_ratio'],
  },
];


export default function FeatureEngineering({ datasetId, onCompleted }) {
  const { featuresResult, setFeaturesResult, markPrepStep,
          selectedFeatures, setSelectedFeatures, DEFAULT_FEATURES,
          bumpDatasetVersion } = useApp();
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  // États accordion : par défaut tout fermé sauf la section 1 + tableau V1
  const [openSections, setOpenSections] = useState(() => new Set(['stats', 'preview_v1', 'features_toggle']));
  const toggleSection = (id) => setOpenSections(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  // Row selector
  const [previewRows, setPreviewRows] = useState(10);

  // 🆕 Statut par étape pour affichage progressif
  const [stepStatus, setStepStatus] = useState({});
  const [stepData,   setStepData]   = useState({});   // dernières data reçues par step_id

  // ⚠️ Retour au endpoint SYNCHRONE /features — plus fiable.
  // Le streaming /features_stream existe toujours en backend mais le browser
  // a parfois "network error" sur les très longs streams (>60s). On simule
  // la progression visuelle côté client avec un timer.
  const runFeatures = async () => {
    setLoading(true); setError(null);
    // Initialise le stepper : 1ère étape running, reste pending
    setStepStatus(Object.fromEntries(ETAPES.map((s, i) => [s.id, i === 0 ? 'running' : 'pending'])));
    setStepData({});

    // ⏱ Animation côté client : on avance les étapes une par une toutes les ~3s
    // pour donner du visuel pendant que le backend bosse. Si le backend finit
    // avant que toutes les étapes soient cochées, on les complète d'un coup.
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
      stepIdx += 1;
      if (stepIdx >= ETAPES.length) return clearInterval(stepTimer);
      setStepStatus(prev => {
        const next = { ...prev };
        if (stepIdx - 1 < ETAPES.length) next[ETAPES[stepIdx - 1].id] = 'done';
        if (stepIdx     < ETAPES.length) next[ETAPES[stepIdx].id]     = 'running';
        return next;
      });
    }, 3500);

    try {
      const res = await fetch(`${API}/api/datasets/${datasetId}/features`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      clearInterval(stepTimer);

      // Toutes les étapes done
      setStepStatus(Object.fromEntries(ETAPES.map(s => [s.id, 'done'])));
      setFeaturesResult(data);
      markPrepStep('features');
      bumpDatasetVersion();
      toast.success(`Dataset_V1 généré · ${data.n_rows?.toLocaleString() || '?'} lignes`);
    } catch (e) {
      clearInterval(stepTimer);
      setError(e.message);
      toast.error(`Feature Engineering échoué : ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadV1 = () => {
    window.open(`${API}/api/datasets/${datasetId}/download_v1`, '_blank');
  };

  // Reset features sélectionnées aux defaults notebook
  const resetFeatures = () => setSelectedFeatures(DEFAULT_FEATURES);

  // Toggle une feature
  const toggleFeature = (col) => {
    setSelectedFeatures(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  // ─── État initial : pas encore lancé ───
  if (!featuresResult && !loading) {
    return (
      <div className="flex flex-col items-center justify-center py-10">
        <Cog size={36} style={{ color: 'var(--brand-primary)' }} />
        <h3 className="text-lg font-bold mt-3" style={{ color: 'var(--text-primary)' }}>
          Feature Engineering · 8 étapes
        </h3>
        <p className="text-sm mt-1 mb-4 max-w-xl text-center" style={{ color: 'var(--text-tertiary)' }}>
          Construit le <b>Dataset_V1</b> : panel composant × jour + RUL + 23 colonnes
          dont les 9 features modèle (pannes_7j/30j/90j, maint_7j/30j/90j, DSLF, DSLM, comp_level).
        </p>

        <div className="rounded-xl border px-4 py-3 mb-4 w-full max-w-xl"
          style={{
            background:  'var(--bg-elevated)',
            borderColor: 'var(--border-default)',
          }}>
          {ETAPES.map((e) => (
            <div key={e.n} className="flex items-start gap-2 text-xs py-1 font-mono"
                 style={{ color: 'var(--text-tertiary)' }}>
              <span className="font-bold" style={{ color: 'var(--brand-primary)' }}>{e.n}.</span>
              <span><b>{e.label}</b> — <span style={{ color: 'var(--text-muted)' }}>{e.desc}</span></span>
            </div>
          ))}
        </div>

        <button onClick={runFeatures}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background:  'var(--brand-primary)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--brand-primary)',
          }}>
          <Play size={14} /> Lancer Feature Engineering
        </button>

        {error && (
          <div className="mt-4 rounded-lg px-3 py-2 text-xs max-w-2xl border"
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

  // ─── Pendant calcul : stepper progressif ───
  if (loading) {
    return (
      <div className="py-4">
        <div className="flex items-center gap-3 mb-3">
          <Loader size={20} className="animate-spin" style={{ color: 'var(--brand-primary)' }} />
          <h3 className="text-base font-bold" style={{ color: 'var(--text-primary)' }}>
            Pipeline en cours · {ETAPES.filter(s => stepStatus[s.id] === 'done').length} / 8 étapes
          </h3>
        </div>
        <div className="space-y-2">
          {ETAPES.map((s) => {
            const st = stepStatus[s.id] || 'pending';
            const data = stepData[s.id];
            const cfg = {
              pending: { bg: 'var(--bg-card-alt)',    color: 'var(--text-muted)',   icon: '○' },
              running: { bg: 'var(--tint-info-bg)',   color: 'var(--accent-blue)',  icon: '⟳' },
              done:    { bg: 'var(--tint-success-bg)',color: 'var(--success)',      icon: '✓' },
              error:   { bg: 'var(--tint-error-bg)',  color: 'var(--error)',        icon: '✗' },
            }[st];
            return (
              <div key={s.id} className="rounded-lg border p-3"
                style={{ background: cfg.bg, borderColor: cfg.color }}>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-bold ${st === 'running' ? 'animate-spin inline-block' : ''}`}
                        style={{ color: cfg.color }}>{cfg.icon}</span>
                  <span className="font-mono font-bold text-sm" style={{ color: cfg.color }}>
                    {s.n}.
                  </span>
                  <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {s.label}
                  </span>
                </div>
                <p className="text-[11px] mt-1 ml-6" style={{ color: 'var(--text-muted)' }}>
                  {s.desc}
                </p>
                {/* Détails diagnostiques par étape (apparaissent quand done) */}
                {st === 'done' && data && (
                  <div className="ml-6 mt-2 grid grid-cols-2 md:grid-cols-3 gap-2">
                    <StepDataPanel stepId={s.id} data={data} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {error && (
          <div className="mt-3 rounded-lg px-3 py-2 text-xs border"
            style={{
              background:  'var(--tint-error-bg)',
              color:       'var(--error)',
              borderColor: 'var(--error)',
            }}>
            <AlertCircle size={14} className="inline mr-1.5"/>
            <span className="whitespace-pre-wrap font-mono">{error}</span>
          </div>
        )}
      </div>
    );
  }

  // ─── Résultats avec accordion ───
  const r = featuresResult;
  const allColsInV1 = r.preview_final?.[0] ? Object.keys(r.preview_final[0]) : [];

  return (
    <div className="space-y-3">
      {/* ─── 1. Stats globales (toujours ouvert) ─── */}
      <Accordion id="stats" open={openSections.has('stats')} onToggle={toggleSection}
        icon={Database} color="var(--brand-primary)"
        title="📊 Stats globales du Dataset_V1"
        subtitle="Résumé après les 8 étapes du pipeline">

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="🧱 Lignes"       value={r.n_rows} highlight />
          <StatCard label="💥 Pannes"       value={r.n_failures} />
          <StatCard label="🔧 Maintenances" value={r.n_maintenances} />
          <StatCard label="🔩 Composants"   value={r.n_composants} highlight />
          <StatCard label="⚙️ Machines"     value={r.n_machines} />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-4 text-xs font-mono"
             style={{ color: 'var(--text-tertiary)' }}>
          <span>📅 {r.period_start?.slice(0,10)} → {r.period_end?.slice(0,10)}</span>
          {r.rul_stats && (
            <span>
              RUL : min={r.rul_stats.min}j · max={r.rul_stats.max}j ·
              moy={Number(r.rul_stats.mean).toFixed(1)}j · médiane={r.rul_stats.median}j
            </span>
          )}
        </div>

        {/* 🆕 Diagnostic filtrage (Distribution niveaux + OT totaux/3+4 + composants) */}
        {(r.level_distribution_all || r.n_ot_year != null) && (
          <div className="mt-4 rounded-lg border p-3"
               style={{ background: 'var(--bg-card-alt)', borderColor: 'var(--border-default)' }}>
            <p className="text-xs font-semibold mb-2"
               style={{ color: 'var(--text-secondary)' }}>
              🔎 Diagnostic filtrage (notebook PFE) — year={r.year} · MIN_FAILURES={r.min_failures}
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-1.5 text-[11px] font-mono mb-2">
              <span style={{ color: 'var(--text-tertiary)' }}>
                OT totaux {r.year} : <b style={{ color: 'var(--text-primary)' }}>
                  {Number(r.n_ot_year || 0).toLocaleString()}
                </b>
              </span>
              <span style={{ color: 'var(--text-tertiary)' }}>
                OT niveaux 3+4 modélisés : <b style={{ color: 'var(--accent-orange)' }}>
                  {Number(r.n_ot_year_levels_34 || 0).toLocaleString()}
                </b>
              </span>
              <span style={{ color: 'var(--text-tertiary)' }}>
                Composants uniques (3+4) : <b style={{ color: 'var(--text-primary)' }}>
                  {Number(r.n_composants_uniques_34 || 0).toLocaleString()}
                </b>
              </span>
              <span style={{ color: 'var(--text-tertiary)' }}>
                Composants modélisables (≥{r.min_failures}) : <b style={{ color: 'var(--success)' }}>
                  {Number(r.n_composants_modelisables || 0).toLocaleString()}
                </b>
              </span>
              <span style={{ color: 'var(--text-tertiary)' }}>
                Machines mères : <b style={{ color: 'var(--accent-blue)' }}>
                  {Number(r.n_machines_meres || 0).toLocaleString()}
                </b>
              </span>
            </div>

            {r.level_distribution_all && Object.keys(r.level_distribution_all).length > 0 && (
              <>
                <p className="text-[10px] mt-2 mb-1" style={{ color: 'var(--text-tertiary)' }}>
                  Distribution WOWO_EQUIPMENT_LEVEL :
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(r.level_distribution_all).map(([lvl, n]) => (
                    <span key={lvl}
                      className="px-2 py-0.5 rounded text-[11px] font-mono border"
                      style={{
                        background:  'var(--bg-elevated)',
                        borderColor: ['3','4'].includes(lvl) ? 'var(--accent-orange)' : 'var(--border-default)',
                        color:       ['3','4'].includes(lvl) ? 'var(--accent-orange)' : 'var(--text-secondary)',
                      }}>
                      niveau {lvl} : <b>{Number(n).toLocaleString()}</b>
                    </span>
                  ))}
                </div>
              </>
            )}

            {/* 🆕 Nombre de pannes par seuil (le user veut les valeurs absolues, pas les %) */}
            {(r.n_rul_le_5 != null || r.n_rul_le_30 != null) && (
              <>
                <p className="text-[10px] mt-3 mb-1" style={{ color: 'var(--text-tertiary)' }}>
                  Nombre de jours par seuil RUL :
                </p>
                <div className="flex flex-wrap gap-1.5">
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                        style={{ borderColor: 'var(--error)', color: 'var(--error)' }}>
                    RUL ≤ 5j : <b>{Number(r.n_rul_le_5 || 0).toLocaleString()}</b>
                  </span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                        style={{ borderColor: 'var(--accent-orange)', color: 'var(--accent-orange)' }}>
                    RUL ≤ 10j : <b>{Number(r.n_rul_le_10 || 0).toLocaleString()}</b>
                  </span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                        style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}>
                    RUL ≤ 30j : <b>{Number(r.n_rul_le_30 || 0).toLocaleString()}</b>
                  </span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                        style={{ borderColor: 'var(--success)', color: 'var(--success)' }}>
                    RUL ≤ 90j : <b>{Number(r.n_rul_le_90 || 0).toLocaleString()}</b>
                  </span>
                </div>
              </>
            )}
          </div>
        )}
      </Accordion>

      {/* ─── 2. Aperçu panel (étape 5) ─── */}
      {r.preview_panel?.length > 0 && (
        <Accordion id="preview_panel" open={openSections.has('preview_panel')} onToggle={toggleSection}
          icon={Layers} color="var(--accent-blue)"
          title="🏗️ Aperçu panel composant × jour (Étape 5)"
          subtitle="Une ligne = (composant, jour) — avant les features dérivées">
          <PreviewTable rows={r.preview_panel.slice(0, previewRows)}
                        rowSelector={previewRows}
                        onRowsChange={setPreviewRows}/>
        </Accordion>
      )}

      {/* ─── 3. Grand tableau Dataset_V1 (toutes les 23 cols) ─── */}
      {r.preview_final?.length > 0 && (
        <Accordion id="preview_v1" open={openSections.has('preview_v1')} onToggle={toggleSection}
          icon={FileText} color="var(--accent-purple)"
          title="📑 Aperçu Dataset_V1 — toutes les 23 colonnes"
          subtitle={`Tu peux voir ici TOUTES les features disponibles (MTBF, has_mtbf, dslf_mtbf_ratio inclus). Plus loin, tu choisis lesquelles partent au modèle.`}>
          <PreviewTable rows={r.preview_final.slice(0, previewRows)}
                        rowSelector={previewRows}
                        onRowsChange={setPreviewRows}/>
        </Accordion>
      )}

      {/* ─── 4. Feature Toggle UI 🆕 ─── */}
      <Accordion id="features_toggle" open={openSections.has('features_toggle')} onToggle={toggleSection}
        icon={Filter} color="var(--accent-orange)"
        title="🎛️ Sélection des features pour le modèle"
        subtitle={`Coche/décoche les features que le LSTM/GRU utilisera. ${selectedFeatures.length} sélectionnées sur ${allColsInV1.filter(c => !['date','machineID','machineID_num','machineID_level','comp_num','failure_comp','comp_level','failure','maintenance','RUL'].includes(c)).length} disponibles.`}>

        <div className="flex items-center justify-between mb-3 px-3 py-2 rounded-lg border"
          style={{
            background:  'var(--bg-card-alt)',
            borderColor: 'var(--border-default)',
          }}>
          <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
            <b style={{ color: 'var(--accent-orange)' }}>💡 Recommandation</b> : laisse les
            <b> 9 features par défaut du notebook</b> (catégories 📊 + ⏱) pour reproduire
            les résultats du PFE. MTBF & saisonnalité = pour expérimentation avancée.
          </p>
          <button onClick={resetFeatures}
            className="px-3 py-1 rounded text-xs font-semibold border whitespace-nowrap ml-3"
            style={{
              background:  'var(--bg-elevated)',
              borderColor: 'var(--brand-primary)',
              color:       'var(--brand-primary)',
            }}>
            ↺ Reset notebook
          </button>
        </div>

        <div className="space-y-3">
          {FEATURE_CATEGORIES.map(cat => (
            <div key={cat.title} className="rounded-lg border p-3"
              style={{
                background:  'var(--bg-card)',
                borderColor: cat.color,
                opacity:     cat.fixed ? 0.7 : 1,
              }}>
              <div className="flex items-start justify-between mb-2 gap-2">
                <div>
                  <p className="text-xs font-semibold" style={{ color: cat.color }}>{cat.title}</p>
                  <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{cat.note}</p>
                </div>
                {!cat.fixed && (
                  <button
                    onClick={() => {
                      // Toggle all in this category
                      const allSelected = cat.cols.every(c => selectedFeatures.includes(c));
                      if (allSelected) {
                        setSelectedFeatures(prev => prev.filter(c => !cat.cols.includes(c)));
                      } else {
                        setSelectedFeatures(prev => [...new Set([...prev, ...cat.cols])]);
                      }
                    }}
                    className="text-[11px] px-2 py-0.5 rounded border whitespace-nowrap"
                    style={{
                      background:  'var(--bg-elevated)',
                      borderColor: cat.color,
                      color:       cat.color,
                    }}>
                    Tout cocher/décocher
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {cat.cols.map(col => {
                  const isFixed    = cat.fixed;
                  const isSelected = isFixed || selectedFeatures.includes(col);
                  const inV1       = allColsInV1.includes(col);
                  return (
                    <button key={col}
                      onClick={() => { if (!isFixed) toggleFeature(col); }}
                      disabled={isFixed || !inV1}
                      title={!inV1 ? 'Cette colonne n\'est pas dans Dataset_V1' : ''}
                      className="px-2.5 py-1 rounded text-xs font-mono border transition-all flex items-center gap-1"
                      style={{
                        background:  isSelected ? `color-mix(in srgb, ${cat.color} 20%, var(--bg-elevated))` : 'var(--bg-elevated)',
                        borderColor: isSelected ? cat.color : 'var(--border-default)',
                        color:       !inV1 ? 'var(--text-muted)' : (isSelected ? cat.color : 'var(--text-secondary)'),
                        cursor:      isFixed ? 'not-allowed' : 'pointer',
                        opacity:     !inV1 ? 0.4 : 1,
                      }}>
                      {isSelected && !isFixed && <span style={{ color: cat.color }}>✓</span>}
                      {isFixed && <span style={{ color: cat.color }}>📌</span>}
                      {col}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-3 rounded-lg p-3 flex items-start gap-2"
          style={{ background: 'var(--tint-info-bg)' }}>
          <Info size={14} style={{ color: 'var(--accent-blue)', flexShrink: 0, marginTop: 1 }}/>
          <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            Les <b>{selectedFeatures.length} features cochées</b> seront envoyées au
            backend lors du <b>Prétraitement</b> (étape suivante). L'architecture LSTM/GRU
            s'adaptera automatiquement à <code>n_features = {selectedFeatures.length}</code>.
          </p>
        </div>
      </Accordion>

      {/* ─── 5. Features modèle actuellement sélectionnées ─── */}
      <Accordion id="selected_summary" open={openSections.has('selected_summary')} onToggle={toggleSection}
        icon={Sliders} color="var(--success)"
        title={`✅ Features sélectionnées pour le modèle (${selectedFeatures.length})`}
        subtitle="Liste ordonnée — exactement ce qui partira au LSTM/GRU">
        <div className="flex flex-wrap gap-2">
          {selectedFeatures.map((f, i) => (
            <span key={f}
              className="px-2 py-1 rounded text-xs font-mono border"
              style={{
                background:  'color-mix(in srgb, var(--success) 15%, var(--bg-elevated))',
                borderColor: 'var(--success)',
                color:       'var(--success)',
              }}>
              {i + 1}. {f}
            </span>
          ))}
          <span className="px-2 py-1 rounded text-xs font-mono border"
            style={{
              background:  'var(--bg-elevated)',
              borderColor: 'var(--accent-orange)',
              color:       'var(--accent-orange)',
            }}>
            🎯 target = RUL
          </span>
        </div>
      </Accordion>

      {/* ─── Actions ─── */}
      <div className="flex flex-col sm:flex-row justify-between items-stretch sm:items-center gap-3 pt-3 border-t"
           style={{ borderColor: 'var(--border-subtle)' }}>
        <button onClick={downloadV1}
          className="px-4 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 border"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--accent-blue)',
            color:       'var(--accent-blue)',
          }}>
          <Download size={14} /> Télécharger Dataset_V1.csv
        </button>
        <button onClick={onCompleted}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2"
          style={{
            background:  'var(--success)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--success)',
          }}>
          <CheckCircle2 size={14} /> Continuer vers EDA Features <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}


// ─── Helpers ───────────────────────────────────────────────────

// Affiche les diagnostics d'une étape (compacts, ml-6)
function StepDataPanel({ stepId, data }) {
  if (!data) return null;

  const Mini = ({ label, value, color }) => (
    <div className="rounded px-2 py-1 text-[11px] font-mono border"
      style={{
        background:  'var(--bg-elevated)',
        borderColor: color || 'var(--border-default)',
        color:       color || 'var(--text-secondary)',
      }}>
      <span style={{ color: 'var(--text-tertiary)' }}>{label} : </span>
      <b>{typeof value === 'number' ? value.toLocaleString() : (value ?? '—')}</b>
    </div>
  );

  switch (stepId) {
    case 'filter_year':
      return (
        <>
          <Mini label="Année"            value={data.year} color="var(--accent-blue)"/>
          <Mini label="OT totaux (CSV)"  value={data.n_rows_total}/>
          <Mini label="OT année"         value={data.n_ot_year} color="var(--accent-orange)"/>
          <Mini label="OT niveaux 3+4"   value={data.n_ot_year_levels_34} color="var(--accent-orange)"/>
          {/* Distribution niveaux complète */}
          <div className="col-span-full mt-1">
            <p className="text-[10px] mb-1" style={{ color: 'var(--text-tertiary)' }}>
              Distribution WOWO_EQUIPMENT_LEVEL (toutes années) :
            </p>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(data.level_distribution_all || {}).map(([lvl, n]) => (
                <span key={lvl}
                  className="px-2 py-0.5 rounded text-[11px] font-mono border"
                  style={{
                    background:  'var(--bg-elevated)',
                    borderColor: ['3','4'].includes(lvl) ? 'var(--accent-orange)' : 'var(--border-default)',
                    color:       ['3','4'].includes(lvl) ? 'var(--accent-orange)' : 'var(--text-secondary)',
                  }}>
                  niveau {lvl} : <b>{Number(n).toLocaleString()}</b>
                </span>
              ))}
            </div>
          </div>
        </>
      );

    case 'hierarchy':
      return (
        <>
          <Mini label="Pannes rattachées"     value={data.n_pannes_with_machine} color="var(--success)"/>
          <Mini label="Pannes orphelines"     value={data.n_pannes_orphelines}   color="var(--text-muted)"/>
          <Mini label="Machines mères trouvées" value={data.n_machines_meres}    color="var(--accent-blue)"/>
        </>
      );

    case 'select_comps':
      return (
        <>
          <Mini label="Composants uniques (3+4)" value={data.n_composants_uniques_34}/>
          <Mini label={`Modélisables (≥${data.min_failures})`}
                value={data.n_composants_modelisables} color="var(--accent-orange)"/>
          <Mini label="Machines mères modélisées" value={data.n_machines_meres_modelisees}
                color="var(--accent-blue)"/>
          <Mini label="OT après filtrage" value={data.n_ot_apres_min_failures}/>
        </>
      );

    case 'lookup':
      return (
        <>
          <Mini label="Composants indexés"   value={data.n_composants_indexes}/>
          <Mini label="Dates de panne"       value={data.n_dates_panne_distinct}/>
          <Mini label="Dates de maintenance" value={data.n_dates_maint_distinct}/>
        </>
      );

    case 'panel':
      return (
        <>
          <Mini label="Lignes (panel)" value={data.n_rows_panel} color="var(--accent-purple)"/>
          <Mini label="Jours"          value={data.n_jours}/>
          <Mini label="Composants"     value={data.n_composants}/>
          <Mini label="Pannes (=1)"    value={data.n_failures_in_panel} color="var(--error)"/>
          <Mini label="Maintenances (=1)" value={data.n_maintenances_in_panel} color="var(--success)"/>
        </>
      );

    case 'rul':
      return (
        <>
          <Mini label="RUL min" value={data.rul_min}/>
          <Mini label="RUL max" value={data.rul_max}/>
          <Mini label="RUL moy" value={data.rul_mean != null ? Number(data.rul_mean).toFixed(1) : '—'}/>
          <Mini label="RUL médiane" value={data.rul_median}/>
          {/* 🆕 Le user veut les valeurs ABSOLUES par seuil, pas que les % */}
          <div className="col-span-full mt-1">
            <p className="text-[10px] mb-1" style={{ color: 'var(--text-tertiary)' }}>
              Nombre de jours par seuil RUL :
            </p>
            <div className="flex flex-wrap gap-1.5">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                    style={{ borderColor: 'var(--error)', color: 'var(--error)' }}>
                RUL ≤ 5j : <b>{Number(data.n_rul_le_5 || 0).toLocaleString()}</b>
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                    style={{ borderColor: 'var(--accent-orange)', color: 'var(--accent-orange)' }}>
                RUL ≤ 10j : <b>{Number(data.n_rul_le_10 || 0).toLocaleString()}</b>
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                    style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}>
                RUL ≤ 30j : <b>{Number(data.n_rul_le_30 || 0).toLocaleString()}</b>
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono border"
                    style={{ borderColor: 'var(--success)', color: 'var(--success)' }}>
                RUL ≤ 90j : <b>{Number(data.n_rul_le_90 || 0).toLocaleString()}</b>
              </span>
            </div>
          </div>
        </>
      );

    case 'rolling':
      return (
        <>
          <Mini label="Lignes" value={data.n_rows} color="var(--accent-purple)"/>
          <div className="col-span-full">
            <p className="text-[10px] mb-1" style={{ color: 'var(--text-tertiary)' }}>
              Features créées :
            </p>
            <div className="flex flex-wrap gap-1">
              {[...(data.feature_cols || []), ...(data.extra_features || [])].map(f => (
                <code key={f} className="text-[10px] px-1.5 py-0.5 rounded"
                  style={{ background: 'var(--bg-elevated)', color: 'var(--accent-purple)' }}>
                  {f}
                </code>
              ))}
            </div>
          </div>
        </>
      );

    case 'export':
      return (
        <>
          <Mini label="Lignes" value={data.n_rows} color="var(--brand-primary)"/>
          <Mini label="Colonnes" value={data.n_cols}/>
          <Mini label="Composants" value={data.n_composants}/>
          <Mini label="Machines" value={data.n_machines}/>
        </>
      );

    default:
      return null;
  }
}


function Accordion({ id, open, onToggle, icon: Icon, color, title, subtitle, children }) {
  return (
    <div className="rounded-xl border overflow-hidden"
      style={{
        background:  'var(--bg-card)',
        borderColor: open ? color : 'var(--border-default)',
      }}>
      <button onClick={() => onToggle(id)}
        className="w-full flex items-center justify-between px-4 py-3 transition-all"
        style={{ background: open ? 'var(--bg-elevated)' : 'transparent' }}>
        <div className="flex items-start gap-2 text-left">
          {Icon && <Icon size={15} style={{ color, flexShrink: 0, marginTop: 2 }}/>}
          <div>
            <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{title}</p>
            {subtitle && (
              <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-tertiary)' }}>{subtitle}</p>
            )}
          </div>
        </div>
        {open
          ? <ChevronUp   size={16} style={{ color }}/>
          : <ChevronDown size={16} style={{ color: 'var(--text-muted)' }}/>}
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


function StatCard({ label, value, highlight }) {
  return (
    <div className="rounded-lg p-3 border"
      style={{
        background:  'var(--bg-elevated)',
        borderColor: highlight ? 'var(--brand-primary)' : 'var(--border-default)',
      }}>
      <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
      <p className="text-xl font-bold font-mono"
         style={{ color: highlight ? 'var(--brand-primary)' : 'var(--text-primary)' }}>
        {value == null ? '—' : Number(value).toLocaleString()}
      </p>
    </div>
  );
}


function PreviewTable({ rows, rowSelector, onRowsChange }) {
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]);

  return (
    <div>
      {/* Row selector style Azure */}
      {onRowsChange && (
        <div className="flex items-center justify-between mb-2">
          <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
            Affichage : {rows.length} ligne{rows.length > 1 ? 's' : ''} · {cols.length} colonnes
          </p>
          <div className="flex items-center gap-1.5 text-xs">
            <span style={{ color: 'var(--text-tertiary)' }}>Lignes :</span>
            {[5, 10, 20, 50].map(n => (
              <button key={n}
                onClick={() => onRowsChange(n)}
                className="px-2 py-0.5 rounded text-xs font-mono border"
                style={{
                  background:  rowSelector === n ? 'var(--brand-primary)' : 'var(--bg-elevated)',
                  borderColor: rowSelector === n ? 'var(--brand-primary)' : 'var(--border-default)',
                  color:       rowSelector === n ? 'var(--bg-elevated)' : 'var(--text-tertiary)',
                }}>
                {n}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border"
           style={{ borderColor: 'var(--border-subtle)' }}>
        <table className="text-xs font-mono w-full" style={{ color: 'var(--text-secondary)' }}>
          <thead style={{ background: 'var(--bg-card-alt)' }}>
            <tr>
              <th className="px-2 py-1.5 text-left font-semibold sticky left-0 z-10"
                  style={{ color: 'var(--text-tertiary)', background: 'var(--bg-card-alt)' }}>#</th>
              {cols.map(c => (
                <th key={c} className="px-2 py-1.5 text-left whitespace-nowrap font-semibold"
                    style={{ color: 'var(--text-tertiary)' }}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} style={{
                background: i % 2 ? 'var(--bg-card)' : 'transparent',
              }}>
                <td className="px-2 py-1 sticky left-0 z-10"
                    style={{
                      color: 'var(--text-muted)',
                      background: i % 2 ? 'var(--bg-card)' : 'var(--bg-card)',
                    }}>{i}</td>
                {cols.map(c => {
                  let v = row[c];
                  if (v == null) v = '—';
                  else if (typeof v === 'number') v = isFinite(v) ? +v.toFixed(3) : v;
                  else if (typeof v === 'string' && v.length > 22) v = v.slice(0, 20) + '…';
                  const isEvent = (c === 'failure' || c === 'maintenance') && row[c] === 1;
                  return (
                    <td key={c} className="px-2 py-1 whitespace-nowrap"
                        style={{
                          color: isEvent ? 'var(--accent-orange)' :
                                 c === 'RUL' ? 'var(--accent-purple)' : 'inherit',
                          fontWeight: isEvent || c === 'RUL' ? 'bold' : 'normal',
                        }}>
                      {String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
