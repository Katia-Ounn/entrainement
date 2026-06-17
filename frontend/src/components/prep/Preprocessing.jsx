/**
 * Preprocessing.jsx — Sous-onglet 4 ⭐ Pédagogique pour jury.
 *
 *   Structure en 5 sections (chacune avec titre + explication) :
 *     1. ⚙️ Configuration des paramètres
 *     2. 🔪 Split temporel par date (Train 70% / Val 15% / Test 15%)
 *     3. 📏 Normalisation MinMax (AVANT / APRÈS côte-à-côte)
 *     4. 🧱 Séquençage temporel (visualisation t=0 → t=lookback)
 *     5. ⚖️ Poids d'entraînement — formule notebook w=1+factor×(1−RUL/MAX_RUL)
 *
 *   Tout est expliqué inline → le jury comprend SANS commentaire externe.
 */
import { useState } from 'react';
import {
  Play, Loader, AlertCircle, ChevronRight, Layers,
  CheckCircle2, Clock, Target, Weight, Percent, Filter,
  Scale, Scissors, Ruler, Boxes, BarChart3,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import toast from 'react-hot-toast';
import { useApp } from '../../AppContext';

const API = 'http://localhost:8000';

const LOOKBACK_PRESETS = [7, 14, 17, 21, 30, 60, 90];
const MAX_RUL_PRESETS  = [10, 20, 30, 60, 90];
// holdoutRatio = val_ratio = test_ratio → train = 1 - 2 × holdout
const HOLDOUT_PRESETS  = [0.10, 0.15, 0.20];
// weight_factor : w = 1 + factor × (1 − RUL/MAX_RUL)  — notebook PFE
const WEIGHT_PRESETS   = [2, 4, 6, 8];


const STEP_ORDER = [
  { id: "config",    label: "Configuration" },
  { id: "split",     label: "Split temporel" },
  { id: "normalize", label: "Normalisation MinMax" },
  { id: "sequence",  label: "Sequencage temporel" },
  { id: "weights",   label: "Poids entrainement" },
];


export default function Preprocessing({ datasetId, onCompleted }) {
  const { preprocResult, setPreprocResult, markPrepStep, selectedFeatures,
          bumpDatasetVersion } = useApp();

  const [lookback,      setLookback]      = useState(preprocResult?.lookback        ?? 21);
  const [currentMaxRul, setCurrentMaxRul] = useState(preprocResult?.current_max_rul ?? 30);
  const [weightFactor,  setWeightFactor]  = useState(preprocResult?.weight_factor   ?? 4);
  const [holdoutRatio,  setHoldoutRatio]  = useState(preprocResult?.val_ratio       ?? 0.15);

  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  // 🆕 statut par étape : 'pending' | 'running' | 'done'
  const [stepStatus, setStepStatus] = useState({});
  const [activeStep, setActiveStep] = useState(null);  // id de l'étape en cours

  const run = async () => {
    setLoading(true); setError(null);
    // Reset progress : toutes les étapes "running" (la première) ou "pending"
    setStepStatus(Object.fromEntries(STEP_ORDER.map((s, i) => [s.id, i === 0 ? 'running' : 'pending'])));
    setActiveStep(STEP_ORDER[0].id);
    // ⚠️ on n'efface PAS preprocResult tout de suite → on enrichit pas-à-pas
    // pour éviter le flash blanc. On part d'un objet vide qui sera mergé.
    let partial = {
      lookback,
      current_max_rul: currentMaxRul,
      weight_factor:   weightFactor,
    };
    setPreprocResult(partial);

    try {
      const res = await fetch(`${API}/api/datasets/${datasetId}/preprocessing_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lookback,
          current_max_rul: currentMaxRul,
          weight_factor:   weightFactor,
          val_ratio:       holdoutRatio,
          test_ratio:      holdoutRatio,
          feature_cols:    selectedFeatures,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }

      // ── Lecture NDJSON en streaming ──
      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';
      let finalData = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Découper par lignes complètes (\n)
        const lines = buffer.split('\n');
        buffer = lines.pop();   // dernière ligne incomplète → rebuffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith('//')) continue;   // padding/comment de flush — on ignore
          let evt;
          try { evt = JSON.parse(trimmed); } catch { continue; }

          if (evt.type === 'ping')    continue;   // padding anti-buffering (NDJSON valide)
          if (evt.type === 'step') {
            // step_id === 'started' = signal early "le backend a reçu, ça bosse"
            if (evt.step_id === 'started') continue;

            // Merge data dans partial → render progressif
            partial = { ...partial, ...evt.data };
            setPreprocResult({ ...partial });

            // Marque l'étape comme done + active la suivante
            setStepStatus(prev => {
              const next = { ...prev, [evt.step_id]: 'done' };
              const idx  = STEP_ORDER.findIndex(s => s.id === evt.step_id);
              const nxt  = STEP_ORDER[idx + 1];
              if (nxt && next[nxt.id] !== 'done') {
                next[nxt.id] = 'running';
                setActiveStep(nxt.id);
              }
              return next;
            });
          } else if (evt.type === 'done') {
            finalData = evt.data;
            partial = { ...partial, ...evt.data };
            setPreprocResult({ ...partial });
          } else if (evt.type === 'error') {
            throw new Error(evt.message || 'Erreur backend');
          }
        }
      }

      // Toutes les étapes finalisées
      setStepStatus(Object.fromEntries(STEP_ORDER.map(s => [s.id, 'done'])));
      setActiveStep(null);
      markPrepStep('preprocessing');
      // 🆕 force le re-fetch de la liste datasets dans TrainingPanel / PreparationPanel
      // → le statut "preprocessed" est synchronisé partout
      bumpDatasetVersion();
      const xs = (finalData || partial).X_train_num_shape || [];
      toast.success(
        `Prétraitement OK · ${xs[0] || '?'} séquences · ${(finalData || partial).num_classes_comp || '?'} composants`,
        { duration: 4000 }
      );
    } catch (e) {
      setError(e.message);
      toast.error(`Prétraitement échoué : ${e.message}`);
      setStepStatus(prev => {
        const next = { ...prev };
        // L'étape active passe en "error"
        if (activeStep) next[activeStep] = 'error';
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  // Helpers : a-t-on déjà les données pour rendre une section donnée ?
  const isReady = (stepId) => stepStatus[stepId] === 'done';

  return (
    <div className="space-y-6">
      {/* 🆕 Bandeau features sélectionnées (transparence vs FeatureEngineering) */}
      <div className="rounded-xl border px-4 py-3 flex items-start gap-2 flex-wrap"
        style={{
          background:  'var(--tint-info-bg)',
          borderColor: 'var(--accent-blue)',
        }}>
        <Filter size={14} style={{ color: 'var(--accent-blue)', flexShrink: 0, marginTop: 2 }}/>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold" style={{ color: 'var(--accent-blue)' }}>
            🎛️ Features qui seront envoyées au modèle ({selectedFeatures.length})
          </p>
          <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Modifiable depuis l'onglet précédent <b>Feature Engineering</b> → section
            "Sélection des features". Liste actuelle :
          </p>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {selectedFeatures.map(f => (
              <span key={f} className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                style={{
                  background: 'color-mix(in srgb, var(--accent-blue) 18%, var(--bg-elevated))',
                  color:      'var(--accent-blue)',
                }}>
                {f}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ═══════ Section 1 : Configuration ═══════ */}
      <Section
        icon={Layers} color="var(--brand-primary)"
        title="⚙️ Étape 1 — Configuration des paramètres"
        description="Règle les 5 paramètres qui contrôlent comment le pipeline transforme le Dataset_V1 en tenseurs LSTM/GRU. Chaque paramètre a un impact direct expliqué ci-dessous.">

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <ConfigCard
            icon={Clock} color="var(--accent-blue)"
            title="Lookback — fenêtre temporelle"
            description="Le modèle voit les N derniers JOURS pour prédire le RUL du jour N+1. Plus long = capture les tendances longues mais entraînement plus lent."
            example={`Si lookback=${lookback}, le modèle voit J-${lookback}, J-${lookback-1}, …, J-1 pour prédire RUL(J).`}
            unit="jours" presets={LOOKBACK_PRESETS}
            value={lookback} onChange={setLookback}
            min={3} max={180}
          />

          <ConfigCard
            icon={Target} color="var(--accent-purple)"
            title="Current Max RUL — plafond du RUL"
            description="Au-delà de ce plafond, on considère le composant SAIN. La cible RUL est clipée → focalise le modèle sur la zone critique."
            example={`Si max=${currentMaxRul}j : RUL réel=120j → ramené à ${currentMaxRul}j (sain). RUL=3j → reste 3j (critique).`}
            unit="jours" presets={MAX_RUL_PRESETS}
            value={currentMaxRul} onChange={setCurrentMaxRul}
            min={5} max={365} highlight
          />

          <ConfigCard
            icon={Weight} color="var(--accent-orange)"
            title="Weight Factor — poids RUL faibles"
            description="Facteur dans la formule notebook : w = 1 + factor×(1−RUL/MAX_RUL). Défaut=4 (notebook PFE). Plus haut = le modèle punit davantage les erreurs proches de la panne."
            example={`Formule notebook : w = 1 + ${weightFactor}×(1−RUL/${currentMaxRul}). RUL=0 → w=${(1 + Number(weightFactor)).toFixed(1)} · RUL=max → w=1.`}
            unit="×" presets={WEIGHT_PRESETS}
            value={weightFactor} onChange={setWeightFactor}
            min={0} max={20}
          />

          <ConfigCard
            icon={Percent} color="var(--accent-green)"
            title="Split temporel — val + test holdout"
            description="Split PAR PLAGE DE DATES (chronologique). Les données les plus récentes vont en val puis test. Val et test ont le même ratio. Train = 1 − 2 × holdout."
            example={`Holdout=${Math.round(holdoutRatio * 100)}% → Train ${Math.round((1 - 2 * holdoutRatio) * 100)}% / Val ${Math.round(holdoutRatio * 100)}% / Test ${Math.round(holdoutRatio * 100)}%`}
            unit="" presetSuffix="%"
            presets={HOLDOUT_PRESETS}
            value={holdoutRatio} onChange={setHoldoutRatio}
            min={0.05} max={0.35} step={0.05}
            formatPreset={(v) => `${Math.round(v * 100)}`}
          />

        </div>

        <button
          onClick={run} disabled={loading}
          className="w-full mt-3 px-5 py-3 rounded-lg text-sm font-bold flex items-center justify-center gap-2"
          style={{
            background:  loading ? 'var(--bg-card-alt)' : 'var(--brand-primary)',
            color:       'var(--bg-elevated)',
            border:      '1px solid var(--brand-primary)',
            cursor:      loading ? 'wait' : 'pointer',
          }}>
          {loading
            ? <><Loader size={14} className="animate-spin"/> Prétraitement en cours…</>
            : <><Play size={14}/> Lancer le prétraitement</>}
        </button>

        {/* 🆕 Stepper progressif — visible dès qu'un run est lancé */}
        {(loading || Object.keys(stepStatus).length > 0) && (
          <div className="mt-3 rounded-lg border p-2 flex flex-wrap gap-1.5"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
            {STEP_ORDER.map((s, i) => {
              const st = stepStatus[s.id] || 'pending';
              const cfg = {
                pending: { bg: 'var(--bg-card-alt)', color: 'var(--text-muted)',    icon: '○' },
                running: { bg: 'var(--tint-info-bg)', color: 'var(--accent-blue)',  icon: '⟳' },
                done:    { bg: 'var(--tint-success-bg)', color: 'var(--success)',   icon: '✓' },
                error:   { bg: 'var(--tint-error-bg)',   color: 'var(--error)',     icon: '✗' },
              }[st];
              return (
                <span key={s.id}
                  className="px-2 py-1 rounded text-[11px] font-mono flex items-center gap-1.5 border"
                  style={{ background: cfg.bg, color: cfg.color, borderColor: cfg.color }}>
                  <span className={st === 'running' ? 'animate-spin inline-block' : ''}>
                    {cfg.icon}
                  </span>
                  <span className="font-semibold">{i + 1}.</span>
                  <span>{s.label}</span>
                </span>
              );
            })}
          </div>
        )}

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
      </Section>

      {!preprocResult && !loading && (
        <div className="rounded-xl border p-8 text-center text-sm"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--border-default)',
            borderStyle: 'dashed',
            color:       'var(--text-tertiary)',
          }}>
          👆 Configure tes paramètres puis clique sur <b>Lancer le prétraitement</b>.
          Les 5 étapes apparaîtront ici (config, split, normalisation, séquençage, poids).
        </div>
      )}

      {preprocResult && (
        <>
          {/* ═══════ Section 2 : Split temporel ═══════ */}
          {isReady('split') && (
          <Section
            icon={Scissors} color="var(--accent-green)"
            title="🔪 Étape 3 — Split temporel par date"
            description="Les données sont triées par date. Les plus anciennes → Train. Les suivantes → Validation. Les plus récentes → Test. Respecte la chronologie réelle.">

            <div className="grid grid-cols-3 gap-4 mb-3">
              <SplitVisualBar
                label="Train" pct={Math.round((1 - 2 * holdoutRatio) * 100)}
                count={preprocResult.n_train_rows}
                color="var(--accent-blue)"
                dateEnd={preprocResult.train_date_end}
              />
              <SplitVisualBar
                label="Validation" pct={Math.round(holdoutRatio * 100)}
                count={preprocResult.n_val_rows}
                color="var(--accent-purple)"
                dateEnd={preprocResult.val_date_end}
              />
              <SplitVisualBar
                label="Test" pct={Math.round(holdoutRatio * 100)}
                count={preprocResult.n_test_rows}
                color="var(--accent-green)"
                dateEnd={preprocResult.test_date_end}
              />
            </div>
          </Section>
          )}

          {/* ═══════ Section 3 : Normalisation AVANT / APRÈS ═══════ */}
          {isReady('normalize') && (
          <Section
            icon={Ruler} color="var(--accent-orange)"
            title="📏 Étape 3 — Normalisation MinMax"
            description="MinMaxScaler transforme chaque feature [min, max] → [0, 1]. Le LSTM converge mieux avec des inputs normalisés. Le scaler est FITTÉ sur train uniquement (pas de leak).">

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <BeforeAfterTable
                title="⏪ AVANT — Données brutes (5 premières lignes train)"
                accent="var(--text-muted)"
                rows={preprocResult.preview_X_raw}
                features={preprocResult.features}
                rawY={preprocResult.preview_y_raw}
              />
              <BeforeAfterTable
                title="⏩ APRÈS — Normalisé entre 0 et 1"
                accent="var(--accent-orange)"
                rows={(preprocResult.preview_normalized_X || []).map((row, i) => {
                  const obj = {};
                  preprocResult.features?.forEach((f, j) => { obj[f] = row[j]; });
                  return obj;
                })}
                features={preprocResult.features}
                colored
              />
            </div>
          </Section>
          )}

          {/* ═══════ Section 4 : Séquençage temporel ═══════ */}
          {isReady('sequence') && (
          <Section
            icon={Boxes} color="var(--accent-blue)"
            title="🧱 Étape 4 — Séquençage temporel"
            description={`On glisse une fenêtre de ${lookback} jours sur la timeline de chaque composant. À chaque position : X = ${lookback} jours d'historique, y = RUL du jour suivant.`}>

            <SequenceVisualizer
              lookback={lookback}
              features={preprocResult.features?.length || 9}
              featureCols={preprocResult.features || []}
              sequenceRaw={preprocResult.sequence_full_raw}
              sequenceNorm={preprocResult.sequence_full_norm}
              sequenceMeta={preprocResult.sequence_meta}
            />

            <div className="grid grid-cols-3 gap-3 mt-4">
              <ShapeBadge label="X_train" shape={preprocResult.X_train_num_shape} color="var(--accent-blue)" />
              <ShapeBadge label="X_val"   shape={preprocResult.X_val_num_shape}   color="var(--accent-purple)" />
              <ShapeBadge label="X_test"  shape={preprocResult.X_test_num_shape}  color="var(--accent-green)" />
            </div>
            <div className="grid grid-cols-3 gap-3 mt-2">
              <ShapeBadge label="y_train" shape={preprocResult.y_train_shape} color="var(--accent-blue)" />
              <ShapeBadge label="y_val"   shape={preprocResult.y_val_shape}   color="var(--accent-purple)" />
              <ShapeBadge label="y_test"  shape={preprocResult.y_test_shape}  color="var(--accent-green)" />
            </div>
            <div className="mt-2">
              <ShapeBadge label="Embedding (n_composants)" shape={[preprocResult.num_classes_comp]} color="var(--accent-orange)" />
            </div>
          </Section>
          )}

          {/* ═══════ Section 5 : Poids d'entraînement ═══════ */}
          {isReady('weights') && (
          <Section
            icon={Weight} color="var(--accent-orange)"
            title="⚖️ Étape 5 — Poids d'entraînement"
            description={`Formule notebook PFE : w = 1 + ${weightFactor}×(1−RUL/${currentMaxRul}). Plage : [1, ${1 + Number(weightFactor)}]. RUL=0 (panne) → poids max. RUL=max (sain) → poids 1. Loss asymétrique : sous-estimation ×4 plus pénalisée.`}>

            {preprocResult.weight_stats && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                <StatBadge label="Min"  value={preprocResult.weight_stats.min?.toFixed(2)}  color="var(--success)"/>
                <StatBadge label="Max"  value={preprocResult.weight_stats.max?.toFixed(2)}  color="var(--error)"/>
                <StatBadge label="Moy." value={preprocResult.weight_stats.mean?.toFixed(2)} color="var(--accent-blue)"/>
                <StatBadge label="w > 8" value={`${preprocResult.weight_stats.n_above_8} (${Math.round(100 * preprocResult.weight_stats.n_above_8 / preprocResult.weight_stats.n_total)}%)`} color="var(--accent-orange)"/>
              </div>
            )}

            {preprocResult.weight_histogram && (
              <div className="rounded-xl border p-3 mb-3"
                style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
                <p className="text-xs mb-2" style={{ color: 'var(--text-tertiary)' }}>
                  📊 Distribution des poids sur les {preprocResult.weight_stats?.n_total?.toLocaleString()} séquences train
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={(preprocResult.weight_histogram.bins || []).slice(0, -1).map((b, i) => ({
                    range: `${b.toFixed(1)}–${preprocResult.weight_histogram.bins[i + 1].toFixed(1)}`,
                    count: preprocResult.weight_histogram.counts[i] || 0,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)"/>
                    <XAxis dataKey="range" tick={{ fill: 'var(--text-tertiary)', fontSize: 9 }}/>
                    <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}/>
                    <Tooltip contentStyle={{
                      background: 'var(--bg-elevated)',
                      border: '1px solid var(--border-strong)',
                      borderRadius: 6,
                      color: 'var(--text-primary)',
                    }}/>
                    <Bar dataKey="count" radius={[2,2,0,0]}>
                      {(preprocResult.weight_histogram.counts || []).map((_, i) => {
                        const bin = (preprocResult.weight_histogram.bins || [])[i] || 1;
                        const color = bin >= 14 ? 'var(--error)'
                                    : bin >= 8  ? 'var(--accent-orange)'
                                    : bin >= 2  ? 'var(--accent-blue)'
                                    :             'var(--success)';
                        return <Cell key={i} fill={color}/>;
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
                  🟢 sains (~1) · 🔵 modérés · 🟠 dégradés · 🔴 critiques (~{1 + Number(weightFactor)})
                </p>
              </div>
            )}

            {/* Échantillons critiques + explication pédagogique */}
            {preprocResult.preview_weights_critical?.length > 0 && (
              <div className="rounded-xl border p-3"
                style={{ background: 'var(--bg-card)', borderColor: 'var(--error)' }}>
                <p className="text-xs font-semibold mb-1.5" style={{ color: 'var(--error)' }}>
                  🔥 Aperçu des poids amplifiés (séquences critiques : RUL ≤ 5 jours)
                </p>

                {/* Explication pédagogique */}
                <div className="rounded-lg p-2 mb-2 text-[11px] leading-relaxed"
                  style={{ background: 'var(--bg-card-alt)', color: 'var(--text-secondary)' }}>
                  <p className="mb-1">
                    <b>À quoi ça sert ?</b> Pendant <code>model.fit()</code>, chaque
                    séquence est multipliée par son poids dans la fonction de coût.
                    Une séquence avec un poids de <b>15</b> est <b>15× plus pénalisée</b>
                    qu'une séquence sans danger (poids = 1).
                  </p>
                  <p>
                    <b>Formule notebook PFE :</b>
                    <code> w = 1 + {Number(weightFactor)} × (1 − RUL / {currentMaxRul})</code>
                    <br/>· RUL = 0j → w = {(1 + Number(weightFactor)).toFixed(1)} (panne imminente — max)
                    <br/>· RUL = 5j → w ≈ {(1 + (1 - 5/Number(currentMaxRul)) * Number(weightFactor)).toFixed(1)}
                    <br/>· RUL = {currentMaxRul}j → w = 1 (composant sain — poids minimal)
                  </p>
                </div>

                <p className="text-[10px] mb-1.5" style={{ color: 'var(--text-muted)' }}>
                  10 premières séquences où RUL ≤ 5j :
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {preprocResult.preview_weights_critical.map((w, i) => (
                    <span key={i}
                      className="px-2 py-1 rounded text-xs font-mono border font-bold"
                      style={{
                        background:  'var(--tint-error-bg)',
                        borderColor: 'var(--error)',
                        color:       'var(--error)',
                      }}>
                      {Number(w).toFixed(2)}
                    </span>
                  ))}
                </div>
                <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>
                  ↑ Ces poids amplifiés <b>forcent le modèle à bien prédire les pannes
                  imminentes</b> au détriment d'erreurs sur les composants sains (qui sont
                  moins critiques pour l'opération).
                </p>
              </div>
            )}
          </Section>
          )}

          {/* CTA continuer — seulement quand tout est fini */}
          {isReady('weights') && !loading && (
          <div className="rounded-xl p-3 flex items-center justify-between"
            style={{ background: 'var(--tint-success-bg)' }}>
            <span className="flex items-center gap-2 text-sm font-semibold"
                  style={{ color: 'var(--success)' }}>
              <CheckCircle2 size={16}/> Prétraitement terminé — prêt pour l'entraînement
            </span>
            <button onClick={onCompleted}
              className="px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-1 border"
              style={{
                background:  'var(--success)',
                color:       'var(--bg-elevated)',
                borderColor: 'var(--success)',
              }}>
              Aller à l'entraînement <ChevronRight size={14}/>
            </button>
          </div>
          )}
        </>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════
// Sous-composants
// ═══════════════════════════════════════════════════════════════

function Section({ icon: Icon, color, title, description, children }) {
  return (
    <div className="rounded-xl border p-4"
      style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-default)' }}>
      <div className="flex items-start gap-2 mb-3">
        {Icon && <Icon size={16} style={{ color, flexShrink: 0, marginTop: 2 }}/>}
        <div className="flex-1">
          <h3 className="text-base font-bold" style={{ color: 'var(--text-primary)' }}>{title}</h3>
          {description && (
            <p className="text-xs mt-1 leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
              {description}
            </p>
          )}
        </div>
      </div>
      <div className="mt-3">{children}</div>
    </div>
  );
}


function ConfigCard({ icon: Icon, color, title, description, example,
                      unit, presetSuffix = '', presets, value, onChange,
                      min, max, step = 1, highlight, formatPreset }) {
  return (
    <div className="rounded-lg border p-3"
      style={{
        background:  highlight ? 'var(--bg-card)' : 'var(--bg-card-alt)',
        borderColor: highlight ? color : 'var(--border-default)',
      }}>
      <div className="flex items-center gap-2 mb-1.5">
        <Icon size={13} style={{ color }}/>
        <p className="text-xs font-semibold uppercase tracking-widest"
           style={{ color: 'var(--text-tertiary)' }}>{title}</p>
      </div>
      {description && (
        <p className="text-[11px] mb-1.5" style={{ color: 'var(--text-muted)' }}>{description}</p>
      )}
      {example && (
        <p className="text-[10px] mb-2 px-2 py-1 rounded font-mono italic"
          style={{
            background: 'var(--bg-elevated)',
            color:      color,
            border:     `1px dashed ${color}`,
          }}>
          💡 {example}
        </p>
      )}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {presets.map((p) => {
          const isActive = Math.abs(value - p) < 1e-6;
          return (
            <button key={p}
              onClick={() => onChange(p)}
              className="px-2.5 py-1 rounded text-xs font-mono border transition-all"
              style={{
                background:  isActive ? color : 'var(--bg-elevated)',
                color:       isActive ? 'var(--bg-elevated)' : 'var(--text-secondary)',
                borderColor: isActive ? color : 'var(--border-default)',
              }}>
              {formatPreset ? formatPreset(p) : p}{presetSuffix}
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-2">
        <input type="number" value={value} min={min} max={max} step={step}
          onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v); }}
          className="flex-1 px-2 py-1 rounded border text-xs font-mono outline-none"
          style={{
            background:  'var(--bg-elevated)',
            borderColor: 'var(--border-default)',
            color:       'var(--text-primary)',
          }}/>
        {unit && <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>{unit}</span>}
      </div>
    </div>
  );
}


function BalanceCard({ label, value, detail, color }) {
  return (
    <div className="rounded-lg border p-4 flex items-center gap-3"
      style={{ background: 'var(--bg-card)', borderColor: color }}>
      <div className="flex-1">
        <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
        <p className="text-2xl font-bold font-mono mt-0.5" style={{ color }}>
          {value != null ? Number(value).toLocaleString() : '—'}
        </p>
        <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{detail}</p>
      </div>
    </div>
  );
}


function SplitVisualBar({ label, pct, count, color, dateEnd }) {
  return (
    <div className="rounded-lg border p-3"
      style={{ background: 'var(--bg-card)', borderColor: color }}>
      <div className="flex justify-between text-xs mb-1.5">
        <span className="font-semibold" style={{ color }}>{label}</span>
        <span className="font-mono font-bold" style={{ color }}>{pct}%</span>
      </div>
      <div className="w-full h-2.5 rounded-full overflow-hidden mb-1.5"
           style={{ background: 'var(--bg-card-alt)' }}>
        <div className="h-full rounded-full transition-all"
             style={{ background: color, width: `${pct}%` }}/>
      </div>
      <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
        {count != null ? `${Number(count).toLocaleString()} lignes` : '—'}
        {dateEnd && <span className="ml-1">· jusqu'au {String(dateEnd).slice(0, 10)}</span>}
      </p>
    </div>
  );
}


function BeforeAfterTable({ title, accent, rows, features, rawY, colored }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="rounded-lg border p-3 text-xs"
        style={{
          background: 'var(--bg-card)',
          borderColor: 'var(--border-default)',
          color: 'var(--text-muted)',
        }}>
        {title} — données non disponibles
      </div>
    );
  }
  return (
    <div className="rounded-lg border overflow-hidden"
      style={{ background: 'var(--bg-card)', borderColor: accent }}>
      <p className="text-xs font-semibold uppercase tracking-widest px-3 py-2 border-b"
         style={{
           color: accent,
           borderColor: 'var(--border-subtle)',
           background: 'var(--bg-card-alt)',
         }}>
        {title}
      </p>
      <div className="overflow-x-auto">
        <table className="text-xs font-mono w-full" style={{ color: 'var(--text-secondary)' }}>
          <thead style={{ background: 'var(--bg-elevated)' }}>
            <tr>
              <th className="px-2 py-1 text-left" style={{ color: 'var(--text-tertiary)' }}>#</th>
              {features?.map(f => (
                <th key={f} className="px-2 py-1 text-right whitespace-nowrap"
                    style={{ color: 'var(--text-tertiary)' }}>{f}</th>
              ))}
              {rawY && <th className="px-2 py-1 text-right"
                          style={{ color: 'var(--accent-orange)' }}>RUL</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} style={{ background: i % 2 ? 'var(--bg-card-alt)' : 'transparent' }}>
                <td className="px-2 py-1" style={{ color: 'var(--text-muted)' }}>{i}</td>
                {features?.map(f => {
                  const v = row[f];
                  if (colored && typeof v === 'number') {
                    return (
                      <td key={f} className="px-2 py-1 text-right">
                        <span style={{
                          background: `color-mix(in srgb, ${accent} ${Math.round(v * 55)}%, transparent)`,
                          padding: '2px 5px',
                          borderRadius: 3,
                        }}>
                          {Number(v).toFixed(3)}
                        </span>
                      </td>
                    );
                  }
                  return (
                    <td key={f} className="px-2 py-1 text-right">
                      {typeof v === 'number' ? Number(v).toFixed(2) : (v ?? '—')}
                    </td>
                  );
                })}
                {rawY && (
                  <td className="px-2 py-1 text-right font-bold" style={{ color: 'var(--accent-orange)' }}>
                    {rawY[i]}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


function SequenceVisualizer({ lookback, features, featureCols, sequenceRaw, sequenceNorm, sequenceMeta }) {
  // 🆕 Mode tableau : si on a les vraies données, affiche un tableau complet
  const [view, setView] = useState('norm'); // 'norm' | 'raw'
  const hasData = sequenceRaw?.length > 0 || sequenceNorm?.length > 0;

  return (
    <div className="rounded-lg border p-4 space-y-3"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border-default)' }}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
          Construction d'UNE séquence pour un composant donné :
        </p>
        {hasData && (
          <div className="flex items-center gap-1 text-xs">
            <span style={{ color: 'var(--text-tertiary)' }}>Vue :</span>
            {[
              { id: 'raw',  label: 'AVANT (raw)',     color: 'var(--text-muted)' },
              { id: 'norm', label: 'APRÈS (normalisé)', color: 'var(--accent-orange)' },
            ].map(opt => (
              <button key={opt.id}
                onClick={() => setView(opt.id)}
                className="px-2 py-0.5 rounded text-xs font-mono border"
                style={{
                  background:  view === opt.id ? opt.color : 'var(--bg-elevated)',
                  borderColor: view === opt.id ? opt.color : 'var(--border-default)',
                  color:       view === opt.id ? 'var(--bg-elevated)' : opt.color,
                }}>
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Métadonnées de la séquence */}
      {sequenceMeta?.comp && (
        <div className="rounded p-2 text-[11px] font-mono flex flex-wrap gap-x-4 gap-y-1"
             style={{ background: 'var(--bg-card-alt)', color: 'var(--text-tertiary)' }}>
          <span>📦 Composant : <b style={{ color: 'var(--text-primary)' }}>{sequenceMeta.comp}</b></span>
          {sequenceMeta.date_start && (
            <span>📅 X : <b style={{ color: 'var(--text-primary)' }}>{String(sequenceMeta.date_start).slice(0,10)} → {String(sequenceMeta.date_end).slice(0,10)}</b></span>
          )}
          {sequenceMeta.date_target && (
            <span>🎯 cible : <b style={{ color: 'var(--accent-orange)' }}>{String(sequenceMeta.date_target).slice(0,10)} · RUL = {sequenceMeta.y_target_raw}j</b></span>
          )}
        </div>
      )}

      {/* TABLEAU complet : t=0..lookback-1 × features */}
      {hasData ? (
        <div className="overflow-x-auto rounded border"
             style={{ borderColor: 'var(--border-subtle)', maxHeight: 360 }}>
          <table className="text-xs font-mono w-full" style={{ color: 'var(--text-secondary)' }}>
            <thead style={{ background: 'var(--bg-card-alt)', position: 'sticky', top: 0 }}>
              <tr>
                <th className="px-2 py-1.5 text-left font-semibold sticky left-0 z-10"
                    style={{
                      color: 'var(--accent-blue)',
                      background: 'var(--bg-card-alt)',
                      minWidth: 50,
                    }}>t</th>
                {featureCols?.map(f => (
                  <th key={f} className="px-2 py-1.5 text-right font-semibold whitespace-nowrap"
                      style={{ color: 'var(--text-tertiary)' }}>{f}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(view === 'raw' ? sequenceRaw : sequenceNorm).map((row, i) => (
                <tr key={i} style={{ background: i % 2 ? 'var(--bg-card)' : 'transparent' }}>
                  <td className="px-2 py-1 sticky left-0 z-10 font-bold"
                      style={{
                        color: 'var(--accent-blue)',
                        background: i % 2 ? 'var(--bg-card)' : 'var(--bg-card)',
                      }}>t={i}</td>
                  {(view === 'raw'
                    ? featureCols.map(f => row[f])
                    : row
                  ).map((v, j) => {
                    if (view === 'norm' && typeof v === 'number') {
                      return (
                        <td key={j} className="px-2 py-1 text-right">
                          <span style={{
                            background: `color-mix(in srgb, var(--accent-orange) ${Math.round(v * 55)}%, transparent)`,
                            padding: '2px 5px', borderRadius: 3,
                          }}>
                            {Number(v).toFixed(3)}
                          </span>
                        </td>
                      );
                    }
                    return (
                      <td key={j} className="px-2 py-1 text-right">
                        {typeof v === 'number' ? Number(v).toFixed(2) : (v ?? '—')}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {/* Ligne de la cible y(t=lookback) */}
              {sequenceMeta?.y_target_raw != null && (
                <tr style={{ background: 'color-mix(in srgb, var(--accent-orange) 18%, var(--bg-card))' }}>
                  <td className="px-2 py-1.5 sticky left-0 font-bold"
                      style={{ background: 'color-mix(in srgb, var(--accent-orange) 18%, var(--bg-card))', color: 'var(--accent-orange)' }}>
                    t={lookback}<br/><span className="text-[9px]">(cible)</span>
                  </td>
                  <td colSpan={featureCols.length} className="px-2 py-1.5"
                      style={{ color: 'var(--accent-orange)' }}>
                    🎯 <b>ŷ = RUL = {sequenceMeta.y_target_raw} jours</b>
                    {view === 'norm' && sequenceMeta.y_target_norm != null && (
                      <span className="ml-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                        (normalisé : {Number(sequenceMeta.y_target_norm).toFixed(3)})
                      </span>
                    )}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        // Fallback visuel si data pas dispo
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {Array.from({ length: Math.min(lookback, 12) }).map((_, i) => (
            <div key={i} className="flex flex-col items-center" style={{ minWidth: 38 }}>
              <div className="rounded text-[10px] font-mono px-1 py-0.5 mb-1"
                style={{
                  background:  'color-mix(in srgb, var(--accent-blue) 25%, var(--bg-card))',
                  color:       'var(--accent-blue)',
                  border:      '1px solid var(--accent-blue)',
                }}>
                t={i}
              </div>
              <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{features}f</div>
            </div>
          ))}
          {lookback > 12 && <span style={{ color: 'var(--text-muted)' }}>… t={lookback - 1}</span>}
          <div className="text-2xl mx-1" style={{ color: 'var(--accent-orange)' }}>→</div>
          <div className="rounded text-xs font-mono px-2 py-1"
            style={{
              background:  'color-mix(in srgb, var(--accent-orange) 30%, var(--bg-card))',
              color:       'var(--accent-orange)',
              border:      '2px solid var(--accent-orange)',
            }}>
            ŷ = RUL(t={lookback})
          </div>
        </div>
      )}

      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        📐 <b>X_num</b> shape = <code>(N_séquences, {lookback}, {features})</code> ·
        <b className="ml-1">X_comp</b> = <code>(N_séquences,)</code> idx pour Embedding ·
        <b className="ml-1">y</b> = <code>(N_séquences,)</code> RUL normalisé
      </p>
    </div>
  );
}


function ShapeBadge({ label, shape, color }) {
  return (
    <div className="rounded-lg border p-3 text-center"
      style={{
        background:  'var(--bg-card)',
        borderColor: color,
      }}>
      <p className="text-[10px] mb-1" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
      <p className="text-sm font-bold font-mono" style={{ color }}>
        {Array.isArray(shape) ? `(${shape.join(', ')})` : '—'}
      </p>
    </div>
  );
}


function StatBadge({ label, value, color }) {
  return (
    <div className="rounded-lg border p-2 text-center"
      style={{ background: 'var(--bg-card)', borderColor: color }}>
      <p className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
      <p className="text-base font-bold font-mono" style={{ color }}>{value ?? '—'}</p>
    </div>
  );
}
