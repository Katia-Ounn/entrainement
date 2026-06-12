/**
 * MergeDatasets.jsx — Sous-onglet 5 : fusion temporelle d'un dataset existant
 * avec un nouveau CSV failure (enrichissement progressif).
 *
 * Workflow :
 *   1. Sélection dataset à enrichir (déjà fait via dropdown global)
 *   2. Upload nouveau CSV failure (drag/drop bouton)
 *      → POST /api/datasets/{id}/analyze_merge (preview SANS fusionner)
 *   3. Affichage diff (ancien / nouveau / fusion estimée)
 *   4. Bouton "Confirmer fusion" → POST /api/datasets/{id}/merge
 */
import { useState, useRef } from 'react';
import {
  GitMerge, FileText, AlertTriangle, AlertCircle, Loader,
  ChevronRight, Upload, CheckCircle2, X,
} from 'lucide-react';

const API = 'http://localhost:8000';

export default function MergeDatasets({ datasetId, datasets, onMergeDone }) {
  const [newFile,     setNewFile]     = useState(null);
  const [analysis,    setAnalysis]    = useState(null);
  const [analyzing,   setAnalyzing]   = useState(false);
  const [merging,     setMerging]     = useState(false);
  const [error,       setError]       = useState(null);
  const [success,     setSuccess]     = useState(null);
  const inputRef = useRef();

  const currentDs = datasets.find(d => d.id === datasetId);

  const resetAll = () => {
    setNewFile(null); setAnalysis(null); setError(null); setSuccess(null);
  };

  const handlePickFile = (file) => {
    setNewFile(file); setAnalysis(null); setError(null); setSuccess(null);
    if (file && datasetId) analyzeMerge(file);
  };

  const analyzeMerge = async (file) => {
    setAnalyzing(true); setError(null);
    try {
      const fd = new FormData();
      fd.append('new_failure_file', file);
      const res = await fetch(`${API}/api/datasets/${datasetId}/analyze_merge`, {
        method: 'POST', body: fd,
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      setAnalysis(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const confirmMerge = async () => {
    if (!newFile || !datasetId) return;
    setMerging(true); setError(null);
    try {
      const fd = new FormData();
      fd.append('new_failure_file', newFile);
      const res = await fetch(`${API}/api/datasets/${datasetId}/merge`, {
        method: 'POST', body: fd,
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setSuccess(data);
      onMergeDone?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setMerging(false);
    }
  };

  if (!datasetId) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <GitMerge size={32} style={{ color: 'var(--text-muted)' }} />
        <p className="text-sm mt-2" style={{ color: 'var(--text-tertiary)' }}>
          Sélectionne d'abord un dataset à enrichir dans le dropdown en haut.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <GitMerge size={16} style={{ color: 'var(--brand-primary)' }} />
        <h3 className="text-base font-bold" style={{ color: 'var(--text-primary)' }}>
          Fusion temporelle — enrichir « {currentDs?.name} »
        </h3>
      </div>

      <p className="text-xs leading-relaxed px-3 py-2 rounded-lg"
        style={{
          background:  'var(--tint-info-bg)',
          color:       'var(--text-secondary)',
          borderLeft:  '3px solid var(--accent-blue)',
        }}>
        Cette fonction permet d'ajouter <b>de nouvelles données failure</b> à un dataset
        Cevital existant (ex : période suivante, nouvelle usine). Le système re-calcule
        automatiquement les features et le RUL. ⚠️ Tu devras refaire le <b>prétraitement</b>
        et <b>réentraîner</b> ton modèle après la fusion.
      </p>

      {/* Drop zone */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest mb-1.5"
           style={{ color: 'var(--text-tertiary)' }}>
          Étape 1 — Choisir le nouveau CSV failure
        </p>
        <button onClick={() => inputRef.current?.click()}
          className="w-full rounded-lg px-3 py-3 flex items-center justify-between border text-sm text-left"
          style={{
            background:  'var(--bg-card)',
            borderColor: newFile ? 'var(--success)' : 'var(--border-default)',
            borderStyle: 'dashed',
            color:       'var(--text-secondary)',
          }}>
          <span className="flex items-center gap-2 truncate">
            <FileText size={14}
              style={{ color: newFile ? 'var(--success)' : 'var(--text-muted)' }}/>
            <span className="font-mono text-xs truncate">
              {newFile ? newFile.name : 'Cliquer pour choisir un fichier CSV…'}
            </span>
            {newFile && (
              <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>
                ({Math.round(newFile.size / 1024).toLocaleString()} ko)
              </span>
            )}
          </span>
          {newFile && (
            <button onClick={(e) => { e.stopPropagation(); resetAll(); }}
              className="text-xs px-2 py-0.5 rounded ml-2"
              style={{ background: 'var(--bg-elevated)', color: 'var(--text-tertiary)' }}>
              <X size={12} />
            </button>
          )}
        </button>
        <input ref={inputRef} type="file" accept=".csv" className="hidden"
          onChange={(e) => handlePickFile(e.target.files?.[0] || null)}/>
      </div>

      {/* Analyzing spinner */}
      {analyzing && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm"
             style={{ background: 'var(--bg-elevated)', color: 'var(--text-tertiary)' }}>
          <Loader size={14} className="animate-spin"/>
          Analyse en cours…
        </div>
      )}

      {/* Analysis result */}
      {analysis && !success && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-widest"
             style={{ color: 'var(--text-tertiary)' }}>
            Étape 2 — Aperçu de la fusion
          </p>

          {/* 3 colonnes : avant / nouveau / fusion estimée */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <DiffCard
              title="Dataset existant"
              color="var(--text-tertiary)"
              rows={[
                ['Lignes',     analysis.before?.n_rows],
                ['Composants', analysis.before?.n_composants],
              ]}
            />
            <DiffCard
              title="Nouveau fichier"
              color="var(--accent-blue)"
              rows={[
                ['Lignes',      analysis.new?.n_rows_new],
                ['Composants',  analysis.new?.n_composants_new],
                ['Période min', String(analysis.new?.date_min_new || '').slice(0,10)],
                ['Période max', String(analysis.new?.date_max_new || '').slice(0,10)],
              ]}
            />
            <DiffCard
              title="Après fusion (estimé)"
              color="var(--success)"
              highlight
              rows={[
                ['Lignes total',   analysis.after_merge?.n_rows_total],
                ['Composants',     analysis.after_merge?.n_composants_total],
                ['Communs',        analysis.after_merge?.common_components],
                ['Nouveaux',       analysis.after_merge?.new_components],
                ['Années',         (analysis.after_merge?.years_covered || []).join(', ')],
              ]}
            />
          </div>

          {/* Liste des nouveaux composants */}
          {analysis.after_merge?.list_new_components?.length > 0 && (
            <div className="rounded-lg p-3 border"
              style={{
                background:  'var(--bg-elevated)',
                borderColor: 'var(--border-default)',
              }}>
              <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-tertiary)' }}>
                {analysis.after_merge.list_new_components.length} premiers nouveaux composants
              </p>
              <div className="flex flex-wrap gap-1">
                {analysis.after_merge.list_new_components.map((c) => (
                  <span key={c}
                    className="px-2 py-0.5 rounded text-xs font-mono"
                    style={{
                      background: 'var(--bg-card)',
                      color:      'var(--accent-purple)',
                    }}>
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Warning + confirm */}
          <div className="rounded-lg px-3 py-2 text-xs flex items-start gap-2 border"
            style={{
              background:  'var(--tint-error-bg)',
              borderColor: 'var(--accent-orange)',
              color:       'var(--accent-orange)',
            }}>
            <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
            <span>
              La fusion va <b>recalculer toutes les features</b> et <b>réinitialiser le prétraitement</b>.
              Tu devras refaire les sous-onglets 3-4 et réentraîner ton modèle après.
            </span>
          </div>

          <div className="flex justify-end gap-2">
            <button onClick={resetAll}
              className="px-4 py-2 rounded-lg text-sm font-semibold border"
              style={{
                background:  'var(--bg-card)',
                borderColor: 'var(--border-default)',
                color:       'var(--text-tertiary)',
              }}>
              Annuler
            </button>
            <button onClick={confirmMerge} disabled={merging}
              className="px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 border"
              style={{
                background:  merging ? 'var(--bg-card-alt)' : 'var(--brand-primary)',
                borderColor: 'var(--brand-primary)',
                color:       'var(--bg-elevated)',
                cursor:      merging ? 'wait' : 'pointer',
              }}>
              {merging
                ? <><Loader size={14} className="animate-spin" /> Fusion…</>
                : <><GitMerge size={14} /> Confirmer la fusion</>}
            </button>
          </div>
        </div>
      )}

      {/* Erreur */}
      {error && (
        <div className="rounded-lg px-3 py-2 text-xs border"
          style={{
            background:  'var(--tint-error-bg)',
            color:       'var(--error)',
            borderColor: 'var(--error)',
          }}>
          <AlertCircle size={14} className="inline mr-1.5" />
          <span className="whitespace-pre-wrap font-mono">{error}</span>
        </div>
      )}

      {/* Succès */}
      {success && (
        <div className="rounded-lg p-4 border"
          style={{
            background:  'var(--tint-success-bg)',
            borderColor: 'var(--success)',
            color:       'var(--success)',
          }}>
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 size={16} />
            <b>Fusion réussie</b>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Lignes total : {success.after_merge?.n_rows_total ?? '—'} ·
            Composants : {success.after_merge?.n_composants_total ?? '—'} ·
            Années : {(success.after_merge?.years_covered || []).join(', ')}
          </p>
          <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
            ➡️ Va dans <b>Feature Engineering</b> pour recalculer le Dataset_V1
            avec les données fusionnées, puis refais le <b>Prétraitement</b>.
          </p>
        </div>
      )}
    </div>
  );
}


// ─── Helpers ───────────────────────────────────────────────
function DiffCard({ title, color, rows, highlight }) {
  return (
    <div className="rounded-xl p-3 border"
      style={{
        background:  highlight ? 'var(--tint-success-bg)' : 'var(--bg-elevated)',
        borderColor: highlight ? color : 'var(--border-default)',
      }}>
      <p className="text-xs font-semibold uppercase tracking-widest mb-2"
         style={{ color }}>
        {title}
      </p>
      <div className="space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between text-xs">
            <span style={{ color: 'var(--text-tertiary)' }}>{k}</span>
            <span className="font-mono font-semibold"
                  style={{ color: 'var(--text-primary)' }}>
              {v == null || v === '' ? '—' : String(v).length > 18 ? String(v).slice(0,18)+'…' : v}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
