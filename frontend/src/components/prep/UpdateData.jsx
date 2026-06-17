/**
 * UpdateData.jsx — Mise à jour des données GMAO.
 *
 * Remplace "Fusion / Réentraîn." par une fonctionnalité de mise à jour :
 *   - L'utilisateur uploade un nouvel export GMAO complet (2023 + 2024 + ...)
 *   - Le backend détecte le format automatiquement (nouveau ou ancien)
 *   - Les données du dataset sont remplacées
 *   - Le pipeline est remis à zéro pour re-run
 */
import { useState, useRef, useEffect } from 'react';
import { Upload, RefreshCw, RotateCcw, CheckCircle2, AlertCircle, Calendar, Database, FileText, Loader } from 'lucide-react';
import toast from 'react-hot-toast';
import { useApp } from '../../AppContext';

const API = 'http://localhost:8000';

export default function UpdateData({ datasetId, currentDataset, onUpdated }) {
  const { resetPipelineResults } = useApp();
  const [file,      setFile]      = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);
  const [dragging,  setDragging]  = useState(false);
  const [canUndo,   setCanUndo]   = useState(false);
  const [undoInfo,  setUndoInfo]  = useState(null);
  const [undoing,   setUndoing]   = useState(false);
  const inputRef = useRef();

  // Vérifie au montage si le dernier ajout peut être annulé (backup présent côté serveur)
  useEffect(() => {
    if (!datasetId) { setCanUndo(false); return; }
    let alive = true;
    fetch(`${API}/api/datasets/${datasetId}/can_undo_update`)
      .then(r => r.json())
      .then(d => { if (alive) { setCanUndo(!!d.can_undo); setUndoInfo(d.info || null); } })
      .catch(() => {});
    return () => { alive = false; };
  }, [datasetId]);

  const handleUndo = async () => {
    if (!datasetId) return;
    if (!window.confirm("Annuler le dernier ajout ? Le failure.csv reviendra à l'état précédent.")) return;
    setUndoing(true);
    try {
      const res  = await fetch(`${API}/api/datasets/${datasetId}/undo_update`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur serveur');
      setCanUndo(false); setUndoInfo(null); setResult(null);
      resetPipelineResults();
      toast.success(data.msg || 'Dernier ajout annulé');
      if (onUpdated) onUpdated();
    } catch (e) {
      toast.error(`Erreur annulation : ${e.message}`);
    } finally {
      setUndoing(false);
    }
  };

  const handleFile = (f) => {
    if (!f) return;
    if (!f.name.endsWith('.csv')) {
      toast.error('Fichier CSV requis');
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleUpdate = async () => {
    if (!file || !datasetId) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch(`${API}/api/datasets/${datasetId}/update_data`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur serveur');
      setResult(data);
      setCanUndo(true);
      setUndoInfo({ n_added: data.n_new, n_before: data.n_existing, n_after: data.n_total, date_max: data.date_max });
      resetPipelineResults();
      toast.success('Données mises à jour — pipeline réinitialisé !');
      if (onUpdated) onUpdated();
    } catch (e) {
      setError(e.message);
      toast.error('Erreur lors de la mise à jour');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--accent-blue)' }}>
            <RefreshCw size={20} style={{ color: 'var(--accent-blue)' }} />
          </div>
          <div>
            <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
              Mise à jour des données GMAO
            </h3>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Importez un fichier de nouvelles interventions — elles seront ajoutées au dataset existant
            </p>
          </div>
        </div>

      </div>

      {/* ── Données actuelles ── */}
      {currentDataset && (
        <div className="rounded-xl border p-4 space-y-2"
          style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-default)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-3"
            style={{ color: 'var(--text-muted)' }}>Données actuelles</p>
          <div className="flex flex-wrap gap-4">
            <Stat icon={<Database size={14}/>} label="Dataset" value={currentDataset.name} />
            <Stat icon={<FileText size={14}/>} label="Lignes" value={currentDataset.n_rows?.toLocaleString() ?? '—'} />
            <Stat icon={<Calendar size={14}/>} label="Période"
              value={currentDataset.period_start
                ? `${String(currentDataset.period_start).slice(0,10)} → ${String(currentDataset.period_end).slice(0,10)}`
                : '—'} />
          </div>
        </div>
      )}

      {/* ── Explication ── */}
      <div className="rounded-xl border p-4 text-sm space-y-2"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--accent-blue)', borderWidth: '1px' }}>
        <p className="font-semibold" style={{ color: 'var(--accent-blue)' }}>
          📋 Comment ça fonctionne ?
        </p>
        <ul className="space-y-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <li>• Exportez uniquement les <b>nouvelles interventions</b> depuis votre GMAO</li>
          <li>• Le fichier peut avoir le <b>nouveau format</b> (<code>date_declaration</code>, <code>equipment_code</code>...) ou <b>l'ancien format</b> (<code>WOWO_DECLARATION_DATE</code>...)</li>
          <li>• Le format est <b>détecté automatiquement</b> et les colonnes sont normalisées</li>
          <li>• Les nouvelles lignes sont <b>ajoutées</b> au failure existant (doublons supprimés)</li>
          <li>• Le pipeline est remis à zéro → relancez EDA → Features → Preprocessing</li>
        </ul>
      </div>

      {/* ── Zone de drop ── */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className="rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-all"
        style={{
          borderColor: dragging ? 'var(--accent-blue)' : file ? 'var(--success)' : 'var(--border-strong)',
          background: dragging ? 'var(--bg-elevated)' : 'var(--bg-card)',
        }}
      >
        <input ref={inputRef} type="file" accept=".csv" className="hidden"
          onChange={(e) => handleFile(e.target.files[0])} />

        {file ? (
          <div className="space-y-1">
            <CheckCircle2 size={32} className="mx-auto" style={{ color: 'var(--success)' }} />
            <p className="font-semibold text-sm" style={{ color: 'var(--success)' }}>{file.name}</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {(file.size / 1024).toFixed(0)} Ko — Cliquez pour changer
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload size={32} className="mx-auto" style={{ color: 'var(--text-muted)' }} />
            <p className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
              Glissez votre export GMAO ici
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              ou cliquez pour sélectionner un fichier CSV
            </p>
          </div>
        )}
      </div>

      {/* ── Bouton lancer ── */}
      <button
        onClick={handleUpdate}
        disabled={!file || loading || !datasetId}
        className="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
        style={{
          background: (!file || loading || !datasetId) ? 'var(--bg-elevated)' : 'var(--accent-blue)',
          color:      (!file || loading || !datasetId) ? 'var(--text-muted)'   : '#fff',
          cursor:     (!file || loading || !datasetId) ? 'not-allowed' : 'pointer',
          border: '1px solid transparent',
        }}
      >
        {loading
          ? <><Loader size={16} className="animate-spin" /> Mise à jour en cours...</>
          : <><RefreshCw size={16} /> Mettre à jour les données</>}
      </button>

      {/* ── Annuler le dernier ajout ── */}
      {canUndo && (
        <button
          onClick={handleUndo}
          disabled={undoing}
          className="w-full py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
          style={{
            background: 'var(--bg-elevated)',
            color:      'var(--accent-orange)',
            border:     '1px solid var(--accent-orange)',
            cursor:     undoing ? 'not-allowed' : 'pointer',
          }}
        >
          {undoing
            ? <><Loader size={16} className="animate-spin" /> Annulation...</>
            : <><RotateCcw size={16} /> Annuler le dernier ajout{undoInfo?.n_added ? ` (+${undoInfo.n_added.toLocaleString()} lignes)` : ''}</>}
        </button>
      )}


      {/* ── Résultat ── */}
      {result && (
        <div className="rounded-xl border p-4 space-y-4"
          style={{ background: 'var(--bg-elevated)', borderColor: 'var(--success)' }}>
          <div className="flex items-center gap-2">
            <CheckCircle2 size={18} style={{ color: 'var(--success)' }} />
            <p className="font-semibold text-sm" style={{ color: 'var(--success)' }}>
              Données mises à jour avec succès !
            </p>
          </div>

          {/* Compteurs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <CountBox label="Lignes existantes"  value={result.n_existing?.toLocaleString()}  color="var(--text-secondary)" />
            <CountBox label="Nouvelles lignes"   value={`+${result.n_new?.toLocaleString()}`} color="var(--accent-blue)" />
            <CountBox label="Doublons supprimés" value={result.n_duplicates?.toLocaleString() ?? '0'} color="var(--accent-orange)" />
            <CountBox label="Total final"        value={result.n_total?.toLocaleString()}     color="var(--success)" />
          </div>

          <div className="flex flex-wrap gap-4 pt-1">
            <Stat icon={<Calendar size={14}/>}  label="Nouvelle période"
              value={result.date_min ? `${result.date_min} → ${result.date_max}` : '—'} accent="success" />
            <Stat icon={<Database size={14}/>}  label="Format détecté"
              value={result.format === 'new' ? 'Nouveau format GMAO' : 'Format pipeline'} accent="success" />
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            ⚠️ Le pipeline a été remis à zéro. Relancez EDA → Features → Preprocessing pour re-entraîner.
          </p>
        </div>
      )}

      {/* ── Erreur ── */}
      {error && (
        <div className="rounded-xl border p-4 flex items-start gap-3"
          style={{ background: 'var(--bg-elevated)', borderColor: 'var(--error)' }}>
          <AlertCircle size={18} style={{ color: 'var(--error)' }} className="flex-shrink-0 mt-0.5" />
          <p className="text-sm" style={{ color: 'var(--error)' }}>{error}</p>
        </div>
      )}
    </div>
  );
}

function Stat({ icon, label, value, accent }) {
  return (
    <div className="flex items-center gap-2">
      <span style={{ color: accent ? `var(--${accent})` : 'var(--text-muted)' }}>{icon}</span>
      <div>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</p>
        <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{value}</p>
      </div>
    </div>
  );
}

function CountBox({ label, value, color }) {
  return (
    <div className="rounded-lg p-3 text-center"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border-default)' }}>
      <p className="text-lg font-bold font-mono" style={{ color }}>{value ?? '—'}</p>
      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{label}</p>
    </div>
  );
}
