/**
 * UpdateData.jsx — Mise à jour des données GMAO.
 *
 * Remplace "Fusion / Réentraîn." par une fonctionnalité de mise à jour :
 *   - L'utilisateur uploade un nouvel export GMAO complet (2023 + 2024 + ...)
 *   - Le backend détecte le format automatiquement (nouveau ou ancien)
 *   - Les données du dataset sont remplacées
 *   - Le pipeline est remis à zéro pour re-run
 */
import { useState, useRef } from 'react';
import { Upload, RefreshCw, CheckCircle2, AlertCircle, Calendar, Database, FileText, Loader } from 'lucide-react';
import toast from 'react-hot-toast';

const API = 'http://localhost:8000';

export default function UpdateData({ datasetId, currentDataset, onUpdated }) {
  const [file,      setFile]      = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);
  const [dragging,  setDragging]  = useState(false);
  const inputRef = useRef();

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
      toast.success('Données mises à jour !');
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
            Importez un nouvel export complet — les données existantes seront remplacées
          </p>
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
          <li>• Exportez l'historique complet depuis votre GMAO (2003 → aujourd'hui)</li>
          <li>• Le fichier peut avoir le <b>nouveau format</b> (<code>date_declaration</code>, <code>equipment_code</code>...) ou <b>l'ancien format</b> (<code>WOWO_DECLARATION_DATE</code>...)</li>
          <li>• Le format est <b>détecté automatiquement</b></li>
          <li>• Les données sont remplacées et le pipeline est remis à zéro pour re-run</li>
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

      {/* ── Résultat ── */}
      {result && (
        <div className="rounded-xl border p-4 space-y-3"
          style={{ background: 'var(--bg-elevated)', borderColor: 'var(--success)' }}>
          <div className="flex items-center gap-2">
            <CheckCircle2 size={18} style={{ color: 'var(--success)' }} />
            <p className="font-semibold text-sm" style={{ color: 'var(--success)' }}>
              Données mises à jour avec succès !
            </p>
          </div>
          <div className="flex flex-wrap gap-4 pt-1">
            <Stat icon={<FileText size={14}/>}  label="Lignes"   value={result.n_rows?.toLocaleString()} accent="success" />
            <Stat icon={<Calendar size={14}/>}  label="Nouvelle période"
              value={result.date_min ? `${result.date_min} → ${result.date_max}` : '—'} accent="success" />
            <Stat icon={<Database size={14}/>}  label="Format détecté"
              value={result.format === 'new' ? 'Nouveau format GMAO' : 'Format pipeline'} accent="success" />
          </div>
          <p className="text-xs pt-1" style={{ color: 'var(--text-muted)' }}>
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
