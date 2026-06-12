/**
 * UploadDatasetModal.jsx — Modal d'upload d'un nouveau dataset.
 *  - 2 inputs file (failure CSV + equipment CSV)
 *  - 1 input text (nom du dataset)
 *  - POST /api/datasets/upload (multipart/form-data)
 */
import { useState, useRef } from 'react';
import { X, Upload, FileText, AlertCircle, Loader } from 'lucide-react';
import toast from 'react-hot-toast';

const API = 'http://localhost:8000';

export default function UploadDatasetModal({ onClose, onCreated }) {
  const [name, setName]               = useState('');
  const [failureFile, setFailureFile] = useState(null);
  const [equipmentFile, setEquipmentFile] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);

  const failureInput   = useRef();
  const equipmentInput = useRef();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!name.trim())        return setError('Donne un nom au dataset.');
    if (!failureFile)        return setError('Sélectionne le fichier failure CSV.');
    if (!equipmentFile)      return setError('Sélectionne le fichier equipment CSV.');

    setLoading(true);
    const fd = new FormData();
    fd.append('name', name.trim());
    fd.append('failure_file',   failureFile);
    fd.append('equipment_file', equipmentFile);

    try {
      const res = await fetch(`${API}/api/datasets/upload`, {
        method: 'POST',
        body:   fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const ds = await res.json();
      toast.success(`Dataset "${ds.name}" créé (id #${ds.id})`);
      onCreated(ds);
    } catch (err) {
      setError(err.message);
      toast.error(`Échec upload : ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0, 0, 0, 0.55)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="relative rounded-2xl border w-full max-w-lg p-6"
        style={{
          background:  'var(--bg-elevated)',
          borderColor: 'var(--border-default)',
          boxShadow:   '0 20px 60px rgba(0,0,0,0.4)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 w-8 h-8 rounded-lg flex items-center justify-center transition-all"
          style={{ color: 'var(--text-muted)' }}
        >
          <X size={16} />
        </button>

        {/* Header */}
        <div className="flex items-center gap-2 mb-5">
          <Upload size={18} style={{ color: 'var(--brand-primary)' }} />
          <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
            Importer un dataset Cevital
          </h3>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Nom */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-widest block mb-1.5"
              style={{ color: 'var(--text-tertiary)' }}>
              Nom du dataset
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Ex : Cevital_2023"
              className="w-full px-3 py-2 rounded-lg border outline-none text-sm font-mono"
              style={{
                background:  'var(--bg-card)',
                borderColor: 'var(--border-default)',
                color:       'var(--text-primary)',
              }}
            />
          </div>

          {/* Failure CSV */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-widest block mb-1.5"
              style={{ color: 'var(--text-tertiary)' }}>
              Fichier failure CSV
            </label>
            <FileDrop
              file={failureFile}
              onPick={() => failureInput.current?.click()}
              onClear={() => setFailureFile(null)}
            />
            <input
              ref={failureInput}
              type="file"
              accept=".csv"
              onChange={(e) => setFailureFile(e.target.files?.[0] || null)}
              className="hidden"
            />
          </div>

          {/* Equipment CSV */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-widest block mb-1.5"
              style={{ color: 'var(--text-tertiary)' }}>
              Fichier equipment CSV
            </label>
            <FileDrop
              file={equipmentFile}
              onPick={() => equipmentInput.current?.click()}
              onClear={() => setEquipmentFile(null)}
            />
            <input
              ref={equipmentInput}
              type="file"
              accept=".csv"
              onChange={(e) => setEquipmentFile(e.target.files?.[0] || null)}
              className="hidden"
            />
          </div>

          {/* Erreur */}
          {error && (
            <div className="rounded-lg px-3 py-2 text-xs flex items-start gap-2 border"
              style={{
                background:  'var(--tint-error-bg)',
                color:       'var(--error)',
                borderColor: 'var(--error)',
              }}>
              <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
              <span className="whitespace-pre-wrap">{error}</span>
            </div>
          )}

          {/* Boutons */}
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-sm font-semibold border"
              style={{
                background:  'var(--bg-card)',
                borderColor: 'var(--border-default)',
                color:       'var(--text-tertiary)',
              }}>
              Annuler
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 border"
              style={{
                background:  loading ? 'var(--border-default)' : 'var(--brand-primary)',
                borderColor: loading ? 'var(--border-default)' : 'var(--brand-primary)',
                color:       'var(--bg-elevated)',
                cursor:      loading ? 'wait' : 'pointer',
              }}>
              {loading
                ? <><Loader size={14} className="animate-spin" /> Upload…</>
                : <><Upload size={14} /> Créer le dataset</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}


function FileDrop({ file, onPick, onClear }) {
  return (
    <button type="button" onClick={onPick}
      className="w-full rounded-lg px-3 py-3 flex items-center justify-between border text-sm text-left"
      style={{
        background:  'var(--bg-card)',
        borderColor: file ? 'var(--success)' : 'var(--border-default)',
        borderStyle: 'dashed',
        color:       'var(--text-secondary)',
      }}>
      <span className="flex items-center gap-2 truncate">
        <FileText size={14} style={{ color: file ? 'var(--success)' : 'var(--text-muted)' }} />
        <span className="font-mono text-xs truncate">
          {file ? file.name : 'Cliquer pour choisir un fichier…'}
        </span>
        {file && (
          <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>
            ({Math.round(file.size / 1024).toLocaleString()} ko)
          </span>
        )}
      </span>
      {file && (
        <button type="button" onClick={(e) => { e.stopPropagation(); onClear(); }}
          className="text-xs px-2 py-0.5 rounded ml-2"
          style={{ background: 'var(--bg-elevated)', color: 'var(--text-tertiary)' }}>
          ×
        </button>
      )}
    </button>
  );
}
