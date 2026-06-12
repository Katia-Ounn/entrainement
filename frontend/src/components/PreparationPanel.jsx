/**
 * PreparationPanel.jsx — Container Phase 2 : stepper à 5 sous-onglets.
 *
 *   1. EDA Brute            (RawEDA)
 *   2. Feature Engineering  (FeatureEngineering)
 *   3. EDA Features         (FeaturesEDA)
 *   4. Prétraitement        (Preprocessing)
 *   5. Fusion / Réentr.     (MergeDatasets)
 *
 * Règles de navigation :
 *  - L'étape N+1 est désactivée tant que N n'est pas marqué "completed"
 *    (sauf "merge" qui est utilitaire, toujours accessible).
 *  - Quand on complète N, on switche automatiquement vers N+1.
 *  - L'état de chaque étape vit dans AppContext (persistance localStorage).
 *
 * Tout est stylé via les variables CSS — palette CEVITAL dark/light propre.
 */
import { useState, useEffect, useMemo } from 'react';
import {
  BarChart3, Cog, TrendingUp, Layers, GitMerge,
  Upload, Database, CheckCircle2, Lock, ChevronRight,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../AppContext';
import UploadDatasetModal  from './prep/UploadDatasetModal';
import DatasetSelector     from './prep/DatasetSelector';
import RawEDA              from './prep/RawEDA';
import FeatureEngineering  from './prep/FeatureEngineering';
import FeaturesEDA         from './prep/FeaturesEDA';
import Preprocessing       from './prep/Preprocessing';
import MergeDatasets       from './prep/MergeDatasets';

const API = 'http://localhost:8000';

// Ordre des sous-onglets + métadonnées
const SUBTABS = [
  { id: 'raw_eda',       label: 'EDA Brute',           icon: BarChart3,   step: 1, gate: null         },
  { id: 'features',      label: 'Feature Engineering', icon: Cog,         step: 2, gate: 'raw_eda'    },
  { id: 'features_eda',  label: 'EDA Features',        icon: TrendingUp,  step: 3, gate: 'features'   },
  { id: 'preprocessing', label: 'Prétraitement',       icon: Layers,      step: 4, gate: 'features_eda' },
  { id: 'merge',         label: 'Fusion / Réentraîn.', icon: GitMerge,    step: 5, gate: null         },
];


export default function PreparationPanel() {
  const {
    currentDatasetId, selectDataset,
    prepStepCompleted, datasetVersion,
  } = useApp();

  const [activeSubtab, setActiveSubtab] = useState('raw_eda');
  const [datasets, setDatasets]         = useState([]);
  const [showUpload, setShowUpload]     = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(false);

  // ─── Récupération de la liste des datasets ─────────────────
  const fetchDatasets = async () => {
    setLoadingDatasets(true);
    try {
      const res  = await fetch(`${API}/api/datasets`);
      const data = await res.json();
      setDatasets(Array.isArray(data) ? data : []);
    } catch (_) {
      setDatasets([]);
    } finally {
      setLoadingDatasets(false);
    }
  };
  // 🆕 Re-fetch sur changement de datasetVersion (après preprocessing OK)
  useEffect(() => { fetchDatasets(); }, [datasetVersion]);

  // Si aucun dataset sélectionné mais qu'au moins un existe, sélectionner le premier
  useEffect(() => {
    if (!currentDatasetId && datasets.length > 0) {
      selectDataset(datasets[0].id);
    }
  }, [datasets, currentDatasetId, selectDataset]);

  const currentDataset = useMemo(
    () => datasets.find(d => d.id === currentDatasetId),
    [datasets, currentDatasetId],
  );

  // ─── Logique de déblocage ──────────────────────────────────
  const isTabLocked = (tab) => {
    if (!currentDatasetId) return tab.id !== 'raw_eda';  // sans dataset, rien (sauf raw_eda qui propose l'upload)
    if (!tab.gate) return false;
    return !prepStepCompleted.includes(tab.gate);
  };

  const handleAdvance = (fromId) => {
    const idx = SUBTABS.findIndex(t => t.id === fromId);
    if (idx >= 0 && idx < SUBTABS.length - 1) {
      setActiveSubtab(SUBTABS[idx + 1].id);
    }
  };

  // Progression (X/5)
  const stepsValidated = SUBTABS.filter(t => prepStepCompleted.includes(t.id) || t.id === 'merge')
                                .filter(t => t.id !== 'merge')
                                .length;
  const progressPct = Math.round((stepsValidated / 4) * 100);

  // ─── Rendu ─────────────────────────────────────────────────
  return (
    <div className="space-y-5 theme-aware">
      {/* ───── Header : titre + dataset selector ───── */}
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
            📊 Préparation Données
          </h2>
          <p className="text-sm mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Cevital · EDA + Feature Engineering + Prétraitement LSTM/GRU
          </p>
        </div>

        <div className="flex items-center gap-2">
          <DatasetSelector
            datasets={datasets}
            currentId={currentDatasetId}
            onChange={(id) => selectDataset(id)}
            loading={loadingDatasets}
          />
          <button
            onClick={() => setShowUpload(true)}
            className="px-3 py-2 rounded-lg text-sm font-semibold flex items-center gap-1.5 border transition-all"
            style={{
              background:  'var(--brand-primary)',
              color:       'var(--bg-elevated)',
              borderColor: 'var(--brand-primary)',
            }}
          >
            <Upload size={14} />
            Importer dataset
          </button>
        </div>
      </div>

      {/* ───── Bandeau dataset actif ───── */}
      {currentDataset && (
        <div className="rounded-xl border px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs font-mono"
          style={{
            background:  'var(--bg-card)',
            borderColor: 'var(--border-default)',
          }}>
          <span className="flex items-center gap-1.5">
            <Database size={12} style={{ color: 'var(--brand-primary)' }} />
            <b style={{ color: 'var(--text-primary)' }}>{currentDataset.name}</b>
          </span>
          {currentDataset.n_rows > 0 && (
            <span style={{ color: 'var(--text-tertiary)' }}>
              📊 {currentDataset.n_rows.toLocaleString()} lignes · {currentDataset.n_composants} composants
            </span>
          )}
          {currentDataset.n_failures > 0 && (
            <span style={{ color: 'var(--text-tertiary)' }}>
              💥 {currentDataset.n_failures} pannes · 🔧 {currentDataset.n_maintenances} maint.
            </span>
          )}
          {currentDataset.period_start && (
            <span style={{ color: 'var(--text-tertiary)' }}>
              📅 {String(currentDataset.period_start).slice(0,10)} → {String(currentDataset.period_end).slice(0,10)}
            </span>
          )}
          <span className="ml-auto px-2 py-0.5 rounded font-semibold uppercase"
            style={{
              background: 'var(--bg-elevated)',
              color: currentDataset.status === 'preprocessed'
                ? 'var(--success)'
                : currentDataset.status === 'features_done'
                ? 'var(--accent-blue)'
                : 'var(--accent-orange)',
            }}>
            {currentDataset.status}
          </span>
        </div>
      )}

      {/* ───── Stepper + barre de progression ───── */}
      <div className="rounded-xl border p-3" style={{
        background:  'var(--bg-elevated)',
        borderColor: 'var(--border-default)',
      }}>
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
            Progression du pipeline
          </p>
          <p className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
            {stepsValidated}/4 étape{stepsValidated > 1 ? 's' : ''} validée{stepsValidated > 1 ? 's' : ''}
          </p>
        </div>

        {/* Barre de progression */}
        <div className="w-full h-1.5 rounded-full overflow-hidden mb-4"
             style={{ background: 'var(--bg-card-alt)' }}>
          <div className="h-full transition-all rounded-full"
               style={{
                 background: 'var(--brand-primary)',
                 width: `${progressPct}%`,
               }}/>
        </div>

        {/* Nav stepper */}
        <div className="flex flex-wrap gap-2">
          {SUBTABS.map(tab => {
            const Icon     = tab.icon;
            const isActive = activeSubtab === tab.id;
            const isDone   = prepStepCompleted.includes(tab.id);
            const isLocked = isTabLocked(tab);

            return (
              <button
                key={tab.id}
                onClick={() => { if (!isLocked) setActiveSubtab(tab.id); }}
                disabled={isLocked}
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold border transition-all"
                style={{
                  background:  isActive ? 'var(--bg-card)' : 'transparent',
                  color:       isLocked
                                 ? 'var(--text-muted)'
                                 : (isActive
                                     ? 'var(--brand-primary)'
                                     : (isDone ? 'var(--success)' : 'var(--text-secondary)')),
                  borderColor: isActive
                                 ? 'var(--brand-primary)'
                                 : (isDone ? 'var(--success)' : 'var(--border-default)'),
                  cursor:      isLocked ? 'not-allowed' : 'pointer',
                  opacity:     isLocked ? 0.5 : 1,
                }}
              >
                <span className="font-mono">{tab.step}.</span>
                {isLocked
                  ? <Lock size={12} />
                  : isDone
                    ? <CheckCircle2 size={14} />
                    : <Icon size={14} />}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ───── Vue active ───── */}
      <div className="rounded-2xl border p-5"
        style={{
          background:  'var(--bg-card)',
          borderColor: 'var(--border-default)',
        }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSubtab + (currentDatasetId || 'none')}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{    opacity: 0, x: -12 }}
            transition={{ duration: 0.22 }}
          >
            {!currentDatasetId && activeSubtab !== 'merge' && (
              <EmptyState onUpload={() => setShowUpload(true)} />
            )}

            {currentDatasetId && activeSubtab === 'raw_eda' && (
              <RawEDA
                datasetId={currentDatasetId}
                onCompleted={() => handleAdvance('raw_eda')}
              />
            )}
            {currentDatasetId && activeSubtab === 'features' && (
              <FeatureEngineering
                datasetId={currentDatasetId}
                onCompleted={() => { handleAdvance('features'); fetchDatasets(); }}
              />
            )}
            {currentDatasetId && activeSubtab === 'features_eda' && (
              <FeaturesEDA
                datasetId={currentDatasetId}
                onCompleted={() => handleAdvance('features_eda')}
              />
            )}
            {currentDatasetId && activeSubtab === 'preprocessing' && (
              <Preprocessing
                datasetId={currentDatasetId}
                onCompleted={() => { handleAdvance('preprocessing'); fetchDatasets(); }}
              />
            )}
            {activeSubtab === 'merge' && (
              <MergeDatasets
                datasetId={currentDatasetId}
                datasets={datasets}
                onMergeDone={() => { fetchDatasets(); }}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* ───── Modal upload ───── */}
      {showUpload && (
        <UploadDatasetModal
          onClose={() => setShowUpload(false)}
          onCreated={(ds) => {
            setShowUpload(false);
            fetchDatasets().then(() => selectDataset(ds.id));
          }}
        />
      )}
    </div>
  );
}


// ─── Sous-composant : empty state quand aucun dataset ──────────
function EmptyState({ onUpload }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
        style={{
          background: 'var(--bg-elevated)',
          border: '2px dashed var(--border-strong)',
        }}>
        <Database size={28} style={{ color: 'var(--brand-primary)' }} />
      </div>
      <h3 className="text-lg font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
        Aucun dataset chargé
      </h3>
      <p className="text-sm mb-5 max-w-md" style={{ color: 'var(--text-tertiary)' }}>
        Pour démarrer, importe un dataset Cevital : deux fichiers CSV (failure + equipment)
        + un nom. La plateforme se charge du reste (EDA, features, prétraitement).
      </p>
      <button
        onClick={onUpload}
        className="px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2 transition-all"
        style={{
          background: 'var(--brand-primary)',
          color:      'var(--bg-elevated)',
          border:     '1px solid var(--brand-primary)',
        }}>
        <Upload size={16} />
        Importer un dataset
        <ChevronRight size={14} />
      </button>
    </div>
  );
}
