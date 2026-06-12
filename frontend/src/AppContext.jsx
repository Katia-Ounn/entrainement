/**
 * AppContext.jsx — État global partagé entre tous les panels
 * Persist dans localStorage pour survivre aux refreshs
 */
import {
  createContext, useContext, useState, useEffect,
  useCallback, useRef, useLayoutEffect,
} from 'react';

const AppContext = createContext(null);

// Helpers localStorage
const save = (key, val) => { try { localStorage.setItem(key, JSON.stringify(val)); } catch(_) {} };
const load = (key, def)  => { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : def; } catch(_) { return def; } };

export function AppProvider({ children }) {

  // ── 🎨 Theme switcher ───────────────────────────────────────
  // 'dark' (défaut) | 'light'. Persisté dans localStorage, appliqué via
  // data-theme="..." sur <html> (CSS variables index.css).
  //
  // 🐛 Fix Phase 6 : on lit la valeur synchrone (lazy init), on l'applique
  // direct dans useLayoutEffect (avant le paint, pas useEffect = après).
  // index.html a déjà posé le bon data-theme avant React → useEffect ne
  // FAIT que synchroniser localStorage et mettre à jour data-theme sur
  // changement, sans flash.
  const [theme, setTheme] = useState(() => {
    const t = load('cevital_theme', 'dark');
    return t === 'light' ? 'light' : 'dark';
  });

  useLayoutEffect(() => {
    document.documentElement.setAttribute(
      'data-theme', theme === 'light' ? 'light' : 'dark'
    );
  }, [theme]);

  useEffect(() => {
    save('cevital_theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  // ── 🔄 Re-train flow (Phase 4) ──────────────────────────────
  const [pendingRetrain, setPendingRetrain] = useState(null);
  const requestRetrain = useCallback((payload) => {
    setPendingRetrain(payload || null);
  }, []);
  const consumeRetrain = useCallback(() => {
    setPendingRetrain(null);
  }, []);

  // ── 🧭 Navigation programmatique entre onglets ──────────────
  // Tout composant peut faire `goToTab('prep')` pour basculer
  // l'utilisateur sur un autre onglet (lien "Modifier dans Prétraitement").
  const [requestedTab, setRequestedTab] = useState(null);
  const goToTab = useCallback((tabId) => setRequestedTab(tabId), []);
  const consumeRequestedTab = useCallback(() => setRequestedTab(null), []);

  // ── 📊 Préparation Données (Phase 2) ────────────────────────
  // Dataset Cevital actif + résultats cachés des 5 sous-onglets (RawEDA,
  // FeatureEngineering, FeaturesEDA, Preprocessing, MergeDatasets).
  // Le user peut changer de sous-onglet sans perdre les résultats.
  const [currentDatasetId, setCurrentDatasetId] = useState(() => load('cev_dataset_id', null));
  const [prepStepCompleted, setPrepStepCompleted] = useState(() => load('cev_prep_done', []));
  const [edaRawResult,      setEdaRawResult]      = useState(() => load('cev_eda_raw', null));
  const [featuresResult,    setFeaturesResult]    = useState(() => load('cev_features', null));
  const [edaFeatResult,     setEdaFeatResult]     = useState(() => load('cev_eda_feat', null));
  const [preprocResult,     setPreprocResult]     = useState(() => load('cev_preproc', null));

  // 🆕 Compteur de version BDD datasets — incrémenté après chaque mutation
  // (features_stream OK, preprocessing_stream OK, merge, upload, delete).
  // Les composants qui ont leur propre `datasets` local s'en servent comme
  // dépendance d'useEffect → re-fetch automatique → status à jour.
  const [datasetVersion, setDatasetVersion] = useState(0);
  const bumpDatasetVersion = useCallback(
    () => setDatasetVersion(v => v + 1),
    []
  );

  // 🆕 Features sélectionnées pour le modèle (Lot C — user toggle)
  // Default = les 9 features du notebook PFE_CHAMPION
  const DEFAULT_FEATURES = [
    'comp_level',
    'pannes_7j', 'pannes_30j', 'pannes_90j',
    'maint_7j',  'maint_30j',  'maint_90j',
    'DSLF',      'DSLM',
  ];
  const [selectedFeatures, setSelectedFeatures] = useState(
    () => load('cev_selected_features', DEFAULT_FEATURES)
  );

  useEffect(() => { save('cev_dataset_id',  currentDatasetId);  }, [currentDatasetId]);
  useEffect(() => { save('cev_prep_done',   prepStepCompleted); }, [prepStepCompleted]);
  useEffect(() => { save('cev_eda_raw',     edaRawResult);      }, [edaRawResult]);
  useEffect(() => { save('cev_features',    featuresResult);    }, [featuresResult]);
  useEffect(() => { save('cev_eda_feat',    edaFeatResult);     }, [edaFeatResult]);
  useEffect(() => { save('cev_preproc',     preprocResult);     }, [preprocResult]);
  useEffect(() => { save('cev_selected_features', selectedFeatures); }, [selectedFeatures]);

  const markPrepStep = useCallback((stepId) => {
    setPrepStepCompleted(prev => prev.includes(stepId) ? prev : [...prev, stepId]);
  }, []);

  const resetPrep = useCallback(() => {
    setCurrentDatasetId(null);
    setPrepStepCompleted([]);
    setEdaRawResult(null);
    setFeaturesResult(null);
    setEdaFeatResult(null);
    setPreprocResult(null);
    ['cev_dataset_id','cev_prep_done','cev_eda_raw','cev_features','cev_eda_feat','cev_preproc']
      .forEach(k => localStorage.removeItem(k));
  }, []);

  // Quand on change de dataset, on reset les caches d'étapes (résultats
  // précédents = ceux de l'ancien dataset, donc non valides).
  const selectDataset = useCallback((newId) => {
    if (newId !== currentDatasetId) {
      setEdaRawResult(null);
      setFeaturesResult(null);
      setEdaFeatResult(null);
      setPreprocResult(null);
      setPrepStepCompleted([]);
    }
    setCurrentDatasetId(newId);
  }, [currentDatasetId]);

  // ── Ingestion ──────────────────────────────────────────────
  const [ingestionStatus,   setIngestionStatus]   = useState(() => load('pdm_ingestion_status', 'idle'));
  const [ingestionResult,   setIngestionResult]   = useState(() => load('pdm_ingestion_result', null));
  const [ingestionPhases,   setIngestionPhases]   = useState(() => load('pdm_ingestion_phases', {}));
  const [ingestionDone,     setIngestionDone]     = useState(() => load('pdm_ingestion_done', []));
  const [filesData,         setFilesData]         = useState(() => load('pdm_files_data', {}));
  const [validation,        setValidation]        = useState(() => load('pdm_validation', null));
  const [mergeSteps,        setMergeSteps]        = useState(() => load('pdm_merge_steps', []));
  const [featSteps,         setFeatSteps]         = useState(() => load('pdm_feat_steps', []));
  const [tensorSteps,       setTensorSteps]       = useState(() => load('pdm_tensor_steps', []));

  // ── Entraînement ───────────────────────────────────────────
  const [currentEpochCtx,   setCurrentEpochCtx]   = useState(null);
  const [trainingStatus,    setTrainingStatus]    = useState(() => load('pdm_training_status', 'idle'));
  const [epochData,         setEpochData]         = useState(() => load('pdm_epoch_data', []));
  const [trialData,         setTrialData]         = useState(() => load('pdm_trial_data', []));
  const [trainingResult,    setTrainingResult]    = useState(() => load('pdm_training_result', null));
  const [predictions,       setPredictions]       = useState(() => load('pdm_predictions', null));
  const [lastExpId,         setLastExpId]         = useState(() => load('pdm_last_exp_id', null));
  const [trainingLogs,      setTrainingLogs]      = useState(() => load('pdm_training_logs', []));

  // ── Persist automatique ────────────────────────────────────
  useEffect(() => { save('pdm_ingestion_status', ingestionStatus); },   [ingestionStatus]);
  useEffect(() => { save('pdm_ingestion_result', ingestionResult); },   [ingestionResult]);
  useEffect(() => { save('pdm_ingestion_phases', ingestionPhases); },   [ingestionPhases]);
  useEffect(() => { save('pdm_ingestion_done',   ingestionDone); },     [ingestionDone]);
  useEffect(() => { save('pdm_files_data',       filesData); },         [filesData]);
  useEffect(() => { save('pdm_validation',       validation); },        [validation]);
  useEffect(() => { save('pdm_merge_steps',      mergeSteps); },        [mergeSteps]);
  useEffect(() => { save('pdm_feat_steps',       featSteps); },         [featSteps]);
  useEffect(() => { save('pdm_tensor_steps',     tensorSteps); },       [tensorSteps]);
  useEffect(() => { save('pdm_training_status',  trainingStatus); },    [trainingStatus]);
  useEffect(() => { save('pdm_epoch_data',       epochData); },         [epochData]);
  useEffect(() => { save('pdm_trial_data',        trialData); },        [trialData]);
  useEffect(() => { save('pdm_training_result',  trainingResult); },    [trainingResult]);
  useEffect(() => { save('pdm_predictions',      predictions); },       [predictions]);
  useEffect(() => { save('pdm_last_exp_id',      lastExpId); },         [lastExpId]);
  useEffect(() => { save('pdm_training_logs',    trainingLogs.slice(-100)); }, [trainingLogs]);

  // ── Actions ingestion ──────────────────────────────────────
  const resetIngestion = useCallback(() => {
    setIngestionStatus('idle');
    setIngestionResult(null);
    setIngestionPhases({});
    setIngestionDone([]);
    setFilesData({});
    setValidation(null);
    setMergeSteps([]);
    setFeatSteps([]);
    setTensorSteps([]);
    ['pdm_ingestion_status','pdm_ingestion_result','pdm_ingestion_phases',
     'pdm_ingestion_done','pdm_files_data','pdm_validation',
     'pdm_merge_steps','pdm_feat_steps','pdm_tensor_steps'
    ].forEach(k => localStorage.removeItem(k));
  }, []);

  // ── Actions entraînement ───────────────────────────────────
  const resetTraining = useCallback(() => {
    setTrainingStatus('idle');
    setEpochData([]);
    setTrialData([]);
    setTrainingResult(null);
    setPredictions(null);
    setLastExpId(null);
    setTrainingLogs([]);
    ['pdm_training_status','pdm_epoch_data','pdm_trial_data',
     'pdm_training_result','pdm_predictions','pdm_last_exp_id','pdm_training_logs'
    ].forEach(k => localStorage.removeItem(k));
  }, []);

  const addLog = useCallback((msg) => {
    setTrainingLogs(prev => [...prev.slice(-300), { ts: Date.now(), text: msg }]);
  }, []);

  const addEpoch = useCallback((ep) => {
    setEpochData(prev => [...prev, ep]);
  }, []);

  const addTrial = useCallback((trial) => {
    setTrialData(prev => [...prev, trial]);
  }, []);

  // ── WebSocket entraînement — vit dans le contexte ────────────
  const wsRef = useRef(null);

  const connectTrainingWS = useCallback((expId) => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://localhost:8000/ws/${expId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog('🔌 Connexion WebSocket établie...');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch(data.type) {
          case 'log':
            addLog(data.message);
            break;
          case 'epoch':
            const ep = {
              epoch: data.epoch, total: data.total,
              loss: data.loss, val_loss: data.val_loss,
              mae: data.mae, val_mae: data.val_mae,
              elapsed: data.elapsed,
            };
            addEpoch(ep);
            setCurrentEpochCtx(ep);
            break;
          case 'trial_start':
            addLog(`\n🔍 Essai ${data.trial}/${data.total} en cours...`);
            break;
          case 'trial_end':
            addTrial({ trial: data.trial, avg_cv_loss: data.avg_cv_loss, duration: data.duration });
            addLog(`  ✅ Essai ${data.trial} terminé — CV Loss: ${data.avg_cv_loss?.toFixed(5)}`);
            break;
          case 'result':
            setTrainingResult(data);
            addLog(data.message);
            break;
          case 'completed':
            setTrainingStatus('completed');
            if (data.predictions) setPredictions(data.predictions);
            addLog(`\n🏁 Terminé ! R²=${data.r2?.toFixed(4)} | MAE=${data.mae_hours?.toFixed(1)}h`);
            ws.close();
            break;
          case 'error':
            setTrainingStatus('error');
            addLog(`\n❌ Erreur : ${data.message}`);
            ws.close();
            break;
        }
      } catch(_) {}
    };

    ws.onerror = () => addLog('❌ Erreur WebSocket');
    ws.onclose = () => addLog('🔌 Connexion fermée.');
  }, [addLog, addEpoch, addTrial]);

  const disconnectTrainingWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
  }, []);

  return (
    <AppContext.Provider value={{
      // 🎨 Theme
      theme, setTheme, toggleTheme,

      // 🔄 Re-train (Phase 4)
      pendingRetrain, requestRetrain, consumeRetrain,

      // 🧭 Navigation programmatique
      requestedTab, goToTab, consumeRequestedTab,

      // 📊 Préparation Données
      currentDatasetId,  setCurrentDatasetId, selectDataset,
      prepStepCompleted, setPrepStepCompleted, markPrepStep, resetPrep,
      edaRawResult,      setEdaRawResult,
      featuresResult,    setFeaturesResult,
      edaFeatResult,     setEdaFeatResult,
      preprocResult,     setPreprocResult,
      selectedFeatures,  setSelectedFeatures, DEFAULT_FEATURES,
      datasetVersion,    bumpDatasetVersion,

      // Ingestion
      ingestionStatus, setIngestionStatus,
      ingestionResult, setIngestionResult,
      ingestionPhases, setIngestionPhases,
      ingestionDone,   setIngestionDone,
      filesData,       setFilesData,
      validation,      setValidation,
      mergeSteps,      setMergeSteps,
      featSteps,       setFeatSteps,
      tensorSteps,     setTensorSteps,
      resetIngestion,

      // Entraînement
      trainingStatus,  setTrainingStatus,
      epochData,       setEpochData,
      trialData,       setTrialData,
      trainingResult,  setTrainingResult,
      predictions,     setPredictions,
      lastExpId,       setLastExpId,
      trainingLogs,    setTrainingLogs,
      addLog, addEpoch, addTrial,
      resetTraining,
      currentEpochCtx, setCurrentEpochCtx,
      connectTrainingWS, disconnectTrainingWS,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside AppProvider');
  return ctx;
}