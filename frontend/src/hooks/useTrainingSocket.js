/**
 * useTrainingSocket.js
 * Hook React pour la connexion WebSocket au backend FastAPI.
 * Reçoit les logs, métriques d'époque et résultats finaux.
 */
import { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE = 'ws://localhost:8000/ws';

export function useTrainingSocket(experimentId) {
  const wsRef    = useRef(null);
  const [logs,   setLogs]   = useState([]);
  const [epochs, setEpochs] = useState([]);   // historique { epoch, loss, val_loss, mae, val_mae }
  const [trials, setTrials] = useState([]);   // résultats des essais AutoML
  const [status, setStatus] = useState('idle'); // idle | connecting | running | completed | error
  const [result, setResult] = useState(null);

  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-200), { ts: Date.now(), text: msg }]);
  }, []);

  const connect = useCallback((expId) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_BASE}/${expId}`);
    wsRef.current = ws;
    setStatus('connecting');
    setLogs([]);
    setEpochs([]);
    setTrials([]);
    setResult(null);

    ws.onopen = () => {
      setStatus('running');
      addLog('🔌 Connexion WebSocket établie...');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'log':
            addLog(data.message);
            break;

          case 'status':
            setStatus(data.status);
            break;

          case 'epoch':
            setEpochs(prev => [...prev, {
              epoch:    data.epoch,
              loss:     data.loss,
              val_loss: data.val_loss,
              mae:      data.mae,
              val_mae:  data.val_mae,
            }]);
            addLog(
              `  Époque ${data.epoch}/${data.total} — ` +
              `Loss: ${data.loss.toFixed(5)} | Val: ${data.val_loss.toFixed(5)} | ` +
              `MAE: ${data.mae.toFixed(5)}`
            );
            break;

          case 'trial_start':
            addLog(`\n🔍 ${data.message}`);
            break;

          case 'trial_end':
            setTrials(prev => [...prev, {
              trial:       data.trial,
              avg_cv_loss: data.avg_cv_loss,
              duration:    data.duration,
            }]);
            addLog(`  ✅ ${data.message}`);
            break;

          case 'result':
            addLog(data.message);
            setResult(data);
            break;

          case 'completed':
            setStatus('completed');
            addLog(`\n🏁 Entraînement terminé ! R²=${data.r2} | MAE=${data.mae_hours}h`);
            break;

          case 'error':
            setStatus('error');
            addLog(`\n❌ Erreur : ${data.message}`);
            break;

          default:
            break;
        }
      } catch (_) {
        addLog(event.data);
      }
    };

    ws.onerror = () => {
      setStatus('error');
      addLog('❌ Erreur de connexion WebSocket');
    };

    ws.onclose = () => {
      addLog('🔌 Connexion fermée.');
    };
  }, [addLog]);

  // Connecter automatiquement si experimentId change
  useEffect(() => {
    if (experimentId) connect(experimentId);
    return () => wsRef.current?.close();
  }, [experimentId, connect]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    setStatus('idle');
  }, []);

  return { logs, epochs, trials, status, result, connect, disconnect };
}
