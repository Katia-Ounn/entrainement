import { useState, useEffect } from 'react';

export const useTrainingWS = (experimentId) => {
    const [status, setStatus] = useState('idle');
    const [metrics, setMetrics] = useState([]); // Pour le graphique de Loss
    const [finalResult, setFinalResult] = useState(null);

    useEffect(() => {
        if (!experimentId) return;

        const ws = new WebSocket(`ws://localhost:8000/ws/${experimentId}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'epoch_end') {
                // On ajoute la nouvelle ligne au graphique
                setMetrics(prev => [...prev, {
                    epoch: data.epoch,
                    loss: data.loss,
                    val_loss: data.val_loss,
                    mae: data.mae
                }]);
            } 
            else if (data.type === 'completed') {
                setStatus('completed');
                setFinalResult(data);
                ws.close();
            }
        };

        return () => ws.close();
    }, [experimentId]);

    return { metrics, status, finalResult };
};