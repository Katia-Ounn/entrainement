/**
 * DemoPanel.jsx — Wrapper unique qui regroupe toutes les démos d'architectures.
 *
 * Architectures supportées :
 *  - RNN  (réseau récurrent simple)
 *  - LSTM (4 portes + cell state)
 *  - Transformer (encoder + decoder + multi-head attention)
 *  - GRU  (à venir)
 */
import { useState } from 'react';
import RNNDemoPanel         from './RNNDemoPanel';
import LSTMDemoPanel        from './LSTMDemoPanel';
import TransformerDemoPanel from './TransformerDemoPanel';

export default function DemoPanel() {
  const [arch, setArch] = useState('rnn');

  return (
    <div>
      <div style={{ display: arch === 'rnn' ? 'block' : 'none' }}>
        <RNNDemoPanel onSwitchTo={setArch}/>
      </div>
      <div style={{ display: arch === 'lstm' ? 'block' : 'none' }}>
        <LSTMDemoPanel
          onSwitchToRNN={() => setArch('rnn')}
          onSwitchTo={setArch}
        />
      </div>
      <div style={{ display: arch === 'transformer' ? 'block' : 'none' }}>
        <TransformerDemoPanel onSwitchTo={setArch}/>
      </div>
    </div>
  );
}