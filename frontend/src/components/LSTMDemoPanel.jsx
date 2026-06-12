/**
 * LSTMDemoPanel.jsx — Démonstration pédagogique LSTM
 * Style identique à RNNDemoPanel mais avec :
 *  - Cellule LSTM compacte avec 4 portes visibles
 *  - Section spéciale "Anatomie d'une cellule LSTM"
 *  - Forward avec les 4 portes pas-à-pas
 *  - Cell state C_t (mémoire long terme) visualisé séparément
 *  - Loss + Backprop multi-portes
 */
import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import {
  Box, Play, Pause, RotateCcw, Zap, AlertTriangle, Cpu,
  BookOpen, Factory, Settings, Sparkles, ArrowRight,
  Target, Calculator, Brain, Lock, ChevronRight,
  Eye, Filter, Database, GitBranch, Layers
} from 'lucide-react';

const API = 'http://localhost:8000';

// ─── Couleurs des 4 portes LSTM ──────────────────────────────
const GATE_COLORS = {
  forget:    '#f06292',  // rose : oublier
  input:     'var(--accent-blue)',  // bleu : ajouter
  candidate: 'var(--accent-green)',  // vert : nouvelles infos
  output:    'var(--accent-orange)',  // orange : sortir
  cell:      'var(--accent-purple)',  // violet : cell state C
  hidden:    'var(--success)',  // vert foncé : hidden h
};

const LAYER_COLORS = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-orange)'];

// ─── Helpers ──────────────────────────────────────────────────
function valueColor(v, maxAbs = 1) {
  const ratio = Math.max(-1, Math.min(1, v / maxAbs));
  if (ratio >= 0) return `rgba(79, 195, 247, ${0.15 + ratio * 0.55})`;
  return `rgba(240, 98, 146, ${0.15 + Math.abs(ratio) * 0.55})`;
}
function valueText(v, maxAbs = 1) {
  if (Math.abs(v) < maxAbs * 0.1) return 'var(--text-tertiary)';
  return v >= 0 ? 'var(--accent-blue)' : '#f06292';
}

// ═══════════════════════════════════════════════════════════════
// Matrice 2D
// ═══════════════════════════════════════════════════════════════
function Matrix({ data, rowLabels, colLabels, color = 'var(--accent-blue)', maxAbs = null,
                  decimals = 3, small = false, title, subtitle, highlight = false }) {
  if (!data || !data.length) return null;
  const rows = Array.isArray(data[0]) ? data : [data];
  const flat = rows.flat().map(Math.abs);
  const eff = maxAbs || Math.max(...flat, 0.001);
  const cellW = small ? 44 : 54;
  const cellH = small ? 22 : 26;

  return (
    <div className="inline-block">
      {title && (
        <div className="mb-1 flex items-baseline gap-2">
          <span className="text-sm font-semibold" style={{ color }}>{title}</span>
          {subtitle && <span className="text-xs" style={{ color:'var(--text-muted)' }}>{subtitle}</span>}
          <span className="text-xs font-mono ml-1" style={{ color:'var(--text-muted)' }}>
            [{rows.length}×{rows[0].length}]
          </span>
        </div>
      )}
      <div className="relative inline-block"
        style={highlight ? { padding: 4, background: `${color}15`, borderRadius: 6 } : {}}>
        <div style={{
          position:'absolute', left:-6, top:0, bottom:0, width:6,
          borderLeft:`2px solid ${color}`,
          borderTop:`2px solid ${color}`, borderBottom:`2px solid ${color}`,
        }}/>
        <div style={{
          position:'absolute', right:-6, top:0, bottom:0, width:6,
          borderRight:`2px solid ${color}`,
          borderTop:`2px solid ${color}`, borderBottom:`2px solid ${color}`,
        }}/>
        <div className="flex">
          {rowLabels && (
            <div className="flex flex-col mr-1 justify-center">
              {rowLabels.map((lab, i) => (
                <div key={i} className="flex items-center justify-end pr-2 text-xs font-mono"
                  style={{ height:cellH, color:'var(--text-tertiary)', minWidth:42 }}>{lab}</div>
              ))}
            </div>
          )}
          <div>
            {colLabels && (
              <div className="flex">
                {colLabels.map((lab, j) => (
                  <div key={j} className="text-center text-xs font-mono pb-1"
                    style={{ width:cellW, color:'var(--text-muted)' }}>{lab}</div>
                ))}
              </div>
            )}
            {rows.map((row, i) => (
              <div key={i} className="flex">
                {row.map((v, j) => (
                  <div key={j} className="flex items-center justify-center text-xs font-mono border"
                    style={{
                      width:cellW, height:cellH,
                      background: valueColor(v, eff),
                      color: valueText(v, eff),
                      borderColor:'var(--bg-card-alt)', borderWidth:0.5,
                      fontWeight: Math.abs(v) > eff * 0.5 ? 'bold' : 'normal',
                    }}>
                    {typeof v === 'number' ? v.toFixed(decimals) : v}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// LSTMCellBox : la cellule LSTM compacte avec ses 4 portes
// ═══════════════════════════════════════════════════════════════
function LSTMCellBox({ config, weights, slices, forward_steps, prediction, backward,
                       activeLayer = -1, activeT = 0, activeGate = null, mode = 'idle', sceneId = '' }) {
  if (!config || !weights) return null;
  const { layers, n_features, batch_size, seq_length } = config;
  const n_layers = layers.length;
  const W = 1180, H = 600;

  const queueX  = 30;
  const inputX  = 280;
  const firstLayerX = 540;
  const layerXs = layers.map((_, L) => firstLayerX + L * 220);
  const lastLayerX = layerXs[n_layers - 1];
  const outputX = lastLayerX + 240;

  const cy = 220;
  const featNames = ['volt', 'rotate', 'pres', 'vibr'];
  const seqLabels = Array.from({ length: batch_size }, (_, i) => String.fromCharCode(65 + i));

  // Récupérer les valeurs actuelles pour le zoom
  const stepData = forward_steps?.[Math.max(0, activeLayer)]?.[Math.max(0, activeT)];
  const f_t = stepData?.f_t?.[0] || [];
  const i_t = stepData?.i_t?.[0] || [];
  const C_tilde = stepData?.C_tilde?.[0] || [];
  const o_t = stepData?.o_t?.[0] || [];
  const C_t = stepData?.C_t?.[0] || [];
  const H_t = stepData?.H_t?.[0] || [];

  const showForwardFlow = mode === 'forward' && activeLayer >= 0;
  const showBackpropFlow = mode === 'backprop';

  const narrationText = (() => {
    if (mode === 'idle')   return '⏸ Cliquez "Suivant" ou lancez l\'animation pour démarrer';
    if (mode === 'predict') return `🎯 Le dernier état caché H_${seq_length} est multiplié par W_out → on obtient la prédiction ŷ`;
    if (mode === 'backprop') return `🔄 L'erreur remonte à travers les 4 portes (forget, input, candidate, output)`;
    if (mode === 'forward') {
      if (activeGate === 'forget')    return `🔴 Forget gate : décide quoi OUBLIER de la mémoire C_${activeT}`;
      if (activeGate === 'input')     return `🔵 Input gate : décide quoi AJOUTER à la mémoire`;
      if (activeGate === 'candidate') return `🟢 Candidate : nouvelles informations potentielles`;
      if (activeGate === 'output')    return `🟠 Output gate : décide ce qui sort comme h_${activeT+1}`;
      if (activeGate === 'cell')      return `🟣 Cell state : C_t = forget * C_${activeT} + input * candidate`;
      if (activeGate === 'hidden')    return `🟢 Hidden state : h_t = output * tanh(C_t)`;
      return `🔵 Couche ${activeLayer + 1} · temps t=${activeT + 1} en cours...`;
    }
    return '';
  })();

  return (
    <div className="rounded-xl border-2 overflow-hidden"
      style={{ borderColor: 'var(--accent-purple)', background: 'linear-gradient(135deg, var(--bg-base), var(--tint-purple-bg))' }}>
      <div className="flex items-center justify-between px-4 py-2.5"
        style={{ background:'var(--tint-purple-bg)', borderBottom:'1px solid #ce93d840' }}>
        <div className="flex items-center gap-2">
          <Brain size={15} style={{ color:'var(--accent-purple)' }} />
          <span className="text-sm font-bold" style={{ color:'var(--accent-purple)' }}>
            🧬 BOÎTE LSTM — {n_layers} couche{n_layers>1?'s':''}
            {mode === 'forward' && ` · t=${activeT + 1}/${seq_length}`}
            {mode === 'predict' && ` · 🎯 PRÉDICTION`}
            {mode === 'backprop' && ` · ← BACKPROP`}
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color:'var(--text-muted)' }}>
          {weights.layers.reduce((s, L) => {
            const u = L.units, d = L.input_dim;
            return s + 4 * (d * u + u * u + u);
          }, 0) + weights.W_out.length} paramètres
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 700 }}>
        <defs>
          <pattern id="lstmgrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--bg-card)" strokeWidth="0.4"/>
          </pattern>
          <marker id="arrow-lstm" markerWidth="6" markerHeight="6" refX="5" refY="3"
            orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L6,3 z" fill="var(--text-tertiary)"/>
          </marker>
        </defs>
        <rect width={W} height={H} fill="url(#lstmgrid)" />

        {/* ═══════════ ENTRÉE (file de tranches) ═══════════ */}
        <text x={queueX + 60} y={20} textAnchor="middle" fontSize={11}
          fill="var(--accent-purple)" fontFamily="monospace" fontWeight="bold">📦 ENTRÉE</text>
        <text x={queueX + 60} y={34} textAnchor="middle" fontSize={9}
          fill="var(--text-muted)" fontFamily="monospace">{seq_length} tranches</text>

        {Array.from({ length: seq_length }).map((_, t) => {
          const isActive = activeT === t && mode === 'forward';
          const isPast   = activeT > t && mode === 'forward';
          const baseY    = 50 + t * 75;
          const sliceData = slices?.[t];

          return (
            <motion.g key={t}
              animate={{ scale: isActive ? 1.05 : 1, opacity: isPast ? 0.3 : 1 }}>
              <rect x={queueX} y={baseY} width={120} height={56} rx={6}
                fill={isActive ? '#1a3a5c' : isPast ? 'var(--bg-base)' : 'var(--bg-card)'}
                stroke={isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--border-strong)'}
                strokeWidth={isActive ? 2 : 1}
              />
              {sliceData && sliceData.slice(0, batch_size).map((row, i) => (
                row.slice(0, n_features).map((v, j) => (
                  <rect key={`m-${t}-${i}-${j}`}
                    x={queueX + 8 + j * 11} y={baseY + 28 + i * 5}
                    width={9} height={4}
                    fill={valueColor(v * 2 - 1, 1)}
                    opacity={isPast ? 0.4 : 1}
                  />
                ))
              ))}
              <text x={queueX + 60} y={baseY + 16} textAnchor="middle"
                fontSize={11} fontFamily="monospace" fontWeight="bold"
                fill={isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--text-tertiary)'}>
                X_{t + 1} {isActive && '⬅'} {isPast && '✓'}
              </text>
            </motion.g>
          );
        })}

        {/* ═══════════ INPUT ═══════════ */}
        <text x={inputX} y={20} textAnchor="middle" fontSize={11} fill="var(--accent-blue)"
          fontFamily="monospace" fontWeight="bold">INPUT</text>
        <text x={inputX} y={34} textAnchor="middle" fontSize={9} fill="var(--text-muted)"
          fontFamily="monospace">{n_features} features</text>

        {Array.from({ length: n_features }).map((_, i) => {
          const y = cy - 60 + i * 30;
          return (
            <motion.g key={`in-${i}`}
              animate={{ scale: showForwardFlow && activeLayer === 0 ? [1, 1.15, 1] : 1 }}
              transition={{ duration: 0.6, delay: i * 0.05 }}
            >
              <rect x={inputX - 22} y={y - 11} width={44} height={22} rx={3}
                fill={showForwardFlow && activeLayer === 0 ? '#1a4a7c' : '#1a3a5c'}
                stroke="var(--accent-blue)"
                strokeWidth={showForwardFlow && activeLayer === 0 ? 1.5 : 1}/>
              <text x={inputX} y={y + 4} textAnchor="middle" fontSize={10}
                fill="var(--accent-blue)" fontFamily="monospace">
                {featNames[i] || `x${i+1}`}
              </text>
            </motion.g>
          );
        })}

        {/* Flèche INPUT → couche LSTM */}
        {showForwardFlow && activeLayer === 0 && (
          <line x1={inputX + 22} y1={cy} x2={firstLayerX - 90} y2={cy}
            stroke="var(--accent-blue)" strokeWidth={1.5} markerEnd="url(#arrow-lstm)" opacity={0.7}/>
        )}

        {/* ═══════════ CELLULE LSTM (par couche) avec 4 portes ═══════════ */}
        {layerXs.map((x, L) => {
          const isLayerActive = activeLayer === L && showForwardFlow;
          const layerColor = LAYER_COLORS[L];
          // Boîte de cellule
          const cellW = 180, cellH = 200;
          const cellLeft = x - cellW / 2;
          const cellTop  = cy - cellH / 2;

          return (
            <g key={`cell-${L}`}>
              <text x={x} y={20} textAnchor="middle" fontSize={11}
                fill={layerColor} fontFamily="monospace" fontWeight="bold">
                COUCHE LSTM {L + 1}
              </text>
              <text x={x} y={34} textAnchor="middle" fontSize={9}
                fill="var(--text-muted)" fontFamily="monospace">{layers[L]} cellules</text>

              {/* Boîte externe de la cellule */}
              <rect x={cellLeft} y={cellTop} width={cellW} height={cellH} rx={10}
                fill={isLayerActive ? `${layerColor}10` : 'var(--bg-card)'}
                stroke={layerColor}
                strokeWidth={isLayerActive ? 2 : 1}
              />

              {/* 4 portes à l'intérieur */}
              {/* FORGET */}
              <motion.g
                animate={{ scale: activeGate === 'forget' && isLayerActive ? [1, 1.2, 1] : 1 }}>
                <circle cx={cellLeft + 35} cy={cellTop + 40} r={18}
                  fill={activeGate === 'forget' && isLayerActive ? `${GATE_COLORS.forget}50` : 'var(--bg-card)'}
                  stroke={GATE_COLORS.forget}
                  strokeWidth={activeGate === 'forget' && isLayerActive ? 2.5 : 1.2}/>
                <text x={cellLeft + 35} y={cellTop + 36} textAnchor="middle" fontSize={9}
                  fill={GATE_COLORS.forget} fontFamily="monospace" fontWeight="bold">σ</text>
                <text x={cellLeft + 35} y={cellTop + 49} textAnchor="middle" fontSize={8}
                  fill={GATE_COLORS.forget} fontFamily="monospace">forget</text>
              </motion.g>

              {/* INPUT */}
              <motion.g
                animate={{ scale: activeGate === 'input' && isLayerActive ? [1, 1.2, 1] : 1 }}>
                <circle cx={cellLeft + 80} cy={cellTop + 40} r={18}
                  fill={activeGate === 'input' && isLayerActive ? `${GATE_COLORS.input}50` : 'var(--bg-card)'}
                  stroke={GATE_COLORS.input}
                  strokeWidth={activeGate === 'input' && isLayerActive ? 2.5 : 1.2}/>
                <text x={cellLeft + 80} y={cellTop + 36} textAnchor="middle" fontSize={9}
                  fill={GATE_COLORS.input} fontFamily="monospace" fontWeight="bold">σ</text>
                <text x={cellLeft + 80} y={cellTop + 49} textAnchor="middle" fontSize={8}
                  fill={GATE_COLORS.input} fontFamily="monospace">input</text>
              </motion.g>

              {/* CANDIDATE */}
              <motion.g
                animate={{ scale: activeGate === 'candidate' && isLayerActive ? [1, 1.2, 1] : 1 }}>
                <circle cx={cellLeft + 125} cy={cellTop + 40} r={18}
                  fill={activeGate === 'candidate' && isLayerActive ? `${GATE_COLORS.candidate}50` : 'var(--bg-card)'}
                  stroke={GATE_COLORS.candidate}
                  strokeWidth={activeGate === 'candidate' && isLayerActive ? 2.5 : 1.2}/>
                <text x={cellLeft + 125} y={cellTop + 36} textAnchor="middle" fontSize={8}
                  fill={GATE_COLORS.candidate} fontFamily="monospace" fontWeight="bold">tanh</text>
                <text x={cellLeft + 125} y={cellTop + 50} textAnchor="middle" fontSize={7}
                  fill={GATE_COLORS.candidate} fontFamily="monospace">candidate</text>
              </motion.g>

              {/* OUTPUT */}
              <motion.g
                animate={{ scale: activeGate === 'output' && isLayerActive ? [1, 1.2, 1] : 1 }}>
                <circle cx={cellLeft + 145} cy={cellTop + 130} r={18}
                  fill={activeGate === 'output' && isLayerActive ? `${GATE_COLORS.output}50` : 'var(--bg-card)'}
                  stroke={GATE_COLORS.output}
                  strokeWidth={activeGate === 'output' && isLayerActive ? 2.5 : 1.2}/>
                <text x={cellLeft + 145} y={cellTop + 126} textAnchor="middle" fontSize={9}
                  fill={GATE_COLORS.output} fontFamily="monospace" fontWeight="bold">σ</text>
                <text x={cellLeft + 145} y={cellTop + 139} textAnchor="middle" fontSize={8}
                  fill={GATE_COLORS.output} fontFamily="monospace">output</text>
              </motion.g>

              {/* Cell state C (ligne supérieure horizontale = mémoire) */}
              <line x1={cellLeft + 8} y1={cellTop + 80}
                x2={cellLeft + cellW - 8} y2={cellTop + 80}
                stroke={GATE_COLORS.cell}
                strokeWidth={activeGate === 'cell' && isLayerActive ? 3 : 2}
                opacity={activeGate === 'cell' && isLayerActive ? 1 : 0.6}/>
              <text x={cellLeft + cellW - 12} y={cellTop + 73} fontSize={9}
                textAnchor="end"
                fill={GATE_COLORS.cell} fontFamily="monospace" fontWeight="bold">C_t</text>

              {/* Opérateur ⊗ (forget * C_prev) */}
              <circle cx={cellLeft + 35} cy={cellTop + 80} r={6}
                fill="none" stroke={GATE_COLORS.cell} strokeWidth={1.2}/>
              <text x={cellLeft + 35} y={cellTop + 84} textAnchor="middle" fontSize={9}
                fill={GATE_COLORS.cell}>×</text>

              {/* Opérateur ⊕ (+ input*candidate) */}
              <circle cx={cellLeft + 102} cy={cellTop + 80} r={6}
                fill="none" stroke={GATE_COLORS.cell} strokeWidth={1.2}/>
              <text x={cellLeft + 102} y={cellTop + 84} textAnchor="middle" fontSize={9}
                fill={GATE_COLORS.cell}>+</text>

              {/* tanh(C) → output */}
              <rect x={cellLeft + 100} y={cellTop + 120} width={26} height={20} rx={4}
                fill="var(--tint-success-bg)" stroke={GATE_COLORS.hidden} strokeWidth={1}/>
              <text x={cellLeft + 113} y={cellTop + 134} textAnchor="middle" fontSize={8}
                fill={GATE_COLORS.hidden} fontFamily="monospace">tanh</text>

              {/* Opérateur ⊗ (output * tanh(C)) */}
              <circle cx={cellLeft + 145} cy={cellTop + 165} r={6}
                fill="none" stroke={GATE_COLORS.hidden} strokeWidth={1.2}/>
              <text x={cellLeft + 145} y={cellTop + 169} textAnchor="middle" fontSize={9}
                fill={GATE_COLORS.hidden}>×</text>

              {/* Sortie h_t */}
              <text x={cellLeft + cellW - 12} y={cellTop + cellH - 8} fontSize={9}
                textAnchor="end"
                fill={GATE_COLORS.hidden} fontFamily="monospace" fontWeight="bold">h_t</text>

              {/* Connexions internes simplifiées (lignes fines) */}
              <line x1={cellLeft + 53} y1={cellTop + 40} x2={cellLeft + 62} y2={cellTop + 40}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>
              <line x1={cellLeft + 80} y1={cellTop + 58} x2={cellLeft + 102} y2={cellTop + 76}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>
              <line x1={cellLeft + 125} y1={cellTop + 58} x2={cellLeft + 105} y2={cellTop + 76}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>
              <line x1={cellLeft + 113} y1={cellTop + 88} x2={cellLeft + 113} y2={cellTop + 120}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>
              <line x1={cellLeft + 126} y1={cellTop + 130} x2={cellLeft + 142} y2={cellTop + 162}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>
              <line x1={cellLeft + 145} y1={cellTop + 148} x2={cellLeft + 145} y2={cellTop + 159}
                stroke="var(--border-strong)" strokeWidth={0.8} opacity={0.5}/>

              {/* Indication du nombre de cellules */}
              <text x={x} y={cellTop + cellH + 14} textAnchor="middle" fontSize={9}
                fill="var(--text-muted)" fontFamily="monospace">
                × {layers[L]} cellules en parallèle
              </text>
            </g>
          );
        })}

        {/* Connexions entre couches LSTM */}
        {layerXs.slice(0, -1).map((x1, L) => (
          <line key={`L-conn-${L}`}
            x1={x1 + 90} y1={cy} x2={layerXs[L + 1] - 90} y2={cy}
            stroke={activeLayer === L + 1 && showForwardFlow ? LAYER_COLORS[L + 1] : 'var(--border-default)'}
            strokeWidth={activeLayer === L + 1 && showForwardFlow ? 1.5 : 0.5}
            markerEnd="url(#arrow-lstm)"
            opacity={0.7}/>
        ))}

        {/* SORTIE ŷ */}
        <text x={outputX} y={20} textAnchor="middle" fontSize={11}
          fill="#f06292" fontFamily="monospace" fontWeight="bold">SORTIE</text>
        <text x={outputX} y={34} textAnchor="middle" fontSize={9}
          fill="var(--text-muted)" fontFamily="monospace">RUL prédit</text>

        <line x1={lastLayerX + 90} y1={cy} x2={outputX - 22} y2={cy}
          stroke={mode === 'predict' ? '#f06292' : 'var(--border-default)'}
          strokeWidth={mode === 'predict' ? 1.5 : 0.5}
          markerEnd="url(#arrow-lstm)"/>
        <g transform={`translate(${(lastLayerX + 90 + outputX) / 2}, ${cy - 18})`}>
          <rect x={-26} y={-12} width={52} height={20} rx={4}
            fill="var(--bg-base)" stroke="#f06292" strokeWidth={1}/>
          <text x={0} y={2} textAnchor="middle" fontSize={11}
            fill="#f06292" fontFamily="monospace" fontWeight="bold">W_out</text>
        </g>

        <motion.g animate={{ scale: mode === 'predict' ? [1, 1.3, 1] : 1 }}>
          <circle cx={outputX} cy={cy} r={22}
            fill={mode === 'predict' ? '#f0629260' : 'var(--bg-card)'}
            stroke="#f06292" strokeWidth={2}/>
          <text x={outputX} y={cy + 5} textAnchor="middle" fontSize={14}
            fill="#f06292" fontFamily="monospace" fontWeight="bold">ŷ</text>
        </motion.g>

        {mode === 'predict' && prediction && (
          <g transform={`translate(${outputX - 50}, ${cy + 35})`}>
            <rect width={100} height={28} rx={6} fill="var(--bg-card)" stroke="#f06292"/>
            <text x={50} y={12} textAnchor="middle" fontSize={9}
              fill="var(--text-tertiary)" fontFamily="monospace">RUL prédit</text>
            <text x={50} y={24} textAnchor="middle" fontSize={11}
              fill="#f06292" fontFamily="monospace" fontWeight="bold">
              {prediction.Y_pred_hours[0]?.toFixed(0)}h
            </text>
          </g>
        )}

        {/* Bandeau narration */}
        <rect x={0} y={H - 30} width={W} height={30}
          fill="var(--tint-purple-bg)" opacity={0.95}/>
        <line x1={0} y1={H - 30} x2={W} y2={H - 30}
          stroke={mode === 'forward' ? 'var(--accent-purple)' : mode === 'predict' ? '#f06292' : mode === 'backprop' ? '#f06292' : 'var(--border-strong)'}
          strokeWidth={2}/>
        <text x={W / 2} y={H - 11} textAnchor="middle" fontSize={11}
          fill="var(--text-primary)" fontFamily="monospace">
          {narrationText}
        </text>
      </svg>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// LSTMAnatomy : section spéciale "Anatomie d'une cellule LSTM"
// ═══════════════════════════════════════════════════════════════
function LSTMAnatomy() {
  const gates = [
    {
      name: 'Forget Gate',
      symbol: 'f_t',
      color: GATE_COLORS.forget,
      formula: String.raw`f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)`,
      icon: <Filter size={16}/>,
      role: 'Décide quoi OUBLIER',
      explanation: 'Pour chaque valeur de la mémoire C_{t-1}, retourne un nombre entre 0 (oublier complètement) et 1 (garder). La sigmoïde σ produit toujours [0, 1].',
      example: '0 = on oublie · 1 = on garde · 0.5 = on garde à moitié',
    },
    {
      name: 'Input Gate',
      symbol: 'i_t',
      color: GATE_COLORS.input,
      formula: String.raw`i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)`,
      icon: <Database size={16}/>,
      role: 'Décide quoi AJOUTER',
      explanation: 'Combien de la nouvelle info va-t-on ajouter à la mémoire ? Encore une sigmoïde [0, 1] qui filtre les informations entrantes.',
      example: '0 = on n\'ajoute rien · 1 = on ajoute tout',
    },
    {
      name: 'Candidate',
      symbol: 'C̃_t',
      color: GATE_COLORS.candidate,
      formula: String.raw`\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)`,
      icon: <Sparkles size={16}/>,
      role: 'Nouvelles infos potentielles',
      explanation: 'Calcule les NOUVELLES informations qui POURRAIENT être ajoutées. tanh produit des valeurs [-1, 1] pour augmenter ou diminuer la mémoire.',
      example: 'C̃_t = +0.8 → ajout positif fort',
    },
    {
      name: 'Output Gate',
      symbol: 'o_t',
      color: GATE_COLORS.output,
      formula: String.raw`o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)`,
      icon: <Eye size={16}/>,
      role: 'Décide ce qui SORT',
      explanation: 'Filtre la mémoire C_t pour décider quelle partie devient le hidden state h_t (sortie visible).',
      example: '0 = rien ne sort · 1 = on sort toute la mémoire',
    },
  ];

  return (
    <div className="space-y-3">
      {/* Vue d'ensemble */}
      <div className="rounded-xl p-4 border-2"
        style={{ background: 'linear-gradient(135deg, var(--tint-purple-bg), var(--bg-base))', borderColor: 'var(--accent-purple)' }}>
        <h4 className="text-sm font-bold flex items-center gap-2 mb-3" style={{ color: 'var(--accent-purple)' }}>
          <Brain size={14}/> Anatomie d'une cellule LSTM
        </h4>

        <p className="text-xs leading-relaxed mb-4" style={{ color: 'var(--text-secondary)' }}>
          Une cellule LSTM est plus complexe qu'un neurone RNN simple. Elle contient
          <b> 4 portes</b> qui contrôlent le flux d'informations dans une <b>mémoire long terme</b> appelée
          <b style={{color: GATE_COLORS.cell}}> cell state C_t</b>.
        </p>

        {/* Les 2 mémoires */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-base)', border: `1px solid ${GATE_COLORS.cell}40` }}>
            <p className="text-xs font-bold mb-1" style={{ color: GATE_COLORS.cell }}>
              🧠 Cell state C_t (mémoire long terme)
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              Une "autoroute" qui traverse la cellule. Les portes y ajoutent ou retirent
              des informations à chaque pas de temps.
            </p>
          </div>
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-base)', border: `1px solid ${GATE_COLORS.hidden}40` }}>
            <p className="text-xs font-bold mb-1" style={{ color: GATE_COLORS.hidden }}>
              👁️ Hidden state h_t (mémoire court terme)
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              C'est ce qui sort visiblement de la cellule à chaque pas de temps.
              Une version filtrée de C_t.
            </p>
          </div>
        </div>

        {/* Les 4 portes */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {gates.map((gate, i) => (
            <motion.div key={gate.name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              className="rounded-lg p-3"
              style={{
                background: 'var(--bg-base)',
                border: `1px solid ${gate.color}50`,
                minHeight: 200,
              }}>
              <div className="flex items-center gap-2 mb-2">
                <div style={{ color: gate.color }}>{gate.icon}</div>
                <span className="text-xs font-bold uppercase tracking-wide" style={{ color: gate.color }}>
                  {gate.name}
                </span>
              </div>

              <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                {gate.role}
              </p>

              <div className="rounded p-1.5 mb-2 text-center"
                style={{ background: 'var(--bg-card)' }}>
                <span style={{ fontSize: '0.7em' }}>
                  <InlineMath math={gate.formula}/>
                </span>
              </div>

              <p className="text-[10px] leading-snug mb-2" style={{ color: 'var(--text-tertiary)' }}>
                {gate.explanation}
              </p>

              <p className="text-[10px] italic" style={{ color: gate.color }}>
                {gate.example}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Équation finale */}
        <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border-default)' }}>
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--accent-purple)' }}>
            🔬 Mise à jour de la cellule LSTM (en 2 étapes) :
          </p>
          <div className="space-y-2">
            <div className="rounded-lg p-2" style={{ background: 'var(--bg-card)' }}>
              <p className="text-[10px] mb-1" style={{ color: GATE_COLORS.cell }}>
                ① Mise à jour mémoire long terme C_t :
              </p>
              <BlockMath math={String.raw`C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t`}/>
            </div>
            <div className="rounded-lg p-2" style={{ background: 'var(--bg-card)' }}>
              <p className="text-[10px] mb-1" style={{ color: GATE_COLORS.hidden }}>
                ② Calcul de la sortie h_t :
              </p>
              <BlockMath math={String.raw`h_t = o_t \odot \tanh(C_t)`}/>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CycleOverview (réutilisé du RNN)
// ═══════════════════════════════════════════════════════════════
function CycleOverview() {
  const steps = [
    { num: 1, title: 'Forward',  icon: '➡️', desc: 'Calculer ŷ',           formula: String.raw`\hat{y} = f(x ; W)`,                  color: 'var(--accent-blue)', explanation: 'Les données traversent les portes LSTM jusqu\'à la prédiction ŷ.' },
    { num: 2, title: 'Loss',     icon: '🎯', desc: 'Mesurer l\'erreur',    formula: String.raw`\mathcal{L} = (y - \hat{y})^2`,        color: 'var(--accent-orange)', explanation: 'On compare ŷ avec la vraie valeur y. Plus l\'écart est grand, plus la loss est grande.' },
    { num: 3, title: 'Backward', icon: '⬅️', desc: 'Calculer gradients',   formula: String.raw`\nabla W = \frac{\partial \mathcal{L}}{\partial W}`, color: '#f06292', explanation: 'On remonte à travers les 4 portes pour calculer comment ajuster chaque poids.' },
    { num: 4, title: 'Update',   icon: '⚙️', desc: 'Mettre à jour',        formula: String.raw`W \leftarrow W - \eta \cdot \nabla W`, color: 'var(--accent-purple)', explanation: 'Tous les poids des 4 portes sont déplacés dans la bonne direction.' },
    { num: 5, title: 'Répéter',  icon: '🔁', desc: 'Sur tous les batchs',  formula: String.raw`\text{tous batchs, toutes époques}`,  color: 'var(--accent-green)', explanation: 'Le cycle se répète des milliers de fois.' },
  ];

  return (
    <div className="rounded-xl p-4 border-2"
      style={{ background: 'linear-gradient(135deg, var(--bg-card), var(--bg-base))', borderColor: 'var(--accent-blue)' }}>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-bold flex items-center gap-2" style={{ color: 'var(--accent-blue)' }}>
          <Sparkles size={14}/> Cycle d'apprentissage LSTM
        </h4>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
        {steps.map((step, i) => (
          <motion.div key={step.num}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15 }}
            className="rounded-lg p-3 relative"
            style={{ background: 'var(--bg-base)', border: `1px solid ${step.color}50`, minHeight: 150 }}>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold"
                style={{ background: `${step.color}20`, color: step.color, border: `1px solid ${step.color}` }}>
                {step.num}
              </div>
              <span className="text-base">{step.icon}</span>
              <span className="text-xs font-bold uppercase tracking-wide" style={{ color: step.color }}>
                {step.title}
              </span>
            </div>
            <p className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>{step.desc}</p>
            <div className="rounded p-1.5 mb-2 text-center" style={{ background: 'var(--bg-card)', minHeight: 38 }}>
              <span style={{ fontSize: '0.8em' }}><InlineMath math={step.formula}/></span>
            </div>
            <p className="text-[10px] leading-snug" style={{ color: 'var(--text-tertiary)' }}>{step.explanation}</p>
            {i < steps.length - 1 && (
              <div className="hidden md:block absolute top-1/2 -right-2 -translate-y-1/2 z-10">
                <ChevronRight size={16} style={{ color: step.color }}/>
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sidebar (similaire au RNN)
// ═══════════════════════════════════════════════════════════════
function DemoSidebar({ config, setConfig, onRun, loading, onSwitchToRNN }) {
  const setLayers = (n) => {
    let newLayers = [...config.layers];
    while (newLayers.length < n) newLayers.push(4);
    newLayers = newLayers.slice(0, n);
    setConfig({ ...config, layers: newLayers });
  };

  const updateLayer = (idx, value) => {
    const v = parseInt(value) || 4;
    const newLayers = [...config.layers];
    newLayers[idx] = Math.max(2, Math.min(16, v));
    setConfig({ ...config, layers: newLayers });
  };

  return (
    <div className="rounded-xl border p-4 space-y-4"
      style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
      <div className="flex items-center gap-2 pb-2 border-b" style={{ borderColor:'var(--border-default)' }}>
        <Settings size={14} style={{ color:'var(--accent-purple)' }} />
        <p className="text-xs font-semibold uppercase tracking-widest" style={{ color:'var(--accent-purple)' }}>
          Configuration LSTM
        </p>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          🧠 ARCHITECTURE
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          <button onClick={onSwitchToRNN}
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-tertiary)' }}>
            <div>RNN</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>Simple</div>
          </button>
          <button
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:'var(--tint-purple-bg)', borderColor:'var(--accent-purple)', color:'var(--accent-purple)' }}>
            <div>LSTM ✓</div>
            <div className="text-[9px]" style={{ color:'var(--accent-purple)' }}>Actif</div>
          </button>
          <button disabled
            className="px-2 py-2 rounded-lg text-xs font-bold border opacity-50"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-muted)' }}>
            <div className="flex items-center justify-center gap-1"><Lock size={9}/> GRU</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>Bientôt</div>
          </button>
          <button disabled
            className="px-2 py-2 rounded-lg text-xs font-bold border opacity-50"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-muted)' }}>
            <div className="flex items-center justify-center gap-1"><Lock size={9}/> Transformer</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>Bientôt</div>
          </button>
        </div>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          NOMBRE DE COUCHES
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {[1, 2, 3].map(n => (
            <button key={n} onClick={() => setLayers(n)}
              className="px-2 py-2 rounded-lg text-sm font-bold border"
              style={{
                background: config.layers.length === n ? '#1a3a5c' : 'var(--bg-elevated)',
                borderColor: config.layers.length === n ? LAYER_COLORS[n - 1] : 'var(--border-default)',
                color: config.layers.length === n ? LAYER_COLORS[n - 1] : 'var(--text-tertiary)',
              }}>{n}</button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          CELLULES / COUCHE
        </label>
        <div className="space-y-1.5">
          {config.layers.map((u, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs w-12 font-bold" style={{ color: LAYER_COLORS[i] }}>L{i + 1}</span>
              <input type="number" value={u} min={2} max={16} step={1}
                onChange={e => updateLayer(i, e.target.value)}
                className="flex-1 px-2 py-1 rounded border text-xs font-mono"
                style={{ background:'var(--bg-elevated)', borderColor: LAYER_COLORS[i] + '60', color: LAYER_COLORS[i] }}/>
            </div>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>BATCH</label>
        <div className="grid grid-cols-4 gap-1">
          {[2, 3, 4, 6].map(n => (
            <button key={n} onClick={() => setConfig({ ...config, batch_size: n })}
              className="px-1 py-1 rounded text-xs font-mono border"
              style={{
                background: config.batch_size === n ? '#1a3a5c' : 'var(--bg-elevated)',
                borderColor: config.batch_size === n ? 'var(--accent-blue)' : 'var(--border-default)',
                color: config.batch_size === n ? 'var(--accent-blue)' : 'var(--text-tertiary)',
              }}>{n}</button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>SÉQUENCE</label>
        <div className="grid grid-cols-4 gap-1">
          {[2, 3, 4, 5].map(n => (
            <button key={n} onClick={() => setConfig({ ...config, seq_length: n })}
              className="px-1 py-1 rounded text-xs font-mono border"
              style={{
                background: config.seq_length === n ? 'var(--tint-success-bg)' : 'var(--bg-elevated)',
                borderColor: config.seq_length === n ? 'var(--accent-green)' : 'var(--border-default)',
                color: config.seq_length === n ? 'var(--accent-green)' : 'var(--text-tertiary)',
              }}>{n}h</button>
          ))}
        </div>
      </div>

      <button onClick={onRun} disabled={loading}
        className="w-full px-3 py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2"
        style={{
          background: loading ? 'var(--bg-card)' : 'linear-gradient(135deg,var(--tint-purple-bg),#4a1a4a)',
          border: `1px solid ${loading ? 'var(--border-default)' : 'var(--accent-purple)'}`,
          color: loading ? 'var(--text-muted)' : 'var(--accent-purple)',
        }}>
        {loading ? '⚙️ Calcul...' : <><Zap size={14}/> Lancer la démo LSTM</>}
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// NarrationPanel pour LSTM
// ═══════════════════════════════════════════════════════════════
function NarrationPanel({ data, scene, sceneIndex }) {
  if (!data || !scene) return null;

  return (
    <div className="rounded-xl border" style={{ background:'var(--bg-base)', borderColor:'#ce93d840' }}>
      <div className="px-4 py-2.5 border-b flex items-center justify-between"
        style={{ background:'var(--tint-purple-bg)', borderColor:'#ce93d840' }}>
        <div className="flex items-center gap-2">
          <BookOpen size={14} style={{ color:'var(--accent-purple)' }} />
          <span className="text-sm font-semibold" style={{ color:'var(--accent-purple)' }}>
            📝 Narration LSTM
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color:'var(--text-muted)' }}>
          Scène {sceneIndex + 1}
        </span>
      </div>

      <div className="p-5 space-y-4 min-h-[240px]">
        <AnimatePresence mode="wait">
          <motion.div key={scene.id}
            initial={{ opacity:0, y:10 }}
            animate={{ opacity:1, y:0 }}
            exit={{ opacity:0, y:-10 }}
            transition={{ duration:0.4 }}
          >
            <div className="mb-3">
              <p className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: scene.color || 'var(--accent-purple)' }}>
                {scene.section}
              </p>
              <h3 className="text-base font-bold mt-1" style={{ color:'var(--text-primary)' }}>
                {scene.title}
              </h3>
            </div>

            <p className="text-sm leading-relaxed mb-3" style={{ color:'var(--text-secondary)' }}
              dangerouslySetInnerHTML={{ __html: scene.text }}/>

            {scene.showCycleOverview && <div className="my-4"><CycleOverview/></div>}
            {scene.showAnatomy        && <div className="my-4"><LSTMAnatomy/></div>}

            {scene.formula && (
              <div className="rounded-lg p-3 my-3" style={{ background:'var(--bg-card)' }}>
                <BlockMath math={scene.formula}/>
              </div>
            )}

            {scene.matrices && (
              <div className="flex flex-wrap gap-4 mt-3">
                {scene.matrices.map((m, i) => (
                  <Matrix key={i}
                    data={m.data} title={m.title} subtitle={m.subtitle}
                    rowLabels={m.rowLabels} colLabels={m.colLabels}
                    color={m.color} decimals={m.decimals || 3}
                    small={m.small} highlight={m.highlight}/>
                ))}
              </div>
            )}

            {scene.note && (
              <p className="text-xs italic mt-3 px-3 py-2 rounded" style={{
                background:'var(--bg-card)', color:'var(--text-tertiary)',
                borderLeft:`2px solid ${scene.color || 'var(--accent-purple)'}`
              }}>
                💡 {scene.note}
              </p>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Construction du scénario LSTM
// ═══════════════════════════════════════════════════════════════
function buildLSTMScenario(data) {
  if (!data) return [];
  const { config, weights, forward_steps, prediction, loss_info, backward,
          y_true_norm, y_true_hours, feat_names } = data;
  const { layers, batch_size, seq_length, n_features, n_layers } = config;

  const scenes = [];
  const seqLabels = Array.from({length:batch_size}, (_,i)=>`Séq ${String.fromCharCode(65+i)}`);

  // Intro
  scenes.push({
    id: 'intro',
    section: 'Introduction',
    title: '🧬 Bienvenue dans la démonstration LSTM',
    color: 'var(--accent-purple)',
    text: `Le <b>LSTM</b> (Long Short-Term Memory) est un type de RNN amélioré conçu pour
           résoudre le problème du <b>gradient qui disparaît</b>. Il introduit une
           <b>mémoire long terme (cell state C)</b> et <b>4 portes</b> qui contrôlent
           le flux d'informations.<br/><br/>
           Architecture : <b>${n_layers} couche${n_layers>1?'s':''}</b>,
           batch de <b>${batch_size} séquences</b> sur <b>${seq_length} heures</b>.`,
    note: `Inventé en 1997 par Hochreiter & Schmidhuber, le LSTM est devenu le RNN de référence pour les longues séquences.`,
  });

  // Cycle d'apprentissage
  scenes.push({
    id: 'cycle',
    section: 'Plan',
    title: '🗺️ Cycle d\'apprentissage complet — les 5 étapes',
    color: 'var(--accent-blue)',
    text: `Comme tout réseau de neurones, le LSTM suit le même cycle d'apprentissage.
           La différence est <b>ce qui se passe à l'intérieur</b> du forward.`,
    showCycleOverview: true,
    note: `Ce cycle se répète à chaque batch d'entraînement.`,
  });

  // ANATOMIE LSTM (section spéciale)
  scenes.push({
    id: 'anatomy',
    section: 'Anatomie',
    title: '🧬 Anatomie d\'une cellule LSTM',
    color: 'var(--accent-purple)',
    text: `Avant de plonger dans les calculs, comprenons <b>la structure interne</b>
           d'une cellule LSTM. Contrairement au RNN simple qui n'a qu'un seul état caché h_t,
           le LSTM en a <b>deux : C_t (long terme) et h_t (court terme)</b>, contrôlés par <b>4 portes</b>.`,
    showAnatomy: true,
    note: `La clé du LSTM : la cell state C_t agit comme une "autoroute" où l'information peut circuler intacte sur de longues distances.`,
  });

  // Pour chaque couche, chaque temps : 6 scènes (forget, input, candidate, cell, output, hidden)
  for (let L = 0; L < n_layers; L++) {
    const layerColor = LAYER_COLORS[L];
    const layerW = weights.layers[L];
    const units = layerW.units;
    const neuronLabels = Array.from({length:units}, (_,i)=>`c${i+1}`);
    const inputLabels  = L === 0
      ? feat_names.slice(0, layerW.input_dim)
      : Array.from({length:layerW.input_dim}, (_,i)=>`h${i+1}⁽${L}⁾`);

    // Présentation de la couche LSTM
    scenes.push({
      id: `layer-${L}-intro`,
      section: `Couche LSTM ${L+1}`,
      title: `📦 Couche LSTM ${L+1} avec ses ${units} cellules`,
      color: layerColor,
      text: `Cette couche LSTM a <b>${units} cellules</b>. Chaque cellule contient
             les <b>4 portes</b> et a son propre cell state C et hidden state h.<br/>
             Au total cette couche a <b>${4*(layerW.input_dim*units + units*units + units)}</b> paramètres
             (4 portes × (W_x + W_h + b) par porte).`,
      note: `Les 4 portes sont initialisées avec différentes matrices W_xf, W_hf, W_xi, W_hi, W_xC, W_hC, W_xo, W_ho.`,
    });

    for (let t = 0; t < seq_length; t++) {
      const step = forward_steps[L][t];
      const tt = t + 1;
      const tprev = t;

      // FORGET GATE
      scenes.push({
        id: `layer-${L}-t${t}-forget`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'forget',
        title: `🔴 Forget Gate à t=${tt} : que faut-il OUBLIER ?`,
        color: GATE_COLORS.forget,
        text: `La <b>forget gate</b> regarde l'entrée X_${tt} et le hidden précédent h_${tprev}
               pour décider ce qu'il faut <b>retirer de la mémoire C_${tprev}</b>.<br/>
               Sortie entre 0 (oublier) et 1 (garder).`,
        formula: String.raw`f_${tt} = \sigma(W_f \cdot [h_${tprev}, x_${tt}] + b_f)`,
        matrices: [{
          title: `f_${tt}`,
          subtitle: 'forget gate',
          data: step.f_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.forget,
          highlight: true,
        }],
        note: `Toutes les valeurs sont entre 0 et 1 grâce à la sigmoïde σ.`,
      });

      // INPUT GATE
      scenes.push({
        id: `layer-${L}-t${t}-input`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'input',
        title: `🔵 Input Gate à t=${tt} : que faut-il AJOUTER ?`,
        color: GATE_COLORS.input,
        text: `La <b>input gate</b> décide combien des nouvelles informations
               (calculées par la candidate C̃) on va ajouter à la mémoire.`,
        formula: String.raw`i_${tt} = \sigma(W_i \cdot [h_${tprev}, x_${tt}] + b_i)`,
        matrices: [{
          title: `i_${tt}`,
          subtitle: 'input gate',
          data: step.i_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.input,
          highlight: true,
        }],
      });

      // CANDIDATE
      scenes.push({
        id: `layer-${L}-t${t}-candidate`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'candidate',
        title: `🟢 Candidate à t=${tt} : nouvelles infos potentielles`,
        color: GATE_COLORS.candidate,
        text: `On calcule les <b>nouvelles informations</b> qui POURRAIENT être ajoutées
               à la mémoire. tanh produit des valeurs entre -1 et +1.`,
        formula: String.raw`\tilde{C}_${tt} = \tanh(W_C \cdot [h_${tprev}, x_${tt}] + b_C)`,
        matrices: [{
          title: `C̃_${tt}`,
          subtitle: 'candidate',
          data: step.C_tilde,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.candidate,
          highlight: true,
        }],
      });

      // CELL STATE UPDATE
      scenes.push({
        id: `layer-${L}-t${t}-cell`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'cell',
        title: `🟣 Mise à jour du Cell State C_${tt} (mémoire long terme)`,
        color: GATE_COLORS.cell,
        text: `On combine : <b>oubli sélectif</b> de l'ancienne mémoire
               + <b>ajout sélectif</b> de la nouvelle info.<br/>
               <b>⊙</b> = produit élément par élément (Hadamard).`,
        formula: String.raw`C_${tt} = f_${tt} \odot C_${tprev} + i_${tt} \odot \tilde{C}_${tt}`,
        matrices: [{
          title: `C_${tt}`,
          subtitle: 'cell state mis à jour',
          data: step.C_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.cell,
          highlight: true,
        }],
        note: `C_${tt} peut prendre n'importe quelle valeur (pas limité à [-1,1] grâce à l'addition).`,
      });

      // OUTPUT GATE
      scenes.push({
        id: `layer-${L}-t${t}-output`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'output',
        title: `🟠 Output Gate à t=${tt} : que va-t-on SORTIR ?`,
        color: GATE_COLORS.output,
        text: `La <b>output gate</b> filtre la mémoire C_${tt} pour décider
               quelle partie devient le hidden state visible h_${tt}.`,
        formula: String.raw`o_${tt} = \sigma(W_o \cdot [h_${tprev}, x_${tt}] + b_o)`,
        matrices: [{
          title: `o_${tt}`,
          subtitle: 'output gate',
          data: step.o_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.output,
          highlight: true,
        }],
      });

      // HIDDEN STATE FINAL
      scenes.push({
        id: `layer-${L}-t${t}-hidden`,
        section: `Couche ${L+1} · t=${tt}`,
        activeGate: 'hidden',
        title: `✨ Hidden State h_${tt} : la sortie de la cellule`,
        color: GATE_COLORS.hidden,
        text: `Le hidden state est la <b>sortie visible</b> de la cellule LSTM à ce pas.
               C'est lui qui sera passé à la couche suivante (et à t+1 pour cette couche).`,
        formula: String.raw`h_${tt} = o_${tt} \odot \tanh(C_${tt})`,
        matrices: [{
          title: `h_${tt}`,
          subtitle: 'hidden state final',
          data: step.H_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: GATE_COLORS.hidden,
          highlight: true,
        }],
        note: t === seq_length - 1 && L === n_layers - 1
          ? `🎯 Dernier hidden state — il servira à calculer la prédiction.`
          : `Ce hidden state continue son chemin (récurrence ou couche suivante).`,
      });
    }
  }

  // Prédiction
  scenes.push({
    id: 'predict',
    section: 'Prédiction',
    title: '🎯 Calcul de la prédiction Ŷ',
    color: '#f06292',
    text: `Le dernier hidden state <b>h_${seq_length}⁽${n_layers}⁾</b> est multiplié par W_out
           pour obtenir la prédiction de RUL.`,
    formula: String.raw`\hat{Y} = h_${seq_length}^{(${n_layers})} \cdot W_{out} + b_{out}`,
    matrices: [{
      title: 'Ŷ (norm)',
      data: [prediction.Y_pred_norm],
      rowLabels: ['ŷ'],
      colLabels: seqLabels,
      color: '#f06292',
      highlight: true,
    }],
    note: `En heures : ${prediction.Y_pred_hours.map(v => v.toFixed(1)+'h').join(', ')}`,
  });

  // Loss
  scenes.push({
    id: 'loss',
    section: 'Loss',
    title: '📉 Calcul de la fonction de coût',
    color: 'var(--accent-orange)',
    text: `On compare nos prédictions avec les vraies valeurs.`,
    formula: String.raw`\mathcal{L} = \frac{1}{B} \sum (\hat{y}_i - y_i)^2 = ${loss_info.loss.toFixed(4)}`,
    matrices: [
      { title: 'Y vrai', data: [y_true_norm], rowLabels: ['y'], colLabels: seqLabels, color: 'var(--success)', small: true },
      { title: 'Ŷ prédit', data: [prediction.Y_pred_norm], rowLabels: ['ŷ'], colLabels: seqLabels, color: '#f06292', small: true },
      { title: 'Erreurs', data: [loss_info.per_seq_error_norm], rowLabels: ['err'], colLabels: seqLabels, color: 'var(--accent-orange)', small: true, highlight: true },
    ],
  });

  // Backprop
  for (let L = n_layers - 1; L >= 0; L--) {
    const g = backward.layer_grads[L];
    scenes.push({
      id: `backprop-layer-${L}`,
      section: 'Backprop',
      title: `🔄 Gradients pour la couche LSTM ${L+1} (12 matrices)`,
      color: '#f06292',
      text: `Le gradient remonte à travers les <b>4 portes</b>. On obtient des gradients
             pour les 12 matrices de poids de cette couche.<br/>
             Norme totale : <b>${g.norm_total.toFixed(3)}</b>`,
      formula: String.raw`\nabla W^{(${L+1})} = \{\nabla W_f, \nabla W_i, \nabla W_C, \nabla W_o\}`,
      note: `Chaque porte a 3 gradients : dW_x (entrée), dW_h (récurrence), db (biais).
             4 portes × 3 = 12 gradients par couche.`,
    });
  }

  // Fin
  scenes.push({
    id: 'end',
    section: 'Conclusion',
    title: '🎉 Fin de la démonstration LSTM',
    color: 'var(--success)',
    text: `Tu as vu une itération complète d'un LSTM avec ses <b>4 portes</b> et son
           <b>cell state C_t</b> qui permet la mémoire long terme.<br/><br/>
           <b>Différences principales avec un RNN simple :</b><br/>
           • RNN : 1 état (h) · 1 matrice par couche · gradient peut disparaître<br/>
           • LSTM : 2 états (h + C) · 4 portes · résiste au vanishing gradient`,
    note: `Le LSTM est plus lent à entraîner mais bien plus efficace pour les longues séquences.`,
  });

  return scenes;
}

// ═══════════════════════════════════════════════════════════════
// COMPOSANT PRINCIPAL
// ═══════════════════════════════════════════════════════════════
export default function LSTMDemoPanel({ onSwitchToRNN }) {
  const [config, setConfig] = useState({
    layers:        [4],
    batch_size:    4,
    seq_length:    3,
    learning_rate: 0.1,
    seed:          7,
  });
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const [sceneIndex, setSceneIndex] = useState(0);
  const [autoPlay, setAutoPlay]     = useState(false);
  const [speed, setSpeed]           = useState(3500);

  const scenes = useMemo(() => buildLSTMScenario(data), [data]);
  const currentScene = scenes[sceneIndex];

  const { activeLayer, activeT, activeGate, mode } = useMemo(() => {
    if (!currentScene) return { activeLayer: -1, activeT: 0, activeGate: null, mode: 'idle' };
    const id = currentScene.id;
    const m = id.match(/layer-(\d+)-t(\d+)/);
    if (m) return {
      activeLayer: parseInt(m[1]),
      activeT: parseInt(m[2]),
      activeGate: currentScene.activeGate || null,
      mode: 'forward'
    };
    if (id === 'predict') return { activeLayer: -1, activeT: 0, activeGate: null, mode: 'predict' };
    if (id.startsWith('backprop')) return { activeLayer: -1, activeT: 0, activeGate: null, mode: 'backprop' };
    return { activeLayer: -1, activeT: 0, activeGate: null, mode: 'idle' };
  }, [currentScene]);

  const runDemo = async () => {
    setLoading(true); setError(null); setSceneIndex(0); setAutoPlay(false);
    try {
      const res = await fetch(`${API}/api/lstm_demo/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Erreur serveur');
      setData(await res.json());
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  useEffect(() => {
    if (!autoPlay) return;
    const id = setInterval(() => {
      setSceneIndex(s => {
        if (s >= scenes.length - 1) { setAutoPlay(false); return s; }
        return s + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [autoPlay, scenes.length, speed]);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background:'linear-gradient(135deg,var(--tint-purple-bg),#2a1a4a)', border:'1px solid #ce93d840' }}>
            <Brain size={18} style={{ color:'var(--accent-purple)' }} />
          </div>
          <div>
            <h2 className="text-lg font-semibold" style={{ color:'var(--text-primary)' }}>
              Démo LSTM — Mini-dataset synthétique
            </h2>
            <p className="text-xs" style={{ color:'var(--text-tertiary)' }}>
              4 portes (forget, input, candidate, output) · Cell state C_t · Mémoire long terme
            </p>
          </div>
        </div>
        {data && (
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2 py-1 rounded flex items-center gap-1"
              style={{ background:'var(--bg-card)', color:'var(--accent-orange)' }}>
              <Factory size={11}/> Démo · synthétique
            </span>
            <span className="px-2 py-1 rounded uppercase font-bold"
              style={{ background:'var(--tint-purple-bg)', color:'var(--accent-purple)' }}>
              LSTM
            </span>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border p-3 flex items-start gap-2"
          style={{ background:'var(--tint-error-bg)', borderColor:'#f06292' }}>
          <AlertTriangle size={14} style={{ color:'#f06292', flexShrink:0, marginTop:1 }}/>
          <div>
            <span className="text-xs font-semibold" style={{ color:'#f06292' }}>Erreur : {error}</span>
            {error.includes('ingestion') && (
              <p className="text-xs mt-1" style={{ color:'var(--text-tertiary)' }}>
                Va dans l'onglet <b>Ingestion</b> et clique "Lancer" d'abord.
              </p>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-3">
          <DemoSidebar config={config} setConfig={setConfig}
            onRun={runDemo} loading={loading} onSwitchToRNN={onSwitchToRNN}/>
        </div>

        <div className="col-span-9 space-y-4">
          {!data && !loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <Brain size={32} style={{ color:'var(--text-muted)' }} className="mx-auto mb-3"/>
              <p className="text-sm" style={{ color:'var(--text-tertiary)' }}>
                Configure à gauche puis clique <b>Lancer la démo LSTM</b>
              </p>
            </div>
          )}

          {loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <p className="text-sm animate-pulse" style={{ color:'var(--accent-purple)' }}>
                ⚙️ Calcul LSTM en cours (4 portes)...
              </p>
            </div>
          )}

          {data && (
            <>
              <LSTMCellBox config={data.config} weights={data.weights}
                slices={data.slices_2d}
                forward_steps={data.forward_steps}
                prediction={data.prediction}
                backward={data.backward}
                activeLayer={activeLayer} activeT={activeT}
                activeGate={activeGate} mode={mode}
                sceneId={currentScene?.id || ''}/>

              <div className="rounded-xl border p-3" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <button onClick={() => setAutoPlay(!autoPlay)}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5"
                    style={{ background:'var(--bg-elevated)', border:'1px solid #ce93d840', color:'var(--accent-purple)' }}>
                    {autoPlay ? <><Pause size={12}/> Pause</> : <><Play size={12}/> Animer</>}
                  </button>
                  <button onClick={() => setSceneIndex(s => Math.max(0, s - 1))}
                    disabled={sceneIndex === 0}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold"
                    style={{ background:'var(--bg-elevated)', border:'1px solid var(--border-default)', color:'var(--text-tertiary)' }}>
                    ← Précédent
                  </button>
                  <button onClick={() => setSceneIndex(s => Math.min(scenes.length - 1, s + 1))}
                    disabled={sceneIndex >= scenes.length - 1}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold"
                    style={{ background:'var(--bg-elevated)', border:'1px solid #ce93d840', color:'var(--accent-purple)' }}>
                    Suivant →
                  </button>
                  <button onClick={() => { setSceneIndex(0); setAutoPlay(false); }}
                    className="px-3 py-1.5 rounded-lg text-xs flex items-center gap-1"
                    style={{ background:'var(--bg-elevated)', border:'1px solid var(--border-default)', color:'var(--text-tertiary)' }}>
                    <RotateCcw size={11}/> Début
                  </button>
                  <div className="flex items-center gap-1 ml-2">
                    <span className="text-xs" style={{ color:'var(--text-muted)' }}>Vitesse :</span>
                    {[5000, 3500, 2000, 1000].map((s, i) => (
                      <button key={s} onClick={() => setSpeed(s)}
                        className="px-2 py-0.5 rounded text-xs font-mono border"
                        style={{
                          background: speed === s ? 'var(--tint-purple-bg)' : 'var(--bg-elevated)',
                          borderColor: speed === s ? 'var(--accent-purple)' : 'var(--border-default)',
                          color: speed === s ? 'var(--accent-purple)' : 'var(--text-muted)',
                        }}>
                        {['🐢','🚶','🏃','⚡'][i]}
                      </button>
                    ))}
                  </div>
                  <div className="flex-1"/>
                  <span className="text-xs font-mono px-2 py-1 rounded"
                    style={{ background:'var(--bg-elevated)', color:'var(--accent-purple)' }}>
                    {sceneIndex + 1} / {scenes.length}
                  </span>
                </div>
                <div className="w-full h-1 rounded overflow-hidden" style={{ background:'var(--bg-deep)' }}>
                  <div className="h-full transition-all duration-300"
                    style={{
                      width: `${((sceneIndex + 1) / scenes.length) * 100}%`,
                      background: 'linear-gradient(90deg,#f06292,var(--accent-blue),var(--accent-green),var(--accent-orange),var(--accent-purple))',
                    }}/>
                </div>
              </div>

              <NarrationPanel data={data} scene={currentScene} sceneIndex={sceneIndex}/>
            </>
          )}
        </div>
      </div>
    </div>
  );
}