/**
 * RNNDemoPanel.jsx — Démo RNN Multi-Couches avec animation + narration + Loss + Backprop
 * Architecture sélectionnable : RNN (actif) / LSTM / GRU / Transformer (à venir)
 * Mini-dataset synthétique pédagogique (généré côté serveur — voir backend/demos/synthetic_data.py)
 */
import { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import {
  Box, Play, Pause, RotateCcw, Zap, AlertTriangle, Cpu,
  BookOpen, Factory, Settings, Sparkles, ArrowRight,
  Target, Calculator, Sigma, ArrowLeft, FastForward,
  ChevronRight, Brain, Lock
} from 'lucide-react';

const API = 'http://localhost:8000';

// ─── Couleurs par couche ──────────────────────────────────────
const LAYER_COLORS = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-orange)'];

// ─── Helpers couleur ──────────────────────────────────────────
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
// Matrice 2D avec crochets mathématiques
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
// Boîte Modèle CINÉMATIQUE — explication pédagogique complète
// ═══════════════════════════════════════════════════════════════
function ModelBox({ config, weights, slices, forward_steps, prediction, backward,
                    activeLayer = -1, activeT = 0, mode = 'idle', sceneId = '' }) {
  if (!config || !weights) return null;
  const { layers, n_features, batch_size, seq_length } = config;
  const n_layers = layers.length;
  const W = 1180, H = 620;

  // Coordonnées clés
  const queueX  = 30;
  const inputX  = 220;
  const sumX    = 380;       // point de sommation (Z)
  const tanhX   = 480;       // étape tanh
  const layerXs = layers.map((_, L) => 600 + L * 180);
  const lastLayerX = layerXs[n_layers - 1];
  const outputX = lastLayerX + 220;

  const zoomY  = H - 110;    // zone "zoom sur un neurone"

  const neuronYs = (units, cy, height) => {
    const step = Math.min(28, (height - 60) / Math.max(1, units - 1));
    return Array.from({ length: units }, (_, i) =>
      cy - ((units - 1) / 2) * step + i * step
    );
  };

  const cy        = 240;     // centre vertical de la zone modèle
  const inputYs   = neuronYs(n_features, cy, 200);
  const layerYs   = layers.map(units => neuronYs(units, cy, 200));
  const featNames = ['volt', 'rotate', 'pres', 'vibr'];
  const seqLabels = Array.from({ length: batch_size }, (_, i) => String.fromCharCode(65 + i));

  // Détails du neurone "zoomé"
  const zoomNeuronIdx = 0;  // on zoome sur h1
  const currentZ = forward_steps?.[activeLayer]?.[activeT]?.Z_t?.[0]?.[zoomNeuronIdx] ?? 0;
  const currentH = forward_steps?.[activeLayer]?.[activeT]?.H_t?.[0]?.[zoomNeuronIdx] ?? 0;

  // Narration en bas
  const narrationText = (() => {
    if (mode === 'idle')   return '⏸ Cliquez "Suivant" ou lancez l\'animation pour démarrer';
    if (mode === 'predict') return `🎯 Le dernier état caché H_${seq_length} est multiplié par W_out → on obtient la prédiction ŷ`;
    if (mode === 'backprop') return `🔄 L'erreur remonte depuis ŷ et calcule les gradients pour ajuster les poids W`;
    if (mode === 'forward') {
      if (sceneId.includes('matmul-xh')) return `🟦 Multiplication X_${activeT+1} · W_xh : chaque ligne de X est combinée avec chaque colonne de W`;
      if (sceneId.includes('compute'))    return `➕ On somme les 3 contributions : X·W_xh + H_${activeT}·W_hh + biais = Z_${activeT+1}`;
      if (sceneId.includes('-Z'))         return `📊 Z_${activeT+1} est calculé — il faut maintenant le passer dans tanh`;
      if (sceneId.includes('-H'))         return `✨ tanh écrase Z entre -1 et 1 → on obtient H_${activeT+1}, le nouvel état caché`;
      if (sceneId.includes('-prep'))      return `⏰ La tranche X_${activeT+1} entre dans le réseau au temps t=${activeT+1}`;
      return `🔵 Couche ${activeLayer + 1} · temps t=${activeT + 1} en cours...`;
    }
    return '';
  })();

  // Grandeurs visuelles
  const flowSpeed = 1.2;

  // Zone forward visible ?
  const showForwardFlow = mode === 'forward' && activeLayer >= 0;
  const showBackpropFlow = mode === 'backprop';

  return (
    <div className="rounded-xl border-2 overflow-hidden"
      style={{ borderColor: 'var(--accent-orange)', background: 'linear-gradient(135deg, var(--bg-base), #1a1508)' }}>
      <div className="flex items-center justify-between px-4 py-2.5"
        style={{ background:'#2a1a0a', borderBottom:'1px solid #ffb74d40' }}>
        <div className="flex items-center gap-2">
          <Cpu size={15} style={{ color:'var(--accent-orange)' }} />
          <span className="text-sm font-bold" style={{ color:'var(--accent-orange)' }}>
            🏭 BOÎTE MODÈLE — RNN · {n_layers} couche{n_layers>1?'s':''}
            {mode === 'forward' && ` · t=${activeT + 1}/${seq_length}`}
            {mode === 'predict' && ` · 🎯 PRÉDICTION`}
            {mode === 'backprop' && ` · ← BACKPROP`}
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color:'var(--text-muted)' }}>
          {weights.layers.reduce((s, L) =>
            s + L.W_xh.length * L.W_xh[0].length + L.W_hh.length * L.W_hh[0].length + L.b.length, 0
          ) + weights.W_out.length} paramètres
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 700 }}>
        <defs>
          <pattern id="modelgrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--bg-card)" strokeWidth="0.4"/>
          </pattern>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3"
            orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L6,3 z" fill="var(--text-tertiary)"/>
          </marker>
          <marker id="arrowRed" markerWidth="6" markerHeight="6" refX="5" refY="3"
            orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L6,3 z" fill="#f06292"/>
          </marker>
          <radialGradient id="backpropWave">
            <stop offset="0%" stopColor="#f06292" stopOpacity="0.6"/>
            <stop offset="100%" stopColor="#f06292" stopOpacity="0"/>
          </radialGradient>
        </defs>
        <rect width={W} height={H} fill="url(#modelgrid)" />

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 1 — Tranches 2D empilées (gauche)              */}
        {/* ═══════════════════════════════════════════════════════ */}
        <g>
          <text x={queueX + 60} y={20} textAnchor="middle" fontSize={11}
            fill="var(--accent-purple)" fontFamily="monospace" fontWeight="bold">
            📦 ENTRÉE
          </text>
          <text x={queueX + 60} y={34} textAnchor="middle" fontSize={9}
            fill="var(--text-muted)" fontFamily="monospace">
            {seq_length} tranches 2D
          </text>

          {/* Flèche d'ordre temporel verticale */}
          <line x1={queueX + 130} y1={60} x2={queueX + 130} y2={50 + seq_length * 75}
            stroke="var(--accent-purple)" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#arrow)" opacity={0.5}/>
          <text x={queueX + 138} y={50 + seq_length * 38} fontSize={9}
            fill="var(--accent-purple)" fontFamily="monospace">temps</text>

          {Array.from({ length: seq_length }).map((_, t) => {
            const isActive = activeT === t && mode === 'forward';
            const isPast   = activeT > t && mode === 'forward';
            const baseY    = 50 + t * 75;
            const sliceData = slices?.[t];

            return (
              <motion.g key={t}
                animate={{
                  scale: isActive ? 1.05 : 1,
                  opacity: isPast ? 0.3 : 1,
                }}
                transition={{ duration: 0.5 }}
              >
                {/* Boîte tranche */}
                <rect x={queueX} y={baseY} width={120} height={56} rx={6}
                  fill={isActive ? '#1a3a5c' : isPast ? 'var(--bg-base)' : 'var(--bg-card)'}
                  stroke={isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--border-strong)'}
                  strokeWidth={isActive ? 2 : 1}
                />
                {/* Mini représentation des valeurs : 4 lignes (séquences) × 4 colonnes (features) */}
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
                {/* Label */}
                <text x={queueX + 60} y={baseY + 16} textAnchor="middle"
                  fontSize={11} fontFamily="monospace" fontWeight="bold"
                  fill={isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--text-tertiary)'}>
                  X_{t + 1} {isActive && '⬅'} {isPast && '✓'}
                </text>
              </motion.g>
            );
          })}
        </g>

        {/* Flèche file d'attente → INPUT (active uniquement quand forward L=0) */}
        {showForwardFlow && activeLayer === 0 && (
          <>
            <line x1={queueX + 145} y1={cy}
              x2={inputX - 25} y2={cy}
              stroke="var(--accent-blue)" strokeWidth={2}
              markerEnd="url(#arrow)"/>
            <motion.circle r={5} fill="#fff"
              initial={{ cx: queueX + 145, cy: cy }}
              animate={{ cx: [queueX + 145, inputX - 25] }}
              transition={{ duration: flowSpeed, repeat: Infinity, ease: "linear" }}
            />
          </>
        )}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 2 — INPUT (4 carrés)                            */}
        {/* ═══════════════════════════════════════════════════════ */}
        <text x={inputX} y={20} textAnchor="middle" fontSize={11} fill="var(--accent-blue)"
          fontFamily="monospace" fontWeight="bold">INPUT</text>
        <text x={inputX} y={34} textAnchor="middle" fontSize={9} fill="var(--text-muted)"
          fontFamily="monospace">{n_features} features</text>

        {inputYs.map((y, i) => (
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
        ))}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 3 — Sommation Z = X·W_xh + H·W_hh + b           */}
        {/* ═══════════════════════════════════════════════════════ */}
        <text x={sumX} y={20} textAnchor="middle" fontSize={11} fill="var(--accent-orange)"
          fontFamily="monospace" fontWeight="bold">SOMMATION</text>
        <text x={sumX} y={34} textAnchor="middle" fontSize={9} fill="var(--text-muted)"
          fontFamily="monospace">Z = X·W + H·W + b</text>

        {/* 3 flèches qui convergent vers le cercle Z */}
        {/* 1. depuis INPUT (X·W_xh) */}
        {inputYs.map((y, i) => (
          <line key={`xw-${i}`}
            x1={inputX + 22} y1={y}
            x2={sumX - 18} y2={cy}
            stroke={showForwardFlow && activeLayer === 0 ? 'var(--accent-blue)' : 'var(--border-default)'}
            strokeWidth={showForwardFlow && activeLayer === 0 ? 1 : 0.4}
            opacity={0.6}
          />
        ))}
        {/* Étiquette W_xh */}
        <g transform={`translate(${(inputX + sumX) / 2}, ${cy - 60})`}>
          <rect x={-30} y={-12} width={60} height={20} rx={4}
            fill="var(--bg-base)" stroke="var(--accent-blue)" strokeWidth={1}/>
          <text x={0} y={2} textAnchor="middle" fontSize={11}
            fill="var(--accent-blue)" fontFamily="monospace" fontWeight="bold">W_xh</text>
        </g>

        {/* 2. depuis l'état précédent (H_{t-1}·W_hh) — venant d'en bas */}
        {showForwardFlow && (
          <>
            <path d={`M ${sumX} ${cy + 80} Q ${sumX} ${cy + 30}, ${sumX - 12} ${cy + 8}`}
              fill="none" stroke="var(--accent-purple)" strokeWidth={1.2}
              strokeDasharray="3,2" markerEnd="url(#arrow)" opacity={0.7}/>
            <text x={sumX + 8} y={cy + 50} fontSize={10}
              fill="var(--accent-purple)" fontFamily="monospace">H_{activeT}·W_hh</text>
          </>
        )}

        {/* 3. depuis le biais b — venant d'en haut à gauche */}
        {showForwardFlow && (
          <>
            <line x1={sumX - 60} y1={cy - 80}
              x2={sumX - 14} y2={cy - 14}
              stroke="var(--accent-orange)" strokeWidth={1.2}
              markerEnd="url(#arrow)" opacity={0.7}/>
            <text x={sumX - 65} y={cy - 85} fontSize={10}
              fill="var(--accent-orange)" fontFamily="monospace">+ b</text>
          </>
        )}

        {/* Cercle Z (somme) */}
        <motion.g animate={{ scale: showForwardFlow ? [1, 1.2, 1] : 1 }}
          transition={{ duration: 0.6 }}>
          <circle cx={sumX} cy={cy} r={22}
            fill={showForwardFlow ? '#3a2a0d' : 'var(--bg-card)'}
            stroke="var(--accent-orange)" strokeWidth={2}/>
          <text x={sumX} y={cy + 5} textAnchor="middle" fontSize={14}
            fill="var(--accent-orange)" fontFamily="monospace" fontWeight="bold">Σ</text>
        </motion.g>
        <text x={sumX} y={cy + 38} textAnchor="middle" fontSize={9}
          fill="var(--accent-orange)" fontFamily="monospace">
          Z = {showForwardFlow ? currentZ.toFixed(2) : '?'}
        </text>

        {/* Flux X·W_xh animé */}
        {showForwardFlow && activeLayer === 0 && (
          <motion.circle r={4} fill="var(--accent-blue)"
            initial={{ cx: inputX + 22, cy: cy }}
            animate={{ cx: [inputX + 22, sumX - 18] }}
            transition={{ duration: flowSpeed, repeat: Infinity, ease: "linear" }}
          />
        )}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 4 — Activation tanh                              */}
        {/* ═══════════════════════════════════════════════════════ */}
        <text x={tanhX} y={20} textAnchor="middle" fontSize={11} fill="var(--success)"
          fontFamily="monospace" fontWeight="bold">ACTIVATION</text>
        <text x={tanhX} y={34} textAnchor="middle" fontSize={9} fill="var(--text-muted)"
          fontFamily="monospace">tanh écrase [-1, 1]</text>

        {/* Flèche Z → tanh */}
        <line x1={sumX + 22} y1={cy} x2={tanhX - 28} y2={cy}
          stroke={showForwardFlow ? 'var(--accent-orange)' : 'var(--border-default)'}
          strokeWidth={showForwardFlow ? 1.5 : 0.5}
          markerEnd="url(#arrow)" opacity={0.8}/>

        {/* Boîte tanh avec mini courbe */}
        <motion.g animate={{ scale: showForwardFlow ? [1, 1.1, 1] : 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}>
          <rect x={tanhX - 28} y={cy - 28} width={56} height={56} rx={6}
            fill={showForwardFlow ? 'var(--tint-success-bg)' : 'var(--bg-card)'}
            stroke="var(--success)" strokeWidth={2}/>
          {/* Mini courbe tanh */}
          <path d={`M ${tanhX - 22} ${cy + 18}
                    Q ${tanhX - 10} ${cy + 18}, ${tanhX} ${cy}
                    Q ${tanhX + 10} ${cy - 18}, ${tanhX + 22} ${cy - 18}`}
            fill="none" stroke="var(--success)" strokeWidth={2} opacity={0.9}/>
          {/* Axes */}
          <line x1={tanhX - 22} y1={cy} x2={tanhX + 22} y2={cy}
            stroke="var(--success)" strokeWidth={0.3} opacity={0.4}/>
          <line x1={tanhX} y1={cy - 22} x2={tanhX} y2={cy + 22}
            stroke="var(--success)" strokeWidth={0.3} opacity={0.4}/>
        </motion.g>
        <text x={tanhX} y={cy + 50} textAnchor="middle" fontSize={9}
          fill="var(--success)" fontFamily="monospace">tanh</text>

        {/* Flux Z → tanh animé */}
        {showForwardFlow && (
          <motion.circle r={4} fill="var(--accent-orange)"
            initial={{ cx: sumX + 22, cy: cy }}
            animate={{ cx: [sumX + 22, tanhX - 28] }}
            transition={{ duration: flowSpeed, repeat: Infinity, ease: "linear", delay: 0.4 }}
          />
        )}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 5 — Neurones cachés (couches)                   */}
        {/* ═══════════════════════════════════════════════════════ */}
        {layerXs.map((x, L) => (
          <g key={`L-title-${L}`}>
            <text x={x} y={20} textAnchor="middle" fontSize={11}
              fill={LAYER_COLORS[L]} fontFamily="monospace" fontWeight="bold">
              COUCHE {L + 1}
            </text>
            <text x={x} y={34} textAnchor="middle" fontSize={9} fill="var(--text-muted)"
              fontFamily="monospace">{layers[L]} neurones</text>
          </g>
        ))}

        {/* Flèche tanh → couche 1 */}
        <line x1={tanhX + 28} y1={cy} x2={layerXs[0] - 18} y2={cy}
          stroke={showForwardFlow ? LAYER_COLORS[0] : 'var(--border-default)'}
          strokeWidth={showForwardFlow ? 1.5 : 0.5}
          markerEnd="url(#arrow)" opacity={0.8}/>
        {showForwardFlow && (
          <motion.circle r={4} fill="var(--success)"
            initial={{ cx: tanhX + 28, cy: cy }}
            animate={{ cx: [tanhX + 28, layerXs[0] - 18] }}
            transition={{ duration: flowSpeed, repeat: Infinity, ease: "linear", delay: 0.7 }}
          />
        )}

        {/* Connexions entre couches */}
        {layerXs.slice(0, -1).map((x1, L) => (
          <g key={`layerconn-${L}`}>
            {layerYs[L].map((y1, i) =>
              layerYs[L + 1].map((y2, j) => (
                <line key={`l-${L}-${i}-${j}`}
                  x1={x1 + 14} y1={y1} x2={layerXs[L + 1] - 14} y2={y2}
                  stroke={activeLayer === L + 1 && showForwardFlow ? LAYER_COLORS[L + 1] : 'var(--border-default)'}
                  strokeWidth={activeLayer === L + 1 && showForwardFlow ? 1.2 : 0.4}
                  opacity={activeLayer === L + 1 && showForwardFlow ? 0.7 : 0.3}
                />
              ))
            )}
            <g transform={`translate(${(x1 + layerXs[L + 1]) / 2}, ${cy - 100})`}>
              <rect x={-30} y={-12} width={60} height={20} rx={4}
                fill="var(--bg-base)" stroke={LAYER_COLORS[L + 1]} strokeWidth={1}/>
              <text x={0} y={2} textAnchor="middle" fontSize={11}
                fill={LAYER_COLORS[L + 1]} fontFamily="monospace" fontWeight="bold">
                W_xh⁽{L + 2}⁾
              </text>
            </g>
          </g>
        ))}

        {/* Neurones par couche */}
        {layerYs.map((ys, L) =>
          ys.map((y, j) => (
            <motion.g key={`n-${L}-${j}`}
              animate={{
                scale: activeLayer === L && showForwardFlow ? [1, 1.2, 1] : 1,
              }}
              transition={{ duration: 0.5, delay: j * 0.08 }}
            >
              <circle cx={layerXs[L]} cy={y} r={14}
                fill={activeLayer === L && showForwardFlow ? `${LAYER_COLORS[L]}50` : 'var(--bg-card)'}
                stroke={LAYER_COLORS[L]}
                strokeWidth={activeLayer === L && showForwardFlow ? 2.5 : 1}
              />
              {activeLayer === L && showForwardFlow && (
                <circle cx={layerXs[L]} cy={y} r={20}
                  fill="none" stroke={LAYER_COLORS[L]} strokeWidth={1} opacity={0.4}/>
              )}
              <text x={layerXs[L]} y={y + 3} textAnchor="middle" fontSize={9}
                fill={LAYER_COLORS[L]} fontFamily="monospace" fontWeight="bold">
                h{j + 1}
              </text>
            </motion.g>
          ))
        )}

        {/* Boucles récurrentes W_hh */}
        {layerXs.map((x, L) => (
          <g key={`rec-${L}`}>
            <path
              d={`M ${x + 14} ${cy - 50}
                  C ${x + 60} ${cy - 90}, ${x + 60} ${cy + 90}, ${x + 14} ${cy + 50}`}
              fill="none"
              stroke={activeLayer === L && showForwardFlow ? LAYER_COLORS[L] : 'var(--border-strong)'}
              strokeWidth={activeLayer === L && showForwardFlow ? 1.5 : 0.6}
              opacity={activeLayer === L && showForwardFlow ? 0.8 : 0.4}
              markerEnd="url(#arrow)"
              strokeDasharray="3,2"
            />
            <text x={x + 70} y={cy + 4} fontSize={10} fill={LAYER_COLORS[L]}
              fontFamily="monospace" fontWeight="bold">
              W_hh⁽{L + 1}⁾
            </text>
            <text x={x + 70} y={cy + 18} fontSize={9} fill="var(--text-muted)"
              fontFamily="monospace">↺ mémoire</text>
          </g>
        ))}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 6 — Sortie ŷ                                     */}
        {/* ═══════════════════════════════════════════════════════ */}
        <text x={outputX} y={20} textAnchor="middle" fontSize={11}
          fill="#f06292" fontFamily="monospace" fontWeight="bold">SORTIE</text>
        <text x={outputX} y={34} textAnchor="middle" fontSize={9}
          fill="var(--text-muted)" fontFamily="monospace">RUL prédit</text>

        {layerYs[n_layers - 1].map((ly, i) => (
          <line key={`out-${i}`}
            x1={lastLayerX + 14} y1={ly}
            x2={outputX - 20} y2={cy}
            stroke={mode === 'predict' ? '#f06292' : 'var(--border-default)'}
            strokeWidth={mode === 'predict' ? 1.5 : 0.5}
            opacity={mode === 'predict' ? 0.8 : 0.3}
          />
        ))}
        {mode === 'predict' && (
          <motion.circle r={5} fill="#f06292"
            initial={{ cx: lastLayerX + 14, cy: cy }}
            animate={{ cx: [lastLayerX + 14, outputX - 20] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
        )}
        <g transform={`translate(${(lastLayerX + outputX) / 2}, ${cy - 100})`}>
          <rect x={-26} y={-12} width={52} height={20} rx={4}
            fill="var(--bg-base)" stroke="#f06292" strokeWidth={1}/>
          <text x={0} y={2} textAnchor="middle" fontSize={11}
            fill="#f06292" fontFamily="monospace" fontWeight="bold">W_out</text>
        </g>

        <motion.g animate={{ scale: mode === 'predict' ? [1, 1.3, 1] : 1 }}>
          <circle cx={outputX} cy={cy} r={22}
            fill={mode === 'predict' ? '#f0629260' : 'var(--bg-card)'}
            stroke="#f06292" strokeWidth={2}/>
          {mode === 'predict' && (
            <circle cx={outputX} cy={cy} r={28}
              fill="none" stroke="#f06292" strokeWidth={1} opacity={0.4}/>
          )}
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

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 7 — BACKPROP — onde de choc + flèches inverses */}
        {/* ═══════════════════════════════════════════════════════ */}
        {showBackpropFlow && (
          <>
            {/* Onde de choc (cercle qui s'étend depuis ŷ) */}
            <motion.circle cx={outputX} cy={cy}
              fill="url(#backpropWave)"
              animate={{ r: [0, 800] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
              opacity={0.5}
            />
            {/* Flèches rouges depuis ŷ vers les couches */}
            {layerYs[n_layers - 1].map((ly, i) => {
              const norm = backward?.layer_grads?.[n_layers - 1]?.norm_dW_xh || 1;
              const w = Math.min(4, 1 + norm * 0.5);
              return (
                <line key={`bp-out-${i}`}
                  x1={outputX - 22} y1={cy}
                  x2={lastLayerX + 14} y2={ly}
                  stroke="#f06292" strokeWidth={w}
                  markerEnd="url(#arrowRed)"
                  opacity={0.85}/>
              );
            })}
            {/* Flèches inversées entre couches */}
            {layerXs.slice(0, -1).map((x1, L) => (
              <g key={`bp-l-${L}`}>
                {layerYs[L + 1].map((y2, j) =>
                  layerYs[L].map((y1, i) => {
                    const norm = backward?.layer_grads?.[L]?.norm_dW_xh || 1;
                    const w = Math.min(2, 0.5 + norm * 0.3);
                    return (
                      <line key={`bp-${L}-${j}-${i}`}
                        x1={layerXs[L + 1] - 14} y1={y2}
                        x2={x1 + 14} y2={y1}
                        stroke="#f06292" strokeWidth={w * 0.6}
                        opacity={0.5}/>
                    );
                  })
                )}
              </g>
            ))}
            {/* Flèches inversées Couche 1 → INPUT */}
            {layerYs[0].map((ly, j) =>
              inputYs.map((iy, i) => (
                <line key={`bp-in-${j}-${i}`}
                  x1={layerXs[0] - 14} y1={ly}
                  x2={inputX + 22} y2={iy}
                  stroke="#f06292" strokeWidth={0.5}
                  opacity={0.4}/>
              ))
            )}
            {/* Étiquettes gradients */}
            <g transform={`translate(${(lastLayerX + outputX) / 2}, ${cy + 100})`}>
              <rect x={-40} y={-12} width={80} height={20} rx={4}
                fill="var(--tint-error-bg)" stroke="#f06292" strokeWidth={1}/>
              <text x={0} y={2} textAnchor="middle" fontSize={10}
                fill="#f06292" fontFamily="monospace" fontWeight="bold">
                ∂L/∂W_out
              </text>
            </g>
            {layerXs.map((x, L) => (
              <g key={`grad-${L}`} transform={`translate(${x}, ${cy + 100})`}>
                <rect x={-40} y={-12} width={80} height={20} rx={4}
                  fill="var(--tint-error-bg)" stroke="#f06292" strokeWidth={1}/>
                <text x={0} y={2} textAnchor="middle" fontSize={10}
                  fill="#f06292" fontFamily="monospace" fontWeight="bold">
                  ∂L/∂W⁽{L + 1}⁾
                </text>
              </g>
            ))}
          </>
        )}

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 8 — Zoom sur un neurone (en bas)                */}
        {/* ═══════════════════════════════════════════════════════ */}
        <line x1={50} y1={zoomY - 30} x2={W - 50} y2={zoomY - 30}
          stroke="var(--border-default)" strokeWidth={1} strokeDasharray="4,4"/>
        <text x={W / 2} y={zoomY - 18} textAnchor="middle" fontSize={11}
          fill="var(--accent-purple)" fontFamily="monospace" fontWeight="bold">
          🔍 ZOOM SUR UN NEURONE — voici ce qui se passe à l'intérieur
        </text>

        {/* Schéma zoom : Z entre → tanh → h sort */}
        <g transform={`translate(${W / 2 - 280}, ${zoomY + 5})`}>
          {/* Z entrée */}
          <rect x={0} y={0} width={70} height={50} rx={5}
            fill="#3a2a0d" stroke="var(--accent-orange)" strokeWidth={2}/>
          <text x={35} y={20} textAnchor="middle" fontSize={11}
            fill="var(--accent-orange)" fontFamily="monospace" fontWeight="bold">
            Z entrée
          </text>
          <text x={35} y={36} textAnchor="middle" fontSize={11}
            fill="#fff" fontFamily="monospace">
            {currentZ.toFixed(3)}
          </text>

          {/* Flèche */}
          <line x1={75} y1={25} x2={120} y2={25}
            stroke="var(--accent-orange)" strokeWidth={2} markerEnd="url(#arrow)"/>

          {/* Boîte tanh */}
          <rect x={125} y={-5} width={130} height={60} rx={5}
            fill="var(--tint-success-bg)" stroke="var(--success)" strokeWidth={2}/>
          <text x={190} y={10} textAnchor="middle" fontSize={10}
            fill="var(--success)" fontFamily="monospace" fontWeight="bold">
            tanh
          </text>
          {/* Courbe tanh dans la boîte */}
          <path d="M 130 50 Q 160 50, 190 25 T 250 0"
            fill="none" stroke="var(--success)" strokeWidth={2}/>
          {/* Point sur la courbe selon Z */}
          {showForwardFlow && (
            <motion.circle r={5} fill="#fff"
              cx={190 + Math.tanh(currentZ) * 60}
              cy={25 - Math.tanh(currentZ) * 25}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5 }}
            />
          )}
          {/* Axes */}
          <line x1={130} y1={25} x2={250} y2={25}
            stroke="var(--success)" strokeWidth={0.3} opacity={0.4}/>
          <line x1={190} y1={0} x2={190} y2={50}
            stroke="var(--success)" strokeWidth={0.3} opacity={0.4}/>
          <text x={245} y={48} fontSize={8} fill="var(--text-muted)">+1</text>
          <text x={245} y={8} fontSize={8} fill="var(--text-muted)">-1</text>

          {/* Flèche */}
          <line x1={260} y1={25} x2={305} y2={25}
            stroke="var(--success)" strokeWidth={2} markerEnd="url(#arrow)"/>

          {/* h sortie */}
          <rect x={310} y={0} width={70} height={50} rx={5}
            fill="var(--tint-success-bg)" stroke="var(--success)" strokeWidth={2}/>
          <text x={345} y={20} textAnchor="middle" fontSize={11}
            fill="var(--success)" fontFamily="monospace" fontWeight="bold">
            h sortie
          </text>
          <text x={345} y={36} textAnchor="middle" fontSize={11}
            fill="#fff" fontFamily="monospace">
            {currentH.toFixed(3)}
          </text>

          {/* Explication à droite */}
          <text x={400} y={15} fontSize={11}
            fill="var(--text-secondary)" fontFamily="monospace" fontWeight="bold">
            🧠 La fonction tanh "écrase"
          </text>
          <text x={400} y={32} fontSize={10}
            fill="var(--text-tertiary)" fontFamily="monospace">
            les valeurs entre -1 et +1
          </text>
          <text x={400} y={47} fontSize={10}
            fill="var(--text-tertiary)" fontFamily="monospace">
            → 0 = neutre · ±1 = saturé
          </text>
        </g>

        {/* ═══════════════════════════════════════════════════════ */}
        {/* SECTION 9 — Bandeau narration en bas                    */}
        {/* ═══════════════════════════════════════════════════════ */}
        <rect x={0} y={H - 36} width={W} height={36}
          fill="var(--tint-purple-bg)" opacity={0.95}/>
        <line x1={0} y1={H - 36} x2={W} y2={H - 36}
          stroke={mode === 'forward' ? 'var(--accent-blue)' : mode === 'predict' ? '#f06292' : mode === 'backprop' ? '#f06292' : 'var(--border-strong)'}
          strokeWidth={2}/>
        <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={12}
          fill="var(--text-primary)" fontFamily="monospace">
          {narrationText}
        </text>
      </svg>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sélecteur d'architecture
// ═══════════════════════════════════════════════════════════════
function ArchSelector({ value, onChange }) {
  const archs = [
    { id: 'rnn',         label: 'RNN',         color: 'var(--accent-blue)', desc: 'Réseau récurrent simple', enabled: true },
    { id: 'lstm',        label: 'LSTM',        color: 'var(--accent-purple)', desc: '4 portes + cellule', enabled: true },
    { id: 'gru',         label: 'GRU',         color: 'var(--accent-orange)', desc: '2 portes simplifiées', enabled: false },
    { id: 'transformer', label: 'Transformer', color: 'var(--accent-green)', desc: 'Self-attention', enabled: false },
  ];

  return (
    <div>
      <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
        🧠 ARCHITECTURE
      </label>
      <div className="grid grid-cols-2 gap-1.5">
        {archs.map(a => (
          <button key={a.id}
            onClick={() => a.enabled && onChange(a.id)}
            disabled={!a.enabled}
            className="px-2 py-2 rounded-lg text-xs font-bold border transition-all relative"
            style={{
              background: value === a.id ? `${a.color}20` : 'var(--bg-elevated)',
              borderColor: value === a.id ? a.color : 'var(--border-default)',
              color: value === a.id ? a.color : a.enabled ? 'var(--text-tertiary)' : 'var(--text-muted)',
              cursor: a.enabled ? 'pointer' : 'not-allowed',
              opacity: a.enabled ? 1 : 0.5,
            }}>
            <div className="flex items-center justify-center gap-1">
              {!a.enabled && <Lock size={9} />} {a.label}
            </div>
            <div className="text-[9px] mt-0.5" style={{ color:'var(--text-muted)' }}>
              {a.enabled ? a.desc : 'Bientôt'}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sidebar configuration
// ═══════════════════════════════════════════════════════════════
function DemoSidebar({ config, setConfig, arch, setArch, onRun, loading }) {
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
          Configuration
        </p>
      </div>

      <ArchSelector value={arch} onChange={setArch} />

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          NOMBRE DE COUCHES
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {[1, 2, 3].map(n => (
            <button key={n}
              onClick={() => setLayers(n)}
              className="px-2 py-2 rounded-lg text-sm font-bold border transition-all"
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
          NEURONES / COUCHE
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
        {loading ? '⚙️ Calcul...' : <><Zap size={14}/> Lancer la démo</>}
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// InputFlow : Tenseur 3D éclaté qui défile vers le modèle
// ═══════════════════════════════════════════════════════════════
function InputFlow({ tensor3D, slices, featNames, batchSize, seqLength, activeT = -1 }) {
  if (!tensor3D || !slices) return null;
  const seqLabels = Array.from({length: batchSize}, (_, i) => String.fromCharCode(65 + i));

  // ─── Cube isométrique en SVG ───
  // Dimensions du cube
  const cellSize = 22;            // taille d'une cellule
  const depth   = batchSize * cellSize;       // profondeur (axe Batch)
  const width   = featNames.length * cellSize; // largeur (axe Features)
  const heightT = seqLength * cellSize;        // hauteur (axe Time)

  // Décalage isométrique (ratio 0.5 pour l'effet 3D)
  const isoX = 0.7;
  const isoY = 0.4;

  // Fonction pour projeter un point 3D (i,j,k) en 2D
  // i = batch, j = time, k = feature
  const project = (i, j, k, originX, originY) => ({
    x: originX + k * cellSize + i * cellSize * isoX,
    y: originY + j * cellSize - i * cellSize * isoY,
  });

  // Origine du cube (en haut à gauche du SVG)
  const ox = 50;
  const oy = 80 + depth * isoY;

  // SVG dimensions
  const svgW = ox + width + depth * isoX + 50;
  const svgH = oy + heightT + 60;

  // Couleur d'une cellule selon valeur
  const cellColor = (v) => {
    const ratio = Math.max(-1, Math.min(1, v * 2 - 1));
    if (ratio >= 0) return `rgba(79, 195, 247, ${0.2 + ratio * 0.6})`;
    return `rgba(240, 98, 146, ${0.2 + Math.abs(ratio) * 0.6})`;
  };

  return (
    <div className="rounded-xl border p-4 space-y-4"
      style={{ background: 'var(--bg-base)', borderColor: '#ce93d840' }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Box size={14} style={{ color: 'var(--accent-purple)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--accent-purple)' }}>
            🎬 Le tenseur 3D et son découpage en {seqLength} tranches 2D
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          [{batchSize} × {seqLength} × {featNames.length}]
        </span>
      </div>

      {/* Layout : cube 3D | flèche | tranches 2D */}
      <div className="flex items-start gap-3 overflow-x-auto pb-2">

        {/* ═══════════ CUBE ISOMÉTRIQUE 3D ═══════════ */}
        <div className="flex-shrink-0">
          <p className="text-xs text-center mb-1 font-bold" style={{ color: 'var(--accent-purple)' }}>
            🧊 Tenseur X (3D réel)
          </p>
          <svg width={svgW} height={svgH}>
            <defs>
              <marker id="iso-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3"
                orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L6,3 z" fill="var(--text-tertiary)"/>
              </marker>
            </defs>

            {/* ── Face du dessus (Batch × Features, vue d'en haut) ── */}
            {Array.from({ length: batchSize }).map((_, i) =>
              Array.from({ length: featNames.length }).map((_, k) => {
                const p1 = project(i, 0, k, ox, oy);
                const p2 = project(i, 0, k + 1, ox, oy);
                const p3 = project(i + 1, 0, k + 1, ox, oy);
                const p4 = project(i + 1, 0, k, ox, oy);
                const v = tensor3D?.[i]?.[0]?.[k] ?? 0;
                return (
                  <polygon key={`top-${i}-${k}`}
                    points={`${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y} ${p4.x},${p4.y}`}
                    fill={cellColor(v)}
                    stroke="var(--accent-purple)"
                    strokeWidth={0.4}
                    opacity={0.85}
                  />
                );
              })
            )}

            {/* ── Face avant (Time × Features) ── */}
            {Array.from({ length: seqLength }).map((_, j) =>
              Array.from({ length: featNames.length }).map((_, k) => {
                const p1 = project(0, j, k, ox, oy);
                const p2 = project(0, j, k + 1, ox, oy);
                const p3 = project(0, j + 1, k + 1, ox, oy);
                const p4 = project(0, j + 1, k, ox, oy);
                const v = tensor3D?.[0]?.[j]?.[k] ?? 0;
                const isActiveT = activeT === j;
                return (
                  <polygon key={`front-${j}-${k}`}
                    points={`${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y} ${p4.x},${p4.y}`}
                    fill={cellColor(v)}
                    stroke={isActiveT ? 'var(--accent-blue)' : 'var(--accent-purple)'}
                    strokeWidth={isActiveT ? 1.5 : 0.5}
                    opacity={1}
                  />
                );
              })
            )}

            {/* ── Face droite (Time × Batch) ── */}
            {Array.from({ length: seqLength }).map((_, j) =>
              Array.from({ length: batchSize }).map((_, i) => {
                const p1 = project(i, j, featNames.length, ox, oy);
                const p2 = project(i + 1, j, featNames.length, ox, oy);
                const p3 = project(i + 1, j + 1, featNames.length, ox, oy);
                const p4 = project(i, j + 1, featNames.length, ox, oy);
                const v = tensor3D?.[i]?.[j]?.[featNames.length - 1] ?? 0;
                return (
                  <polygon key={`right-${j}-${i}`}
                    points={`${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y} ${p4.x},${p4.y}`}
                    fill={cellColor(v * 0.7)}
                    stroke="var(--accent-purple)"
                    strokeWidth={0.4}
                    opacity={0.6}
                  />
                );
              })
            )}

            {/* ── Surlignage de la tranche active ── */}
            {activeT >= 0 && activeT < seqLength && (() => {
              // Contour de la tranche t = activeT (face supérieure de la tranche)
              const p1 = project(0, activeT, 0, ox, oy);
              const p2 = project(0, activeT, featNames.length, ox, oy);
              const p3 = project(batchSize, activeT, featNames.length, ox, oy);
              const p4 = project(batchSize, activeT, 0, ox, oy);
              return (
                <motion.polygon
                  points={`${p1.x},${p1.y} ${p2.x},${p2.y} ${p3.x},${p3.y} ${p4.x},${p4.y}`}
                  fill="none"
                  stroke="var(--accent-blue)"
                  strokeWidth={3}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              );
            })()}

            {/* ── Axes étiquetés ── */}
            {/* Axe Features (largeur, vers la droite) */}
            <g>
              <line
                x1={ox} y1={oy + heightT + 12}
                x2={ox + width} y2={oy + heightT + 12}
                stroke="var(--accent-blue)" strokeWidth={1.5}
                markerEnd="url(#iso-arrow)"
              />
              <text x={ox + width / 2} y={oy + heightT + 28}
                textAnchor="middle" fontSize={10}
                fill="var(--accent-blue)" fontFamily="monospace" fontWeight="bold">
                Features ({featNames.length})
              </text>
              {/* Labels des features sous le cube */}
              {featNames.map((name, k) => (
                <text key={`f-${k}`}
                  x={ox + k * cellSize + cellSize / 2}
                  y={oy + heightT + 44}
                  textAnchor="middle" fontSize={8}
                  fill="var(--text-tertiary)" fontFamily="monospace">
                  {name.slice(0, 4)}
                </text>
              ))}
            </g>

            {/* Axe Time (vertical) */}
            <g>
              <line
                x1={ox - 12} y1={oy}
                x2={ox - 12} y2={oy + heightT}
                stroke="var(--accent-green)" strokeWidth={1.5}
                markerEnd="url(#iso-arrow)"
              />
              <text x={ox - 24} y={oy + heightT / 2}
                textAnchor="middle" fontSize={10}
                fill="var(--accent-green)" fontFamily="monospace" fontWeight="bold"
                transform={`rotate(-90, ${ox - 24}, ${oy + heightT / 2})`}>
                Time ({seqLength})
              </text>
              {/* Labels des temps */}
              {Array.from({ length: seqLength }).map((_, j) => (
                <text key={`t-${j}`}
                  x={ox - 4}
                  y={oy + j * cellSize + cellSize / 2 + 4}
                  textAnchor="end" fontSize={9}
                  fill={activeT === j ? 'var(--accent-blue)' : 'var(--text-tertiary)'}
                  fontFamily="monospace"
                  fontWeight={activeT === j ? 'bold' : 'normal'}>
                  t{j + 1}
                </text>
              ))}
            </g>

            {/* Axe Batch (vers l'arrière en isométrique) */}
            <g>
              <line
                x1={ox + width + 4} y1={oy + 4}
                x2={ox + width + depth * isoX + 4} y2={oy - depth * isoY + 4}
                stroke="var(--accent-orange)" strokeWidth={1.5}
                markerEnd="url(#iso-arrow)"
              />
              <text
                x={ox + width + depth * isoX / 2 + 8}
                y={oy - depth * isoY / 2 - 4}
                textAnchor="start" fontSize={10}
                fill="var(--accent-orange)" fontFamily="monospace" fontWeight="bold">
                Batch ({batchSize})
              </text>
              {/* Labels des séquences */}
              {seqLabels.map((lab, i) => {
                const p = project(i, 0, featNames.length, ox, oy);
                return (
                  <text key={`b-${i}`}
                    x={p.x + 5} y={p.y - 2}
                    fontSize={8}
                    fill="var(--text-tertiary)" fontFamily="monospace">
                    Séq {lab}
                  </text>
                );
              })}
            </g>

            {/* Légende totale */}
            <text x={svgW - 10} y={svgH - 8}
              textAnchor="end" fontSize={9}
              fill="var(--text-muted)" fontFamily="monospace">
              {batchSize} × {seqLength} × {featNames.length} = {batchSize * seqLength * featNames.length} valeurs
            </text>
          </svg>
        </div>

        {/* ═══════════ FLÈCHE EXPLICITE ═══════════ */}
        <motion.div
          animate={{ x: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="flex flex-col items-center justify-center"
          style={{ minHeight: svgH }}
        >
          <p className="text-xs mb-2 text-center font-bold" style={{ color: 'var(--accent-purple)' }}>
            Découpage<br/>temporel
          </p>
          <ArrowRight size={32} style={{ color: 'var(--accent-purple)' }}/>
          <p className="text-xs mt-2 text-center" style={{ color: 'var(--text-tertiary)' }}>
            On extrait<br/>{seqLength} tranches<br/>2D
          </p>
          <p className="text-[10px] mt-2 font-mono text-center" style={{ color: 'var(--text-muted)' }}>
            X[:, t, :] pour t=1..{seqLength}
          </p>
        </motion.div>

        {/* ═══════════ TRANCHES 2D ═══════════ */}
        <div>
          <p className="text-xs text-center mb-1 font-bold" style={{ color: 'var(--accent-green)' }}>
            📑 {seqLength} tranches 2D
          </p>
          <div className="flex items-end gap-2">
            {slices.map((slice, t) => {
              const isActive  = activeT === t;
              const isPast    = activeT > t;
              return (
                <motion.div key={t}
                  animate={{
                    y: isActive ? -8 : 0,
                    scale: isActive ? 1.05 : isPast ? 0.9 : 1,
                    opacity: isPast ? 0.4 : 1,
                  }}
                  transition={{ duration: 0.4 }}
                  className="flex flex-col items-center"
                >
                  <span className="text-xs font-mono mb-1 font-bold"
                    style={{ color: isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--accent-green)' }}>
                    X<sub>{t + 1}</sub>
                    {isActive && ' ⬅️'}
                  </span>
                  <Matrix
                    data={slice}
                    rowLabels={seqLabels}
                    colLabels={featNames}
                    color={isActive ? 'var(--accent-blue)' : isPast ? 'var(--text-muted)' : 'var(--accent-green)'}
                    decimals={3}
                    small
                    highlight={isActive}
                  />
                  <span className="text-[9px] mt-1" style={{ color: 'var(--text-muted)' }}>
                    [{batchSize}×{featNames.length}]
                    {isPast && ' ✓'}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      <p className="text-xs italic px-3 py-2 rounded" style={{
        background: 'var(--bg-card)', color: 'var(--text-tertiary)', borderLeft: '2px solid var(--accent-purple)'
      }}>
        💡 <b>Comment lire le cube ?</b><br/>
        • <span style={{color:'var(--accent-blue)'}}>Axe horizontal (Features)</span> : les {featNames.length} capteurs (volt, rotate, pressure, vibration)<br/>
        • <span style={{color:'var(--accent-green)'}}>Axe vertical (Time)</span> : les {seqLength} heures consécutives<br/>
        • <span style={{color:'var(--accent-orange)'}}>Axe profondeur (Batch)</span> : les {batchSize} séquences traitées en parallèle<br/>
        Le RNN <b>ne lit pas le cube d'un coup</b> — il découpe selon l'axe Time
        pour obtenir <InlineMath math={`X_t \\in \\mathbb{R}^{${batchSize} \\times ${featNames.length}}`} />
        à chaque pas de temps.
      </p>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MatMulVisual : Animation ligne × colonne pour la multiplication
// ═══════════════════════════════════════════════════════════════
function MatMulVisual({
  matA, matB, result,
  labelA = 'X', labelB = 'W', labelC = 'Z',
  rowLabelsA, colLabelsA, rowLabelsB, colLabelsB,
  colorA = 'var(--accent-blue)', colorB = 'var(--accent-orange)', colorC = 'var(--success)',
  hoverRow = 0, hoverCol = 0,
  decimals = 3,
}) {
  if (!matA || !matB || !result) return null;
  const rowsA = Array.isArray(matA[0]) ? matA : [matA];
  const rowsB = Array.isArray(matB[0]) ? matB : [matB];
  const rowsC = Array.isArray(result[0]) ? result : [result];

  const cellW = 50, cellH = 24;

  // Calcul détaillé de la cellule active
  const aRow = rowsA[hoverRow] || [];
  const bCol = rowsB.map(row => row[hoverCol] || 0);
  const products = aRow.map((a, i) => a * (bCol[i] || 0));
  const sum = products.reduce((s, p) => s + p, 0);

  const renderCell = (v, eff, isHighlighted, isResult, color) => (
    <div className="flex items-center justify-center text-xs font-mono border"
      style={{
        width: cellW, height: cellH,
        background: isHighlighted ? `${color}80` : valueColor(v, eff),
        color: isHighlighted ? '#fff' : valueText(v, eff),
        borderColor: isHighlighted ? color : 'var(--bg-card-alt)',
        borderWidth: isHighlighted ? 1.5 : 0.5,
        fontWeight: isHighlighted ? 'bold' : Math.abs(v) > eff * 0.5 ? 'bold' : 'normal',
        boxShadow: isResult ? `0 0 8px ${color}` : 'none',
      }}>
      {typeof v === 'number' ? v.toFixed(decimals) : v}
    </div>
  );

  const effA = Math.max(...rowsA.flat().map(Math.abs), 0.001);
  const effB = Math.max(...rowsB.flat().map(Math.abs), 0.001);
  const effC = Math.max(...rowsC.flat().map(Math.abs), 0.001);

  return (
    <div className="space-y-3">
      {/* Les 3 matrices côte à côte */}
      <div className="flex items-center gap-2 flex-wrap justify-center">
        {/* Matrice A */}
        <div className="flex flex-col items-center">
          <span className="text-sm font-bold mb-1" style={{ color: colorA }}>
            {labelA} <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              [{rowsA.length}×{rowsA[0].length}]
            </span>
          </span>
          <div className="relative inline-block" style={{ padding: 4 }}>
            <div style={{
              position:'absolute', left:-2, top:0, bottom:0, width:6,
              borderLeft:`2px solid ${colorA}`,
              borderTop:`2px solid ${colorA}`, borderBottom:`2px solid ${colorA}`,
            }}/>
            <div style={{
              position:'absolute', right:-2, top:0, bottom:0, width:6,
              borderRight:`2px solid ${colorA}`,
              borderTop:`2px solid ${colorA}`, borderBottom:`2px solid ${colorA}`,
            }}/>
            <div className="flex">
              {rowLabelsA && (
                <div className="flex flex-col mr-1 justify-center">
                  {rowLabelsA.map((lab, i) => (
                    <div key={i} className="flex items-center justify-end pr-2 text-xs font-mono"
                      style={{ height: cellH, color: i === hoverRow ? colorA : 'var(--text-tertiary)',
                               minWidth: 42, fontWeight: i === hoverRow ? 'bold' : 'normal' }}>
                      {i === hoverRow && '▶'} {lab}
                    </div>
                  ))}
                </div>
              )}
              <div>
                {colLabelsA && (
                  <div className="flex">
                    {colLabelsA.map((lab, j) => (
                      <div key={j} className="text-center text-xs font-mono pb-1"
                        style={{ width: cellW, color: 'var(--text-muted)' }}>{lab}</div>
                    ))}
                  </div>
                )}
                {rowsA.map((row, i) => (
                  <div key={i} className="flex"
                    style={{ background: i === hoverRow ? `${colorA}20` : 'transparent' }}>
                    {row.map((v, j) => renderCell(v, effA, i === hoverRow, false, colorA))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Symbole multiplication */}
        <div className="text-3xl font-bold mx-1" style={{ color: 'var(--text-tertiary)' }}>·</div>

        {/* Matrice B */}
        <div className="flex flex-col items-center">
          <span className="text-sm font-bold mb-1" style={{ color: colorB }}>
            {labelB} <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              [{rowsB.length}×{rowsB[0].length}]
            </span>
          </span>
          <div className="relative inline-block" style={{ padding: 4 }}>
            <div style={{
              position:'absolute', left:-2, top:0, bottom:0, width:6,
              borderLeft:`2px solid ${colorB}`,
              borderTop:`2px solid ${colorB}`, borderBottom:`2px solid ${colorB}`,
            }}/>
            <div style={{
              position:'absolute', right:-2, top:0, bottom:0, width:6,
              borderRight:`2px solid ${colorB}`,
              borderTop:`2px solid ${colorB}`, borderBottom:`2px solid ${colorB}`,
            }}/>
            <div className="flex">
              {rowLabelsB && (
                <div className="flex flex-col mr-1 justify-center">
                  {rowLabelsB.map((lab, i) => (
                    <div key={i} className="flex items-center justify-end pr-2 text-xs font-mono"
                      style={{ height: cellH, color: 'var(--text-tertiary)', minWidth: 42 }}>
                      {lab}
                    </div>
                  ))}
                </div>
              )}
              <div>
                {colLabelsB && (
                  <div className="flex">
                    {colLabelsB.map((lab, j) => (
                      <div key={j} className="text-center text-xs font-mono pb-1"
                        style={{
                          width: cellW,
                          color: j === hoverCol ? colorB : 'var(--text-muted)',
                          fontWeight: j === hoverCol ? 'bold' : 'normal',
                        }}>
                        {j === hoverCol && '▼'} {lab}
                      </div>
                    ))}
                  </div>
                )}
                {rowsB.map((row, i) => (
                  <div key={i} className="flex">
                    {row.map((v, j) => (
                      <div key={j}
                        style={{ background: j === hoverCol ? `${colorB}20` : 'transparent' }}>
                        {renderCell(v, effB, j === hoverCol, false, colorB)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Symbole = */}
        <div className="text-2xl font-bold mx-1" style={{ color: 'var(--text-tertiary)' }}>=</div>

        {/* Matrice résultat */}
        <div className="flex flex-col items-center">
          <span className="text-sm font-bold mb-1" style={{ color: colorC }}>
            {labelC} <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              [{rowsC.length}×{rowsC[0].length}]
            </span>
          </span>
          <div className="relative inline-block" style={{ padding: 4 }}>
            <div style={{
              position:'absolute', left:-2, top:0, bottom:0, width:6,
              borderLeft:`2px solid ${colorC}`,
              borderTop:`2px solid ${colorC}`, borderBottom:`2px solid ${colorC}`,
            }}/>
            <div style={{
              position:'absolute', right:-2, top:0, bottom:0, width:6,
              borderRight:`2px solid ${colorC}`,
              borderTop:`2px solid ${colorC}`, borderBottom:`2px solid ${colorC}`,
            }}/>
            <div>
              {rowsC.map((row, i) => (
                <div key={i} className="flex">
                  {row.map((v, j) =>
                    renderCell(v, effC, i === hoverRow && j === hoverCol,
                               i === hoverRow && j === hoverCol, colorC)
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Calcul détaillé de la cellule active */}
      <motion.div
        key={`${hoverRow}-${hoverCol}`}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="rounded-lg p-3"
        style={{ background: 'var(--bg-card)', border: '1px solid #4caf5040' }}
      >
        <p className="text-xs font-semibold mb-2" style={{ color: 'var(--success)' }}>
          🔍 Calcul cellule {labelC}[{rowLabelsA?.[hoverRow] || hoverRow}, {colLabelsB?.[hoverCol] || hoverCol}] :
        </p>
        <div className="font-mono text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          <div className="mb-1" style={{ color: 'var(--text-tertiary)' }}>
            = (ligne {hoverRow + 1} de {labelA}) · (colonne {hoverCol + 1} de {labelB})
          </div>
          <div className="mb-1">
            = {aRow.map((a, i) => (
              <span key={i}>
                <span style={{ color: colorA }}>{a.toFixed(decimals)}</span>
                <span style={{ color: 'var(--text-muted)' }}> × </span>
                <span style={{ color: colorB }}>{(bCol[i] || 0).toFixed(decimals)}</span>
                {i < aRow.length - 1 && <span style={{ color: 'var(--text-muted)' }}> + </span>}
              </span>
            ))}
          </div>
          <div className="mb-1" style={{ color: 'var(--text-tertiary)' }}>
            = {products.map((p, i) => (
              <span key={i}>
                <span style={{ color: p >= 0 ? 'var(--accent-blue)' : '#f06292' }}>
                  {p >= 0 ? '+' : ''}{p.toFixed(decimals)}
                </span>
                {i < products.length - 1 && ' '}
              </span>
            ))}
          </div>
          <div className="font-bold pt-1 border-t" style={{ borderColor: 'var(--border-default)', color: colorC }}>
            = {sum.toFixed(decimals)}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CycleOverview : Les 5 étapes du cycle d'apprentissage
// ═══════════════════════════════════════════════════════════════
function CycleOverview() {
  const steps = [
    {
      num: 1,
      title: 'Forward',
      icon: '➡️',
      desc: 'Calculer la prédiction',
      formula: String.raw`\hat{y} = f(x ; W)`,
      color: 'var(--accent-blue)',
      explanation: 'Les données d\'entrée x traversent le réseau couche par couche jusqu\'à produire une prédiction ŷ.',
    },
    {
      num: 2,
      title: 'Loss',
      icon: '🎯',
      desc: 'Mesurer l\'erreur',
      formula: String.raw`\mathcal{L} = (y - \hat{y})^2`,
      color: 'var(--accent-orange)',
      explanation: 'On compare la prédiction ŷ avec la vraie valeur y. Plus elles sont éloignées, plus la loss est grande.',
    },
    {
      num: 3,
      title: 'Backward',
      icon: '⬅️',
      desc: 'Calculer les gradients',
      formula: String.raw`\nabla W = \frac{\partial \mathcal{L}}{\partial W}`,
      color: '#f06292',
      explanation: 'On trouve dans quelle direction ajuster chaque poids pour réduire l\'erreur (chain rule).',
    },
    {
      num: 4,
      title: 'Update',
      icon: '⚙️',
      desc: 'Mettre à jour les poids',
      formula: String.raw`W \leftarrow W - \eta \cdot \nabla W`,
      color: 'var(--accent-purple)',
      explanation: 'Les poids sont déplacés d\'un petit pas (learning rate η) dans la bonne direction.',
    },
    {
      num: 5,
      title: 'Répéter',
      icon: '🔁',
      desc: 'Sur tous les batchs',
      formula: String.raw`\text{tous batchs, toutes époques}`,
      color: 'var(--accent-green)',
      explanation: 'Le cycle se répète des milliers de fois jusqu\'à ce que la loss soit minimale.',
    },
  ];

  return (
    <div className="space-y-3">
      <div className="rounded-xl p-4 border-2"
        style={{ background: 'linear-gradient(135deg, var(--bg-card), var(--bg-base))', borderColor: 'var(--accent-blue)' }}>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-bold flex items-center gap-2" style={{ color: 'var(--accent-blue)' }}>
            <Sparkles size={14}/> Cycle d'apprentissage complet
          </h4>
          <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            5 étapes répétées
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
          {steps.map((step, i) => (
            <motion.div key={step.num}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              className="rounded-lg p-3 relative"
              style={{
                background: 'var(--bg-base)',
                border: `1px solid ${step.color}50`,
                minHeight: 150,
              }}>
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

              <p className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>
                {step.desc}
              </p>

              <div className="rounded p-1.5 mb-2 text-center"
                style={{ background: 'var(--bg-card)', minHeight: 38 }}>
                <span style={{ fontSize: '0.8em' }}>
                  <InlineMath math={step.formula}/>
                </span>
              </div>

              <p className="text-[10px] leading-snug" style={{ color: 'var(--text-tertiary)' }}>
                {step.explanation}
              </p>

              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-2 -translate-y-1/2 z-10">
                  <ChevronRight size={16} style={{ color: step.color }}/>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border-default)' }}>
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--accent-purple)' }}>
            📐 En une seule formule mathématique :
          </p>
          <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-card)' }}>
            <BlockMath math={String.raw`\underbrace{\hat{y} = f(x; W)}_{\text{1. Forward}} \;\Rightarrow\; \underbrace{\mathcal{L} = (y - \hat{y})^2}_{\text{2. Loss}} \;\Rightarrow\; \underbrace{\nabla W = \frac{\partial \mathcal{L}}{\partial W}}_{\text{3. Backward}} \;\Rightarrow\; \underbrace{W \leftarrow W - \eta \nabla W}_{\text{4. Update}}`}/>
          </div>
        </div>
      </div>
    </div>
  );
}

function NarrationPanel({ data, scene, sceneIndex }) {
  if (!data || !scene) return null;
  const { config, weights, forward_steps, prediction, loss_info, backward,
          y_true_norm, y_true_hours, input_tensor_3d, slices_2d, feat_names } = data;

  // Animation cellule par cellule pour MatMulVisual
  const [matmulCell, setMatmulCell] = useState({ row: 0, col: 0 });

  useEffect(() => {
    setMatmulCell({ row: 0, col: 0 });
    if (!scene.matmul) return;
    const result = scene.matmul.result;
    if (!result || !result.length) return;
    const nRows = result.length;
    const nCols = Array.isArray(result[0]) ? result[0].length : 1;
    let r = 0, c = 0;
    const id = setInterval(() => {
      c++;
      if (c >= nCols) { c = 0; r++; }
      if (r >= nRows) { clearInterval(id); return; }
      setMatmulCell({ row: r, col: c });
    }, 1500);
    return () => clearInterval(id);
  }, [scene.id]);

  return (
    <div className="rounded-xl border" style={{ background:'var(--bg-base)', borderColor:'#ce93d840' }}>
      <div className="px-4 py-2.5 border-b flex items-center justify-between"
        style={{ background:'var(--tint-purple-bg)', borderColor:'#ce93d840' }}>
        <div className="flex items-center gap-2">
          <BookOpen size={14} style={{ color:'var(--accent-purple)' }} />
          <span className="text-sm font-semibold" style={{ color:'var(--accent-purple)' }}>
            📝 Narration & Calculs détaillés
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
            {/* Titre de la scène */}
            <div className="mb-3">
              <p className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: scene.color || 'var(--accent-purple)' }}>
                {scene.section}
              </p>
              <h3 className="text-base font-bold mt-1" style={{ color:'var(--text-primary)' }}>
                {scene.title}
              </h3>
            </div>

            {/* Narration (texte) */}
            <p className="text-sm leading-relaxed mb-3" style={{ color:'var(--text-secondary)' }}
              dangerouslySetInnerHTML={{ __html: scene.text }}/>

            {/* CycleOverview : Aperçu des 5 étapes */}
            {scene.showCycleOverview && (
              <div className="my-4">
                <CycleOverview/>
              </div>
            )}

            {/* InputFlow : tenseur 3D éclaté */}
            {scene.showInputFlow && (
              <div className="my-4">
                <InputFlow
                  tensor3D={input_tensor_3d}
                  slices={slices_2d}
                  featNames={feat_names}
                  batchSize={config.batch_size}
                  seqLength={config.seq_length}
                  activeT={scene.activeT !== undefined ? scene.activeT : -1}
                />
              </div>
            )}

            {/* MatMulVisual : multiplication animée ligne × colonne */}
            {scene.matmul && (
              <div className="my-4 rounded-lg p-4 overflow-x-auto"
                style={{ background: 'var(--bg-deep)', border: '1px solid #4caf5040' }}>
                <p className="text-xs font-semibold mb-3" style={{ color: 'var(--success)' }}>
                  🎯 Multiplication matricielle visuelle (ligne en cours × colonne en cours)
                </p>
                <MatMulVisual
                  matA={scene.matmul.matA}
                  matB={scene.matmul.matB}
                  result={scene.matmul.result}
                  labelA={scene.matmul.labelA}
                  labelB={scene.matmul.labelB}
                  labelC={scene.matmul.labelC}
                  rowLabelsA={scene.matmul.rowLabelsA}
                  colLabelsA={scene.matmul.colLabelsA}
                  rowLabelsB={scene.matmul.rowLabelsB}
                  colLabelsB={scene.matmul.colLabelsB}
                  colorA={scene.matmul.colorA}
                  colorB={scene.matmul.colorB}
                  colorC={scene.matmul.colorC}
                  hoverRow={matmulCell.row}
                  hoverCol={matmulCell.col}
                  decimals={3}
                />
              </div>
            )}

            {/* Formule */}
            {scene.formula && (
              <div className="rounded-lg p-3 my-3" style={{ background:'var(--bg-card)' }}>
                <BlockMath math={scene.formula}/>
              </div>
            )}

            {/* Formule développée avec valeurs */}
            {scene.formulaWithValues && (
              <div className="rounded-lg p-3 my-3"
                style={{ background:'var(--tint-info-bg)', border:'1px solid #4fc3f740' }}>
                <p className="text-xs mb-1" style={{ color:'var(--text-muted)' }}>Application numérique :</p>
                <BlockMath math={scene.formulaWithValues}/>
              </div>
            )}

            {/* Matrices à afficher */}
            {scene.matrices && (
              <div className="flex flex-wrap gap-4 mt-3">
                {scene.matrices.map((m, i) => (
                  <div key={i}>
                    <Matrix
                      data={m.data}
                      title={m.title}
                      subtitle={m.subtitle}
                      rowLabels={m.rowLabels}
                      colLabels={m.colLabels}
                      color={m.color}
                      decimals={m.decimals || 3}
                      small={m.small}
                      highlight={m.highlight}
                    />
                  </div>
                ))}
              </div>
            )}

            {/* Note */}
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
// Construction du scénario complet (toutes les scènes)
// ═══════════════════════════════════════════════════════════════
function buildScenario(data) {
  if (!data) return [];
  const { config, weights, forward_steps, prediction, loss_info, backward,
          y_true_norm, y_true_hours, feat_names } = data;
  const { layers, batch_size, seq_length, n_features, n_layers } = config;

  const labelOrFirst = (arr) => arr && arr.length ? arr[0] : '?';

  const scenes = [];
  const seqLabels = Array.from({length:batch_size}, (_,i)=>`Séq ${String.fromCharCode(65+i)}`);

  // ═══ INTRODUCTION ═══
  scenes.push({
    id: 'intro',
    section: 'Introduction',
    title: '🎬 Bienvenue dans la démonstration RNN',
    color: 'var(--accent-purple)',
    text: `On va voir <b>pas à pas</b> comment un RNN traite tes données. <br/>
           Architecture choisie : <b>${n_layers} couche${n_layers>1?'s':''}</b>,
           batch de <b>${batch_size} séquences</b> sur <b>${seq_length} heures</b>,
           ${n_features} features par heure.`,
    note: `Tous les nombres affichés proviennent d'un mini-dataset synthétique pédagogique généré côté serveur.`,
  });

  // ═══ APERÇU DES 5 ÉTAPES DU CYCLE ═══
  scenes.push({
    id: 'overview',
    section: 'Plan',
    title: '🗺️ Cycle d\'apprentissage complet — les 5 étapes',
    color: 'var(--accent-blue)',
    text: `Avant de plonger dans les détails, voici la <b>vue d'ensemble</b>
           de ce qu'un RNN fait à chaque itération d'entraînement.<br/>
           Tu verras chacune de ces étapes en action dans la suite :`,
    showCycleOverview: true,  // ← affiche le bel encart 5 étapes
    note: `Ce cycle se répète des milliers de fois pendant l'entraînement, jusqu'à ce que le modèle apprenne à bien prédire la RUL.`,
  });

  // ═══ TENSEUR 3D ÉCLATÉ EN TRANCHES ═══
  scenes.push({
    id: 'tensor-explode',
    section: 'Données d\'entrée',
    title: '📦 Le tenseur 3D X est éclaté en tranches 2D',
    color: 'var(--accent-purple)',
    text: `Le RNN <b>ne lit pas tout d'un coup</b>. Le tenseur 3D
           <InlineMath math="X \\in \\mathbb{R}^{${batch_size} \\times ${seq_length} \\times ${n_features}}" /> 
           est découpé en <b>${seq_length} tranches 2D</b> qui vont être traitées 
           <b>l'une après l'autre dans le temps</b>.`,
    showInputFlow: true,
    activeT: -1,
    formula: String.raw`X \in \mathbb{R}^{${batch_size} \times ${seq_length} \times ${n_features}} \quad \rightarrow \quad \{X_1, X_2, \ldots, X_{${seq_length}}\}`,
    note: `Chaque tranche \\(X_t\\) est une matrice 2D de taille [${batch_size} séquences × ${n_features} features].
           Les ${batch_size} séquences seront traitées EN PARALLÈLE, mais le temps EN SÉRIE.`,
  });

  // ═══ COUCHE PAR COUCHE, TEMPS PAR TEMPS ═══
  for (let L = 0; L < n_layers; L++) {
    const layerColor = LAYER_COLORS[L];
    const layerW = weights.layers[L];
    const units = layerW.units;
    const inputDim = layerW.input_dim;
    const neuronLabels = Array.from({length:units}, (_,i)=>`h${i+1}`);
    const inputLabels  = L === 0
      ? feat_names.slice(0, inputDim)
      : Array.from({length:inputDim}, (_,i)=>`h${i+1}⁽${L}⁾`);

    // Présentation de la couche
    scenes.push({
      id: `layer-${L}-intro`,
      section: `Couche ${L+1}`,
      title: `📦 Présentation de la couche ${L+1}`,
      color: layerColor,
      text: `Cette couche a <b>${units} neurones</b>. Elle reçoit en entrée
             ${L===0
               ? `les <b>${inputDim} features</b> brutes (volt, rotate, ...)`
               : `les <b>${inputDim} sorties</b> de la couche ${L}`}.<br/>
             Voici ses <b>3 matrices de poids</b> initialisées aléatoirement (Xavier) :`,
      matrices: [
        {
          title: `W_xh⁽${L+1}⁾`,
          subtitle: 'entrée → caché',
          data: layerW.W_xh,
          rowLabels: inputLabels,
          colLabels: neuronLabels,
          color: layerColor,
        },
        {
          title: `W_hh⁽${L+1}⁾`,
          subtitle: 'récurrence',
          data: layerW.W_hh,
          rowLabels: neuronLabels,
          colLabels: neuronLabels,
          color: layerColor,
        },
        {
          title: `b⁽${L+1}⁾`,
          subtitle: 'biais',
          data: [layerW.b],
          colLabels: neuronLabels,
          color: layerColor,
          small: true,
        },
      ],
      note: `Ces ${inputDim*units + units*units + units} paramètres seront partagés à TOUS les instants t et pour TOUTES les séquences du batch.`,
    });

    // Une scène par timestep
    for (let t = 0; t < seq_length; t++) {
      const step = forward_steps[L][t];
      const tt = t + 1;
      const tprev = t;

      // Scène : préparation des inputs
      scenes.push({
        id: `layer-${L}-t${t}-prep`,
        section: `Couche ${L+1} · t=${tt}`,
        title: `⏰ Préparation des entrées au temps t=${tt}`,
        color: layerColor,
        text: `À l'instant <b>t=${tt}</b>, la couche ${L+1} reçoit :<br/>
               • Une matrice d'entrée <b>X_${tt}⁽${L+1}⁾</b> (${batch_size} séquences × ${inputDim})<br/>
               • L'état caché précédent <b>H_${tprev}⁽${L+1}⁾</b>${t===0?' = <b>0</b> (vecteur nul à t=1)':''}`,
        // Pour la couche 1, on affiche aussi l'InputFlow avec la tranche active
        showInputFlow: L === 0,
        activeT: t,
        matrices: [
          {
            title: `X_${tt}⁽${L+1}⁾`,
            subtitle: L===0 ? 'features brutes du batch' : `sortie de couche ${L} à t=${tt}`,
            data: step.input_t,
            rowLabels: seqLabels,
            colLabels: inputLabels,
            color: layerColor,
            highlight: true,
          },
          ...(t > 0 ? [{
            title: `H_${tprev}⁽${L+1}⁾`,
            subtitle: 'mémoire de l\'instant précédent',
            data: step.h_prev,
            rowLabels: seqLabels,
            colLabels: neuronLabels,
            color: 'var(--accent-purple)',
          }] : []),
        ],
        note: L === 0
          ? `Cette tranche X_${tt} sort de la file d'attente des ${seq_length} tranches du tenseur 3D.`
          : `Cette matrice est en réalité H_${tt}⁽${L}⁾ = la sortie de la couche précédente.`,
      });

      // Scène : MULTIPLICATION VISUELLE X_t · W_xh
      scenes.push({
        id: `layer-${L}-t${t}-matmul-xh`,
        section: `Couche ${L+1} · t=${tt}`,
        title: `🎯 Multiplication 1 : X_${tt} · W_xh⁽${L+1}⁾`,
        color: 'var(--accent-blue)',
        text: `Première contribution : on multiplie la matrice d'entrée
               <b>X_${tt}</b> par les poids <b>W_xh⁽${L+1}⁾</b>.<br/>
               <b>Comment ça marche ?</b> Pour chaque cellule du résultat, on prend une <b>ligne de X</b>
               et une <b>colonne de W</b>, on multiplie élément par élément, puis on somme.`,
        formula: String.raw`(X_{${tt}} \cdot W_{xh})_{[i,j]} = \sum_{k=1}^{${inputDim}} X_{${tt}[i,k]} \cdot W_{xh[k,j]}`,
        matmul: {
          matA: step.input_t,
          matB: layerW.W_xh,
          result: step.part_xh,
          labelA: `X_${tt}`,
          labelB: `W_xh⁽${L+1}⁾`,
          labelC: `X·W_xh`,
          rowLabelsA: seqLabels,
          colLabelsA: inputLabels,
          rowLabelsB: inputLabels,
          colLabelsB: neuronLabels,
          colorA: layerColor,
          colorB: 'var(--accent-blue)',
          colorC: 'var(--success)',
        },
        note: `🔍 Regarde le calcul détaillé en bas : la cellule en surbrillance ${labelOrFirst(seqLabels)}, ${neuronLabels[0]} montre la somme des produits.`,
      });

      // Scène : calcul Z_t (somme des 3 contributions)
      scenes.push({
        id: `layer-${L}-t${t}-compute`,
        section: `Couche ${L+1} · t=${tt}`,
        title: `🧮 Calcul de la pré-activation Z_${tt}⁽${L+1}⁾`,
        color: layerColor,
        text: `On <b>additionne les 3 contributions</b> élément par élément
               (les 3 matrices ont la même taille ${batch_size}×${units}) :`,
        formula: String.raw`Z_{${tt}}^{(${L+1})} = \underbrace{X_{${tt}} \cdot W_{xh}^{(${L+1})}}_{\text{entrées}} + \underbrace{H_{${tprev}} \cdot W_{hh}^{(${L+1})}}_{\text{passé}} + \underbrace{b^{(${L+1})}}_{\text{biais}}`,
        matrices: [
          {
            title: `X · W_xh`,
            subtitle: 'apport des entrées',
            data: step.part_xh,
            rowLabels: seqLabels,
            colLabels: neuronLabels,
            color: 'var(--accent-blue)',
            small: true,
          },
          {
            title: `H_${tprev} · W_hh`,
            subtitle: 'apport du passé',
            data: step.part_hh,
            rowLabels: seqLabels,
            colLabels: neuronLabels,
            color: 'var(--accent-purple)',
            small: true,
          },
          {
            title: 'b',
            subtitle: 'biais broadcast',
            data: step.part_b,
            rowLabels: seqLabels,
            colLabels: neuronLabels,
            color: 'var(--accent-orange)',
            small: true,
          },
        ],
        note: t === 0
          ? `📌 À t=1, H_0 = 0 donc la 2ème contribution est nulle. Seules X·W_xh et b contribuent.`
          : `📌 La 2ème contribution (H_${tprev}·W_hh) injecte la "mémoire" du temps précédent. C'est ce qui rend le RNN "récurrent" !`,
      });

      // Scène : Z_t calculé
      scenes.push({
        id: `layer-${L}-t${t}-Z`,
        section: `Couche ${L+1} · t=${tt}`,
        title: `📊 Résultat : Z_${tt}⁽${L+1}⁾ (avant activation)`,
        color: layerColor,
        text: `La somme des 3 contributions donne <b>Z_${tt}⁽${L+1}⁾</b>.
               C'est la <b>pré-activation</b> — il manque encore le <code>tanh</code>.`,
        matrices: [{
          title: `Z_${tt}⁽${L+1}⁾`,
          subtitle: 'pré-activation',
          data: step.Z_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: 'var(--accent-orange)',
          highlight: true,
        }],
      });

      // Scène : application du tanh → H_t
      scenes.push({
        id: `layer-${L}-t${t}-H`,
        section: `Couche ${L+1} · t=${tt}`,
        title: `✨ Activation : H_${tt}⁽${L+1}⁾ = tanh(Z_${tt}⁽${L+1}⁾)`,
        color: 'var(--success)',
        text: `On applique la fonction <b>tanh</b> pour écraser les valeurs entre -1 et 1.
               C'est le nouvel <b>état caché</b> de la couche ${L+1} à t=${tt}.`,
        formula: String.raw`H_{${tt}}^{(${L+1})} = \tanh(Z_{${tt}}^{(${L+1})})`,
        matrices: [{
          title: `H_${tt}⁽${L+1}⁾`,
          subtitle: t === seq_length - 1 && L === n_layers - 1
            ? '🎯 état final ! servira pour la prédiction'
            : 'nouvel état caché',
          data: step.H_t,
          rowLabels: seqLabels,
          colLabels: neuronLabels,
          color: 'var(--success)',
          highlight: true,
        }],
        note: t < seq_length - 1
          ? `H_${tt} sera utilisé comme H_${tprev} à l'étape suivante (récurrence ⟲).`
          : L < n_layers - 1
            ? `H_${tt} sera passé comme entrée à la couche ${L+2}.`
            : `H_${tt} est le dernier état caché : on va l'utiliser pour prédire la RUL.`,
      });
    }
  }

  // ═══ PRÉDICTION ═══
  // Scène 1 : Présentation du concept
  scenes.push({
    id: 'predict-intro',
    section: 'Prédiction',
    title: '🎯 Comment obtient-on la prédiction Ŷ ?',
    color: '#f06292',
    text: `Le RNN a fini de traiter les ${seq_length} pas de temps. Le <b>dernier état caché</b>
           <InlineMath math={\`H_{${seq_length}}^{(${n_layers})}\`}/> contient maintenant le
           <b>résumé compressé</b> de toute l'information temporelle.<br/><br/>
           Pour transformer ce résumé en une <b>RUL prédite (en heures)</b>, on applique une
           <b>couche dense de sortie</b> avec une matrice de poids <b>W_out</b> et un biais <b>b_out</b> :`,
    formula: String.raw`\hat{Y} = H_{${seq_length}}^{(${n_layers})} \cdot W_{out} + b_{out}`,
    matrices: [
      {
        title: 'H_final',
        subtitle: `dernier état caché [${batch_size}×${layers[n_layers-1]}]`,
        data: prediction.H_last,
        rowLabels: seqLabels,
        colLabels: Array.from({length:layers[n_layers-1]},(_,i)=>`h${i+1}`),
        color: 'var(--success)',
      },
      {
        title: 'W_out',
        subtitle: `poids de sortie [${layers[n_layers-1]}×1]`,
        data: weights.W_out,
        rowLabels: Array.from({length:layers[n_layers-1]},(_,i)=>`h${i+1}`),
        colLabels: ['ŷ'],
        color: '#f06292',
      },
    ],
    note: `Chaque ligne de H_final (= un état caché par séquence) sera multipliée par la colonne W_out pour donner UN scalaire (la RUL prédite).`,
  });

  // Scène 2 : Multiplication animée H_final · W_out
  scenes.push({
    id: 'predict-matmul',
    section: 'Prédiction',
    title: '🎬 Multiplication animée : H_final · W_out',
    color: '#f06292',
    text: `On applique la <b>multiplication matricielle ligne × colonne</b> entre
           <b>H_final</b> et <b>W_out</b>.<br/>
           Pour chaque séquence (ligne de H), on combine ses
           ${layers[n_layers-1]} valeurs cachées avec les ${layers[n_layers-1]} poids de W_out.<br/>
           <b>Résultat : un seul nombre par séquence</b> (la prédiction).`,
    formula: String.raw`\hat{y}_i = \sum_{k=1}^{${layers[n_layers-1]}} H_{final}[i,k] \cdot W_{out}[k,1]`,
    matmul: {
      matA: prediction.H_last,
      matB: weights.W_out,
      result: prediction.Y_pred_norm.map(v => [v]),
      labelA: 'H_final',
      labelB: 'W_out',
      labelC: 'Ŷ',
      rowLabelsA: seqLabels,
      colLabelsA: Array.from({length:layers[n_layers-1]},(_,i)=>`h${i+1}`),
      rowLabelsB: Array.from({length:layers[n_layers-1]},(_,i)=>`h${i+1}`),
      colLabelsB: ['ŷ'],
      colorA: 'var(--success)',
      colorB: '#f06292',
      colorC: '#f06292',
    },
    note: `🔍 Regarde le calcul détaillé en bas : pour chaque séquence, on somme les produits "valeur cachée × poids" et on ajoute le biais.`,
  });

  // Scène 3 : Résultat final dénormalisé
  scenes.push({
    id: 'predict-result',
    section: 'Prédiction',
    title: '✨ Résultat final : Ŷ en heures réelles',
    color: '#f06292',
    text: `Les valeurs sortent du modèle <b>normalisées</b> (entre 0 et 1).
           On les <b>dénormalise</b> en utilisant le scaler appris pendant l'entraînement
           pour les ramener en heures réelles.`,
    formula: String.raw`\hat{Y}_{\text{heures}} = \text{scaler}^{-1}(\hat{Y}_{\text{norm}})`,
    matrices: [
      {
        title: 'Ŷ (normalisé)',
        subtitle: 'sortie brute du modèle',
        data: [prediction.Y_pred_norm],
        rowLabels: ['ŷ'],
        colLabels: seqLabels,
        color: '#f06292',
        small: true,
      },
      {
        title: 'Ŷ (heures)',
        subtitle: 'après dénormalisation',
        data: [prediction.Y_pred_hours],
        rowLabels: ['RUL'],
        colLabels: seqLabels,
        color: 'var(--accent-orange)',
        small: true,
        highlight: true,
      },
    ],
    note: `🎯 Le modèle prédit que ces 4 séquences ont une RUL de : ${prediction.Y_pred_hours.map((v,i) => `Séq ${seqLabels[i]} = ${v.toFixed(0)}h`).join(' · ')}`,
  });

  // ═══ LOSS ═══
  const errors = loss_info.per_seq_error_norm;
  scenes.push({
    id: 'loss',
    section: 'Loss',
    title: '📉 Calcul de la fonction de coût (MSE)',
    color: 'var(--accent-orange)',
    text: `On compare nos prédictions <b>Ŷ</b> avec les vraies valeurs <b>Y</b> du dataset.
           L'erreur est mise au carré pour pénaliser les grandes erreurs, puis moyennée :`,
    formula: String.raw`\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} (\hat{y}_i - y_i)^2 = ${loss_info.loss.toFixed(4)}`,
    matrices: [
      {
        title: 'Y (vrai)',
        subtitle: 'normalisé',
        data: [y_true_norm],
        rowLabels: ['y'],
        colLabels: seqLabels,
        color: 'var(--success)',
        small: true,
      },
      {
        title: 'Ŷ (prédit)',
        subtitle: 'normalisé',
        data: [prediction.Y_pred_norm],
        rowLabels: ['ŷ'],
        colLabels: seqLabels,
        color: '#f06292',
        small: true,
      },
      {
        title: 'Erreurs',
        subtitle: '(ŷ - y)',
        data: [errors],
        rowLabels: ['err'],
        colLabels: seqLabels,
        color: 'var(--accent-orange)',
        small: true,
        highlight: true,
      },
    ],
    note: `En heures réelles : Y=[${y_true_hours.map(v=>v.toFixed(0)+'h').join(', ')}] vs Ŷ=[${prediction.Y_pred_hours.map(v=>v.toFixed(0)+'h').join(', ')}]`,
  });

  // ═══ BACKPROP — Démonstration mathématique : d'où vient le gradient ? ═══
  scenes.push({
    id: 'backprop-math-1',
    section: 'Backprop · Démonstration',
    title: '📐 D\'où vient la formule du gradient ? (1/3)',
    color: '#f06292',
    text: `Avant de calculer les gradients, on doit comprendre <b>d'où vient la formule</b>
           <InlineMath math="\\frac{\\partial \\mathcal{L}}{\\partial \\hat{Y}} = \\frac{2}{B}(\\hat{Y} - Y)" />.
           <br/><br/>
           <b>Étape 1 — Rappel : la fonction de perte MSE</b><br/>
           Pour un batch de taille <InlineMath math={\`B = ${batch_size}\`}/>, la perte MSE est :`,
    formula: String.raw`\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} (\hat{y}_i - y_i)^2`,
    note: `On somme les erreurs au carré de chaque séquence du batch, puis on divise par B pour avoir une moyenne.`,
  });

  scenes.push({
    id: 'backprop-math-2',
    section: 'Backprop · Démonstration',
    title: '📐 Démonstration : dérivation par rapport à ŷ_i (2/3)',
    color: '#f06292',
    text: `<b>Étape 2 — On dérive par rapport à un seul élément ŷ_i</b><br/>
           On veut savoir : "quand on change un peu <InlineMath math="\\hat{y}_i"/>, comment varie la perte ?"<br/>
           Seul le terme <InlineMath math="i"/> dans la somme dépend de <InlineMath math="\\hat{y}_i"/>, donc :`,
    formula: String.raw`\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left[ \frac{1}{B} (\hat{y}_i - y_i)^2 \right]`,
    note: `💡 Dérivée du carré : si f(u) = u², alors f'(u) = 2u. Ici u = (ŷ_i - y_i), et la dérivée de u par rapport à ŷ_i vaut 1.`,
  });

  scenes.push({
    id: 'backprop-math-3',
    section: 'Backprop · Démonstration',
    title: '📐 Application de la règle de dérivation (3/3)',
    color: '#f06292',
    text: `<b>Étape 3 — Application de la règle de la chaîne</b><br/>
           En appliquant la règle de dérivation du carré (chain rule) :`,
    formula: String.raw`\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{B} \cdot 2 \cdot (\hat{y}_i - y_i) \cdot 1 = \frac{2}{B}(\hat{y}_i - y_i)`,
    formulaWithValues: String.raw`\boxed{\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2}{B}(\hat{Y} - Y)}`,
    note: `🎯 C'est la formule qu'on utilise ! Elle nous dit dans quelle direction ŷ doit bouger pour réduire la perte.
           Si Ŷ > Y → gradient positif → il faut diminuer Ŷ.
           Si Ŷ < Y → gradient négatif → il faut augmenter Ŷ.`,
  });

  // ═══ BACKPROP — Application numérique ═══
  scenes.push({
    id: 'backprop-output',
    section: 'Backprop',
    title: '🔄 Application numérique du gradient à la sortie',
    color: '#f06292',
    text: `Maintenant on applique la formule avec nos vraies valeurs.<br/>
           Avec <InlineMath math={\`B = ${batch_size}\`}/>, l'erreur (Ŷ - Y) et le gradient deviennent :`,
    formula: String.raw`\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2}{${batch_size}}(\hat{Y} - Y) = ${(2/batch_size).toFixed(2)} \times (\hat{Y} - Y)`,
    matrices: [
      {
        title: 'Ŷ - Y',
        subtitle: 'erreurs brutes',
        data: [errors],
        rowLabels: ['err'],
        colLabels: seqLabels,
        color: 'var(--accent-orange)',
        small: true,
      },
      {
        title: 'dY = ∂L/∂Ŷ',
        subtitle: 'gradient initial',
        data: backward.dY,
        rowLabels: seqLabels,
        color: '#f06292',
        small: true,
        highlight: true,
      },
      {
        title: 'dW_out',
        subtitle: 'gradient sur W_out',
        data: backward.dW_out,
        color: '#f06292',
        small: true,
      },
    ],
    note: `Ce gradient dY est le <b>point de départ</b> de la rétropropagation. Il va remonter à travers toutes les couches via la chain rule.`,
  });

  // ═══ Démonstration BPTT pour les couches RNN ═══
  scenes.push({
    id: 'backprop-bptt-math',
    section: 'Backprop · BPTT',
    title: '⏪ Pourquoi "BPTT" (Backpropagation Through Time) ?',
    color: 'var(--accent-purple)',
    text: `Dans un RNN, chaque sortie dépend de <b>tous les pas de temps précédents</b> à cause de
           la récurrence <InlineMath math="W_{hh}"/>. Donc le gradient doit aussi remonter dans le temps !<br/><br/>
           Pour la matrice <InlineMath math="W_{xh}"/>, on doit sommer les contributions à <b>chaque pas de temps t</b> :`,
    formula: String.raw`\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} X_t^T \cdot \frac{\partial \mathcal{L}}{\partial Z_t}`,
    note: `💡 La somme <InlineMath math="\\sum_t"/> vient du fait que le même W_xh est utilisé à T moments différents.
           Chaque pas de temps contribue sa "part" au gradient final.`,
  });

  scenes.push({
    id: 'backprop-tanh-math',
    section: 'Backprop · BPTT',
    title: '🔬 Le rôle du tanh dans la rétropropagation',
    color: 'var(--success)',
    text: `Quand le gradient remonte à travers <InlineMath math="\\tanh"/>, il est multiplié par sa dérivée :`,
    formula: String.raw`\frac{\partial \tanh(z)}{\partial z} = 1 - \tanh^2(z) = 1 - h^2`,
    formulaWithValues: String.raw`\frac{\partial \mathcal{L}}{\partial Z_t} = \frac{\partial \mathcal{L}}{\partial H_t} \cdot (1 - H_t^2)`,
    note: `⚠️ Quand <InlineMath math="|H_t|"/> est proche de 1 (saturation), <InlineMath math="(1 - H_t^2)"/> tend vers 0.
           C'est ce qu'on appelle le <b>vanishing gradient</b> — un gros problème des RNN classiques que LSTM/GRU ont résolu !`,
  });

  // ═══ BACKPROP — Couche par couche ═══
  for (let L = n_layers - 1; L >= 0; L--) {
    const g = backward.layer_grads[L];
    scenes.push({
      id: `backprop-layer-${L}`,
      section: 'Backprop',
      title: `🔄 Gradients pour la couche ${L+1}`,
      color: LAYER_COLORS[L],
      text: `Après application de la <b>chain rule</b> et de la BPTT
             (rétropropagation à travers le temps), on obtient les gradients
             pour les 3 matrices de poids de la couche ${L+1} :`,
      formula: String.raw`\frac{\partial \mathcal{L}}{\partial W_{xh}^{(${L+1})}} = \sum_{t=1}^{T} X_t^T \cdot \frac{\partial \mathcal{L}}{\partial Z_t^{(${L+1})}}`,
      matrices: [
        {
          title: `dW_xh⁽${L+1}⁾`,
          subtitle: `||·||=${g.norm_dW_xh.toFixed(3)}`,
          data: g.dW_xh,
          color: LAYER_COLORS[L],
          small: true,
          highlight: true,
        },
        {
          title: `dW_hh⁽${L+1}⁾`,
          subtitle: `||·||=${g.norm_dW_hh.toFixed(3)}`,
          data: g.dW_hh,
          color: LAYER_COLORS[L],
          small: true,
        },
        {
          title: `db⁽${L+1}⁾`,
          subtitle: `||·||=${g.norm_db.toFixed(3)}`,
          data: [g.db],
          color: LAYER_COLORS[L],
          small: true,
        },
      ],
      note: L > 0
        ? `Ces gradients vont aussi servir à calculer ceux de la couche ${L} en remontant.`
        : `On a maintenant les gradients pour TOUTES les couches.`,
    });
  }

  // ═══ MISE À JOUR ═══
  scenes.push({
    id: 'sgd',
    section: 'Mise à jour',
    title: '⚙️ Mise à jour des poids (SGD)',
    color: 'var(--accent-purple)',
    text: `Tous les poids sont mis à jour dans la direction opposée au gradient,
           proportionnellement au <b>learning rate</b> η = ${data.config.learning_rate}.<br/>
           Par exemple pour W_xh⁽1⁾ :`,
    formula: String.raw`W_{xh, new}^{(1)} = W_{xh, old}^{(1)} - \eta \cdot \nabla W_{xh}^{(1)}`,
    matrices: [
      {
        title: 'W_xh⁽1⁾ (avant)',
        data: weights.layers[0].W_xh,
        color: 'var(--text-tertiary)',
        small: true,
      },
      {
        title: 'W_xh⁽1⁾ (après)',
        data: data.updated.layers[0].W_xh_new,
        color: LAYER_COLORS[0],
        small: true,
        highlight: true,
      },
    ],
    note: `À chaque batch d'entraînement, ce cycle Forward → Loss → Backward → Update se répète.`,
  });

  // ═══ FIN ═══
  scenes.push({
    id: 'end',
    section: 'Conclusion',
    title: '🎉 Fin de la démonstration',
    color: 'var(--success)',
    text: `Tu viens de voir <b>une seule itération</b> d'entraînement pas à pas.<br/>
           Pendant un vrai entraînement, ce processus se répète des milliers de fois
           sur tous les batchs et toutes les époques.`,
    note: `Tu peux maintenant changer la configuration à gauche pour voir l'effet de différentes architectures !`,
  });

  return scenes;
}

// ═══════════════════════════════════════════════════════════════
// COMPOSANT PRINCIPAL
// ═══════════════════════════════════════════════════════════════
export default function RNNDemoPanel({ onSwitchTo }) {
  const [arch, setArch] = useState('rnn');

  // Quand l'utilisateur clique LSTM, déclenche le switch vers le LSTM panel
  const handleArchChange = (newArch) => {
    if (newArch !== 'rnn' && onSwitchTo) {
      onSwitchTo(newArch);
    } else {
      setArch(newArch);
    }
  };

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

  // Animation
  const [sceneIndex, setSceneIndex] = useState(0);
  const [autoPlay, setAutoPlay]     = useState(false);
  const [speed, setSpeed]           = useState(3500);

  const scenes = useMemo(() => buildScenario(data), [data]);
  const currentScene = scenes[sceneIndex];

  // États pour la boîte modèle (synchro avec scène)
  const { activeLayer, activeT, mode } = useMemo(() => {
    if (!currentScene) return { activeLayer: -1, activeT: 0, mode: 'idle' };
    const id = currentScene.id;
    const m = id.match(/layer-(\d+)-t(\d+)/);
    if (m) return { activeLayer: parseInt(m[1]), activeT: parseInt(m[2]), mode: 'forward' };
    if (id === 'predict') return { activeLayer: -1, activeT: 0, mode: 'predict' };
    if (id.startsWith('backprop')) return { activeLayer: -1, activeT: 0, mode: 'backprop' };
    return { activeLayer: -1, activeT: 0, mode: 'idle' };
  }, [currentScene]);

  const runDemo = async () => {
    setLoading(true); setError(null); setSceneIndex(0); setAutoPlay(false);
    try {
      const res = await fetch(`${API}/api/rnn_demo/run`, {
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background:'linear-gradient(135deg,#2a1a4a,#1a0d3a)', border:'1px solid #ce93d840' }}>
            <Brain size={18} style={{ color:'var(--accent-purple)' }} />
          </div>
          <div>
            <h2 className="text-lg font-semibold" style={{ color:'var(--text-primary)' }}>
              Démo Architecture — Mini-dataset synthétique
            </h2>
            <p className="text-xs" style={{ color:'var(--text-tertiary)' }}>
              Animation scénarisée · Narration progressive · Forward + Loss + Backprop
            </p>
          </div>
        </div>
        {data && (
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="px-2 py-1 rounded flex items-center gap-1"
              style={{ background:'var(--bg-card)', color:'var(--accent-orange)' }}>
              <Factory size={11}/> Démo · synthétique
            </span>
            <span className="px-2 py-1 rounded uppercase"
              style={{ background:'var(--bg-card)', color:'var(--accent-blue)' }}>
              {arch}
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
            arch={arch} setArch={handleArchChange} onRun={runDemo} loading={loading}/>
        </div>

        <div className="col-span-9 space-y-4">
          {!data && !loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <Brain size={32} style={{ color:'var(--text-muted)' }} className="mx-auto mb-3"/>
              <p className="text-sm" style={{ color:'var(--text-tertiary)' }}>
                Configure à gauche puis clique <b>Lancer la démo</b>
              </p>
            </div>
          )}

          {loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <p className="text-sm animate-pulse" style={{ color:'var(--accent-purple)' }}>
                ⚙️ Calcul en cours...
              </p>
            </div>
          )}

          {data && (
            <>
              {/* Boîte modèle */}
              <ModelBox config={data.config} weights={data.weights}
                slices={data.slices_2d}
                forward_steps={data.forward_steps}
                prediction={data.prediction}
                backward={data.backward}
                activeLayer={activeLayer} activeT={activeT} mode={mode}
                sceneId={currentScene?.id || ''}/>

              {/* Contrôles navigation */}
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
                    style={{ background:'var(--bg-elevated)', border:'1px solid #4fc3f740', color:'var(--accent-blue)' }}>
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
                          background: speed === s ? '#1a3a5c' : 'var(--bg-elevated)',
                          borderColor: speed === s ? 'var(--accent-blue)' : 'var(--border-default)',
                          color: speed === s ? 'var(--accent-blue)' : 'var(--text-muted)',
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
                      background: 'linear-gradient(90deg,var(--accent-blue),var(--accent-purple),var(--accent-orange),#f06292)',
                    }}/>
                </div>
              </div>

              {/* Narration */}
              <NarrationPanel data={data} scene={currentScene} sceneIndex={sceneIndex}/>
            </>
          )}
        </div>
      </div>
    </div>
  );
}