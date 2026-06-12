/**
 * TransformerDemoPanel.jsx — Démonstration pédagogique TRANSFORMER
 * Encoder + Decoder + Multi-Head Attention + FFN + LayerNorm + Résiduelle
 * Mini-dataset synthétique pédagogique (voir backend/demos/synthetic_data.py)
 */
import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import {
  Box, Play, Pause, RotateCcw, Zap, AlertTriangle, Cpu,
  BookOpen, Factory, Settings, Sparkles, ArrowRight,
  Target, Brain, Lock, ChevronRight, Eye, Layers,
  Network, Compass, Shuffle, GitMerge
} from 'lucide-react';

const API = 'http://localhost:8000';

// ─── Couleurs Transformer ─────────────────────────────────────
const C = {
  embed:    'var(--accent-purple)',   // violet : embedding
  pos:      '#ff8a65',   // orange : positional encoding
  query:    'var(--accent-blue)',   // bleu : Q
  key:      'var(--accent-green)',   // vert : K
  value:    'var(--accent-orange)',   // jaune : V
  attn:     '#f06292',   // rose : attention
  ffn:      '#9c27b0',   // violet foncé : feed-forward
  norm:     '#26a69a',   // turquoise : layer norm
  encoder:  'var(--accent-blue)',   // bleu : encoder
  decoder:  'var(--accent-green)',   // vert : decoder
  output:   '#f06292',   // rose : sortie
};

const HEAD_COLORS = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-orange)', 'var(--accent-purple)'];

// ─── Helpers couleur cellules ─────────────────────────────────
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
  const cellW = small ? 38 : 48;
  const cellH = small ? 20 : 24;

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
                  style={{ height:cellH, color:'var(--text-tertiary)', minWidth:38 }}>{lab}</div>
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
// AttentionHeatmap : visualise une matrice d'attention (T_q × T_k)
// avec gradient de couleur basé sur les poids softmax
// ═══════════════════════════════════════════════════════════════
function AttentionHeatmap({ weights, qLabels, kLabels, title, color = '#f06292' }) {
  if (!weights || !weights.length) return null;
  const cellSize = 36;

  return (
    <div className="inline-block">
      {title && (
        <p className="text-sm font-bold mb-2" style={{ color }}>{title}</p>
      )}
      <div className="relative inline-block">
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
          {qLabels && (
            <div className="flex flex-col mr-1 justify-center">
              {qLabels.map((lab, i) => (
                <div key={i} className="flex items-center justify-end pr-2 text-xs font-mono"
                  style={{ height: cellSize, color: 'var(--text-tertiary)', minWidth: 38 }}>
                  Q={lab}
                </div>
              ))}
            </div>
          )}
          <div>
            {kLabels && (
              <div className="flex">
                {kLabels.map((lab, j) => (
                  <div key={j} className="text-center text-xs font-mono pb-1"
                    style={{ width: cellSize, color: 'var(--text-muted)' }}>
                    K={lab}
                  </div>
                ))}
              </div>
            )}
            {weights.map((row, i) => (
              <div key={i} className="flex">
                {row.map((w, j) => (
                  <div key={j} className="flex items-center justify-center text-xs font-mono border"
                    style={{
                      width: cellSize, height: cellSize,
                      background: `rgba(240, 98, 146, ${0.1 + w * 0.85})`,
                      color: w > 0.5 ? '#fff' : w > 0.25 ? '#f06292' : 'var(--text-tertiary)',
                      borderColor: 'var(--bg-card-alt)', borderWidth: 0.5,
                      fontWeight: w > 0.4 ? 'bold' : 'normal',
                    }}>
                    {(w * 100).toFixed(0)}%
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="text-[10px] mt-1 text-center" style={{ color: 'var(--text-muted)' }}>
        Pourcentage d'attention que chaque Q porte sur chaque K (somme par ligne = 100%)
      </p>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TransformerBox : Boîte modèle SVG avec encoder + decoder
// ═══════════════════════════════════════════════════════════════
function TransformerBox({ config, activeStage = null, mode = 'idle', sceneId = '' }) {
  if (!config) return null;
  const { d_model, n_heads, batch_size, seq_length } = config;
  const W = 1180, H = 580;

  const encX = 280;
  const decX = 720;
  const outX = 1090;

  // Stages dans l'encodeur
  const encStages = [
    { id: 'encoder-input',     y: 80,  label: '① Embed + PE',     color: C.embed },
    { id: 'encoder-attn',      y: 165, label: '② Multi-Head Attn', color: C.attn  },
    { id: 'encoder-norm1',     y: 245, label: '③ Add & Norm',      color: C.norm  },
    { id: 'encoder-ffn',       y: 325, label: '④ Feed-Forward',    color: C.ffn   },
    { id: 'encoder-norm2',     y: 405, label: '⑤ Add & Norm',      color: C.norm  },
  ];

  // Stages dans le décodeur
  const decStages = [
    { id: 'decoder-target',    y: 80,  label: '⑥ Target Token',    color: C.embed },
    { id: 'decoder-masked',    y: 165, label: '⑦ Masked Self-Attn', color: C.attn },
    { id: 'decoder-norm1',     y: 245, label: '⑧ Add & Norm',      color: C.norm  },
    { id: 'decoder-cross',     y: 325, label: '⑨ Cross-Attn (Q←dec, K,V←enc)', color: C.attn },
    { id: 'decoder-norm2',     y: 405, label: '⑩ Add & Norm',      color: C.norm  },
  ];

  const isActive = (id) => activeStage === id || sceneId.includes(id);

  // Bandeau de narration
  const narration = (() => {
    if (mode === 'idle')   return '⏸ Cliquez "Suivant" pour démarrer la démonstration';
    if (mode === 'predict') return '🎯 Le decoder est passé par le head Dense → RUL prédit';
    if (mode === 'backprop') return '🔄 Le gradient remonte à travers le decoder, le cross-attention, puis l\'encoder';
    if (sceneId.includes('embed'))      return '📐 Les features sont projetées dans un espace de dimension d_model + ajout du positional encoding';
    if (sceneId.includes('encoder-attn')) return '🧠 Self-Attention encoder : chaque pas de temps regarde tous les autres pas pour capturer le contexte';
    if (sceneId.includes('encoder-ffn'))  return '🔧 Feed-Forward : transformation non-linéaire (ReLU) indépendante par position';
    if (sceneId.includes('decoder-cross')) return '🔗 Cross-Attention : Q vient du decoder, K et V de l\'encoder — c\'est le lien !';
    if (sceneId.includes('multi-head')) return '🎭 Multi-Head : plusieurs têtes calculent l\'attention en parallèle puis sont concaténées';
    return '🤖 Transformer encoder + decoder · ' + (config.n_heads + ' têtes');
  })();

  return (
    <div className="rounded-xl border-2 overflow-hidden"
      style={{ borderColor: C.encoder, background: 'linear-gradient(135deg, var(--bg-base), var(--tint-info-bg))' }}>
      <div className="flex items-center justify-between px-4 py-2.5"
        style={{ background: 'var(--tint-info-bg)', borderBottom: `1px solid ${C.encoder}40` }}>
        <div className="flex items-center gap-2">
          <Network size={15} style={{ color: C.encoder }} />
          <span className="text-sm font-bold" style={{ color: C.encoder }}>
            🤖 BOÎTE TRANSFORMER — Encoder + Decoder · {n_heads} tête{n_heads>1?'s':''}
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          d_model={d_model} · d_head={d_model/n_heads}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 700 }}>
        <defs>
          <pattern id="trgrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--bg-card)" strokeWidth="0.4"/>
          </pattern>
          <marker id="trarrow" markerWidth="6" markerHeight="6" refX="5" refY="3"
            orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L6,3 z" fill="var(--text-tertiary)"/>
          </marker>
        </defs>
        <rect width={W} height={H} fill="url(#trgrid)" />

        {/* ═══ INPUT (gauche) ═══ */}
        <g>
          <text x={80} y={20} textAnchor="middle" fontSize={11}
            fill="var(--text-tertiary)" fontFamily="monospace" fontWeight="bold">
            📦 ENTRÉE
          </text>
          <text x={80} y={34} textAnchor="middle" fontSize={9}
            fill="var(--text-muted)" fontFamily="monospace">
            X [{batch_size}×{seq_length}×4]
          </text>
          {Array.from({ length: seq_length }).map((_, t) => (
            <motion.g key={t}
              animate={{ scale: isActive('encoder-input') ? [1, 1.05, 1] : 1 }}
              transition={{ duration: 0.6, delay: t * 0.1 }}>
              <rect x={50} y={70 + t * 80} width={60} height={60} rx={5}
                fill="#1a3a5c" stroke="var(--accent-blue)" strokeWidth={1.5}/>
              <text x={80} y={95 + t * 80} textAnchor="middle" fontSize={11}
                fill="var(--accent-blue)" fontFamily="monospace" fontWeight="bold">
                X_{t + 1}
              </text>
              <text x={80} y={110 + t * 80} textAnchor="middle" fontSize={8}
                fill="var(--text-tertiary)" fontFamily="monospace">
                [4×4]
              </text>
            </motion.g>
          ))}
        </g>

        {/* Flèche INPUT → ENCODER */}
        <line x1={115} y1={H/2} x2={encX - 105} y2={H/2}
          stroke="var(--accent-blue)" strokeWidth={1.5}
          markerEnd="url(#trarrow)" opacity={0.6}/>

        {/* ═══ ENCODER ═══ */}
        <g>
          <rect x={encX - 100} y={50} width={200} height={H - 100} rx={12}
            fill={`${C.encoder}10`} stroke={C.encoder} strokeWidth={2}
            strokeDasharray={isActive('encoder') ? 'none' : '5,3'}/>
          <text x={encX} y={42} textAnchor="middle" fontSize={13}
            fill={C.encoder} fontFamily="monospace" fontWeight="bold">
            🔵 ENCODER
          </text>

          {encStages.map(stage => {
            const active = isActive(stage.id);
            return (
              <motion.g key={stage.id}
                animate={{ scale: active ? [1, 1.08, 1] : 1 }}
                transition={{ duration: 0.6 }}>
                <rect x={encX - 90} y={stage.y} width={180} height={50} rx={6}
                  fill={active ? `${stage.color}40` : 'var(--bg-card)'}
                  stroke={stage.color}
                  strokeWidth={active ? 2.5 : 1}/>
                <text x={encX} y={stage.y + 22} textAnchor="middle" fontSize={11}
                  fill={stage.color} fontFamily="monospace" fontWeight="bold">
                  {stage.label}
                </text>
                <text x={encX} y={stage.y + 38} textAnchor="middle" fontSize={9}
                  fill="var(--text-tertiary)" fontFamily="monospace">
                  {stage.id.includes('attn') && 'Q · K^T · V'}
                  {stage.id.includes('ffn') && 'ReLU(xW₁+b₁)W₂+b₂'}
                  {stage.id.includes('norm') && 'LayerNorm + résiduelle'}
                  {stage.id.includes('input') && 'X · W_embed + PE'}
                </text>
              </motion.g>
            );
          })}

          {/* Connexions verticales entre stages encoder */}
          {encStages.slice(0, -1).map((stage, i) => (
            <line key={`enc-conn-${i}`}
              x1={encX} y1={stage.y + 50}
              x2={encX} y2={encStages[i + 1].y}
              stroke={C.encoder} strokeWidth={1} opacity={0.4}/>
          ))}

          {/* Indication "× 1" pour 1 couche */}
          <text x={encX + 110} y={H/2} fontSize={9}
            fill="var(--text-muted)" fontFamily="monospace">
            × 1 couche
          </text>
        </g>

        {/* ═══ FLÈCHE ENCODER → DECODER (cross-attention) ═══ */}
        <motion.g animate={{ opacity: isActive('decoder-cross') ? [0.4, 1, 0.4] : 0.5 }}
          transition={{ duration: 1.5, repeat: isActive('decoder-cross') ? Infinity : 0 }}>
          <line x1={encX + 90} y1={350}
            x2={decX - 90} y2={350}
            stroke={C.attn} strokeWidth={2.5}
            markerEnd="url(#trarrow)"/>
          <text x={(encX + decX) / 2} y={340} textAnchor="middle" fontSize={10}
            fill={C.attn} fontFamily="monospace" fontWeight="bold">
            K, V (encoder output)
          </text>
          <text x={(encX + decX) / 2} y={365} textAnchor="middle" fontSize={9}
            fill="var(--text-tertiary)" fontFamily="monospace">
            cross-attention link
          </text>
        </motion.g>

        {/* ═══ DECODER ═══ */}
        <g>
          <rect x={decX - 100} y={50} width={200} height={H - 100} rx={12}
            fill={`${C.decoder}10`} stroke={C.decoder} strokeWidth={2}
            strokeDasharray={isActive('decoder') ? 'none' : '5,3'}/>
          <text x={decX} y={42} textAnchor="middle" fontSize={13}
            fill={C.decoder} fontFamily="monospace" fontWeight="bold">
            🟢 DECODER
          </text>

          {decStages.map(stage => {
            const active = isActive(stage.id);
            return (
              <motion.g key={stage.id}
                animate={{ scale: active ? [1, 1.08, 1] : 1 }}
                transition={{ duration: 0.6 }}>
                <rect x={decX - 90} y={stage.y} width={180} height={50} rx={6}
                  fill={active ? `${stage.color}40` : 'var(--bg-card)'}
                  stroke={stage.color}
                  strokeWidth={active ? 2.5 : 1}/>
                <text x={decX} y={stage.y + 22} textAnchor="middle" fontSize={11}
                  fill={stage.color} fontFamily="monospace" fontWeight="bold">
                  {stage.label}
                </text>
                <text x={decX} y={stage.y + 38} textAnchor="middle" fontSize={9}
                  fill="var(--text-tertiary)" fontFamily="monospace">
                  {stage.id.includes('target') && 'token cible appris'}
                  {stage.id.includes('masked') && 'auto-attention masquée'}
                  {stage.id.includes('cross') && 'Q←dec, K,V←enc'}
                  {stage.id.includes('norm') && 'LayerNorm + résiduelle'}
                </text>
              </motion.g>
            );
          })}

          {decStages.slice(0, -1).map((stage, i) => (
            <line key={`dec-conn-${i}`}
              x1={decX} y1={stage.y + 50}
              x2={decX} y2={decStages[i + 1].y}
              stroke={C.decoder} strokeWidth={1} opacity={0.4}/>
          ))}
        </g>

        {/* ═══ FLÈCHE DECODER → SORTIE ═══ */}
        <line x1={decX + 90} y1={H/2}
          x2={outX - 30} y2={H/2}
          stroke={mode === 'predict' ? C.output : 'var(--border-strong)'}
          strokeWidth={mode === 'predict' ? 2 : 0.8}
          markerEnd="url(#trarrow)" opacity={0.7}/>
        <text x={(decX + outX) / 2} y={H/2 - 10} textAnchor="middle" fontSize={10}
          fill={C.output} fontFamily="monospace" fontWeight="bold">
          W_out
        </text>

        {/* Sortie ŷ */}
        <motion.g animate={{ scale: mode === 'predict' ? [1, 1.3, 1] : 1 }}>
          <circle cx={outX} cy={H/2} r={26}
            fill={mode === 'predict' ? `${C.output}60` : 'var(--bg-card)'}
            stroke={C.output} strokeWidth={2.5}/>
          {mode === 'predict' && (
            <circle cx={outX} cy={H/2} r={34}
              fill="none" stroke={C.output} strokeWidth={1} opacity={0.4}/>
          )}
          <text x={outX} y={H/2 + 6} textAnchor="middle" fontSize={16}
            fill={C.output} fontFamily="monospace" fontWeight="bold">ŷ</text>
        </motion.g>
        <text x={outX} y={H/2 + 50} textAnchor="middle" fontSize={9}
          fill={C.output} fontFamily="monospace">
          RUL
        </text>

        {/* Bandeau narration en bas */}
        <rect x={0} y={H - 30} width={W} height={30}
          fill="var(--tint-info-bg)" opacity={0.95}/>
        <line x1={0} y1={H - 30} x2={W} y2={H - 30}
          stroke={C.encoder} strokeWidth={2}/>
        <text x={W/2} y={H - 11} textAnchor="middle" fontSize={11}
          fill="var(--text-primary)" fontFamily="monospace">
          {narration}
        </text>
      </svg>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// AnatomySection : Anatomie du Transformer (4 cartes)
// ═══════════════════════════════════════════════════════════════
function TransformerAnatomy() {
  const concepts = [
    {
      name: 'Self-Attention',
      symbol: 'Q · K^T',
      color: C.attn,
      icon: <Brain size={16}/>,
      role: 'Chaque mot regarde les autres',
      formula: String.raw`\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V`,
      explanation: 'Chaque position calcule à quel point elle devrait "écouter" toutes les autres positions. Les scores normalisés (softmax) deviennent des poids appliqués aux valeurs V.',
      example: 'Pour comprendre "elle" dans "Marie sourit, elle est heureuse" → l\'attention pointera vers "Marie".',
    },
    {
      name: 'Multi-Head',
      symbol: 'h₁,h₂,h₃,h₄',
      color: C.query,
      icon: <Shuffle size={16}/>,
      role: 'Plusieurs vues en parallèle',
      formula: String.raw`\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O`,
      explanation: 'Au lieu d\'une seule attention, on en fait plusieurs en parallèle. Chaque "tête" peut apprendre un type différent de relation (syntaxique, sémantique, temporelle...).',
      example: '4 têtes = 4 perspectives différentes sur les mêmes données → richesse capturée.',
    },
    {
      name: 'Positional Enc.',
      symbol: 'PE(pos, i)',
      color: C.pos,
      icon: <Compass size={16}/>,
      role: 'Donne l\'ordre temporel',
      formula: String.raw`\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)`,
      explanation: 'Le Transformer ne traite pas les données séquentiellement (contrairement au RNN). Sans information de position, il ne saurait pas que "je mange" ≠ "mange je". On ajoute donc des sinusoïdes spécifiques à chaque position.',
      example: 'Un sin/cos à plusieurs fréquences : chaque position a sa "signature" unique.',
    },
    {
      name: 'Add & Norm',
      symbol: 'LN(x + sub)',
      color: C.norm,
      icon: <GitMerge size={16}/>,
      role: 'Stabilise + connexions résiduelles',
      formula: String.raw`\text{LN}(x + \text{Sublayer}(x))`,
      explanation: 'Connexion résiduelle (skip connection) : on additionne l\'entrée à la sortie de la sous-couche, ce qui empêche le vanishing gradient. Puis Layer Normalization stabilise les activations.',
      example: 'Permet d\'empiler des dizaines de couches sans que le signal ne disparaisse.',
    },
  ];

  return (
    <div className="rounded-xl p-4 border-2"
      style={{ background: 'linear-gradient(135deg, var(--tint-info-bg), var(--bg-base))', borderColor: C.encoder }}>
      <h4 className="text-sm font-bold flex items-center gap-2 mb-3" style={{ color: C.encoder }}>
        <Network size={14}/> Anatomie d'un Transformer — les 4 mécanismes clés
      </h4>

      <p className="text-xs leading-relaxed mb-4" style={{ color: 'var(--text-secondary)' }}>
        Le <b>Transformer</b> a révolutionné le deep learning en 2017 (article "Attention is all you need").
        Contrairement au RNN qui traite séquentiellement, le Transformer regarde <b>tous les pas de temps en parallèle</b>
        grâce au mécanisme d'<b style={{color: C.attn}}>attention</b>.
      </p>

      {/* Encoder vs Decoder */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-base)', border: `1px solid ${C.encoder}40` }}>
          <p className="text-xs font-bold mb-1" style={{ color: C.encoder }}>
            🔵 ENCODER (compréhension)
          </p>
          <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
            Lit la séquence d'entrée et la transforme en représentation riche.
            Chaque pas de temps "comprend" tous les autres grâce à la self-attention.
          </p>
        </div>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-base)', border: `1px solid ${C.decoder}40` }}>
          <p className="text-xs font-bold mb-1" style={{ color: C.decoder }}>
            🟢 DECODER (génération)
          </p>
          <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
            Utilise un token cible + l'output de l'encoder (cross-attention) pour produire la prédiction.
            Pour ton cas : un seul token cible → 1 valeur de RUL.
          </p>
        </div>
      </div>

      {/* Les 4 concepts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {concepts.map((c, i) => (
          <motion.div key={c.name}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="rounded-lg p-3"
            style={{ background: 'var(--bg-base)', border: `1px solid ${c.color}50` }}>
            <div className="flex items-center gap-2 mb-2">
              <div style={{ color: c.color }}>{c.icon}</div>
              <span className="text-xs font-bold uppercase tracking-wide" style={{ color: c.color }}>
                {c.name}
              </span>
              <span className="text-[10px] font-mono ml-auto" style={{ color: 'var(--text-muted)' }}>
                {c.symbol}
              </span>
            </div>
            <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              {c.role}
            </p>
            <div className="rounded p-1.5 mb-2 text-center" style={{ background: 'var(--bg-card)' }}>
              <span style={{ fontSize: '0.7em' }}>
                <InlineMath math={c.formula}/>
              </span>
            </div>
            <p className="text-[10px] leading-snug mb-1" style={{ color: 'var(--text-tertiary)' }}>
              {c.explanation}
            </p>
            <p className="text-[10px] italic" style={{ color: c.color }}>
              💬 {c.example}
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CycleOverview (réutilisé)
// ═══════════════════════════════════════════════════════════════
function CycleOverview() {
  const steps = [
    { num: 1, title: 'Forward',  icon: '➡️', desc: 'Calculer ŷ',           formula: String.raw`\hat{y} = f(x ; W)`, color: 'var(--accent-blue)', explanation: 'X traverse encoder + decoder + head pour produire ŷ.' },
    { num: 2, title: 'Loss',     icon: '🎯', desc: 'Mesurer l\'erreur',    formula: String.raw`\mathcal{L} = (y - \hat{y})^2`, color: 'var(--accent-orange)', explanation: 'Compare avec la vraie RUL.' },
    { num: 3, title: 'Backward', icon: '⬅️', desc: 'Calculer gradients',   formula: String.raw`\nabla W = \frac{\partial \mathcal{L}}{\partial W}`, color: '#f06292', explanation: 'Le gradient remonte à travers toutes les attention heads.' },
    { num: 4, title: 'Update',   icon: '⚙️', desc: 'Mettre à jour',        formula: String.raw`W \leftarrow W - \eta \nabla W`, color: 'var(--accent-purple)', explanation: 'Ajustement de toutes les matrices Q, K, V, FFN, ...' },
    { num: 5, title: 'Répéter',  icon: '🔁', desc: 'Sur tous les batchs',  formula: String.raw`\text{tous batchs}`, color: 'var(--accent-green)', explanation: 'Cycle répété des milliers de fois.' },
  ];

  return (
    <div className="rounded-xl p-4 border-2"
      style={{ background: 'linear-gradient(135deg, var(--bg-card), var(--bg-base))', borderColor: 'var(--accent-blue)' }}>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-bold flex items-center gap-2" style={{ color: 'var(--accent-blue)' }}>
          <Sparkles size={14}/> Cycle d'apprentissage Transformer
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
// MultiHeadView : visualise les n_heads en parallèle
// ═══════════════════════════════════════════════════════════════
function MultiHeadView({ heads, qLabels, kLabels, color = C.attn }) {
  if (!heads || !heads.length) return null;

  return (
    <div>
      <p className="text-xs font-semibold mb-3" style={{ color }}>
        🎭 Les {heads.length} tête{heads.length>1?'s':''} d'attention en parallèle :
      </p>
      <div className="grid gap-3" style={{
        gridTemplateColumns: heads.length === 1 ? '1fr' : heads.length === 2 ? '1fr 1fr' : '1fr 1fr',
      }}>
        {heads.map((head, h) => {
          // On prend la 1ère séquence du batch pour visualiser
          const attnSample = head.attn_weights[0];
          return (
            <div key={h} className="rounded-lg p-3"
              style={{ background: 'var(--bg-base)', border: `1px solid ${HEAD_COLORS[h]}40` }}>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                  style={{ background: `${HEAD_COLORS[h]}30`, color: HEAD_COLORS[h], border: `1px solid ${HEAD_COLORS[h]}` }}>
                  {h + 1}
                </div>
                <span className="text-xs font-bold" style={{ color: HEAD_COLORS[h] }}>
                  Tête #{h + 1}
                </span>
              </div>
              <AttentionHeatmap
                weights={attnSample}
                qLabels={qLabels}
                kLabels={kLabels}
                color={HEAD_COLORS[h]}
              />
            </div>
          );
        })}
      </div>
      <p className="text-xs italic mt-3 px-3 py-2 rounded" style={{
        background:'var(--bg-card)', color:'var(--text-tertiary)', borderLeft:`2px solid ${color}`
      }}>
        💡 Chaque tête capture un type différent de relation. Par exemple, certaines têtes
        se concentrent sur les positions proches, d'autres sur des positions éloignées.
        À la fin, les sorties des {heads.length} têtes sont concaténées et projetées.
      </p>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Sidebar Transformer
// ═══════════════════════════════════════════════════════════════
function DemoSidebar({ config, setConfig, onRun, loading, onSwitchTo }) {
  const setNHeads = (n) => {
    // S'assurer que d_model reste divisible par n
    let newDm = config.d_model;
    if (newDm % n !== 0) {
      newDm = n * Math.ceil(newDm / n);
    }
    setConfig({ ...config, n_heads: n, d_model: newDm });
  };

  return (
    <div className="rounded-xl border p-4 space-y-4"
      style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
      <div className="flex items-center gap-2 pb-2 border-b" style={{ borderColor:'var(--border-default)' }}>
        <Settings size={14} style={{ color: C.encoder }} />
        <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: C.encoder }}>
          Configuration Transformer
        </p>
      </div>

      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          🧠 ARCHITECTURE
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          <button onClick={() => onSwitchTo && onSwitchTo('rnn')}
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-tertiary)' }}>
            <div>RNN</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>Simple</div>
          </button>
          <button onClick={() => onSwitchTo && onSwitchTo('lstm')}
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-tertiary)' }}>
            <div>LSTM</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>4 portes</div>
          </button>
          <button onClick={() => onSwitchTo && onSwitchTo('gru')}
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:'var(--bg-elevated)', borderColor:'var(--border-default)', color:'var(--text-tertiary)' }}>
            <div>GRU</div>
            <div className="text-[9px]" style={{ color:'var(--text-muted)' }}>2 portes</div>
          </button>
          <button
            className="px-2 py-2 rounded-lg text-xs font-bold border"
            style={{ background:`${C.encoder}20`, borderColor:C.encoder, color:C.encoder }}>
            <div>Transformer ✓</div>
            <div className="text-[9px]" style={{ color: C.encoder }}>Actif</div>
          </button>
        </div>
      </div>

      {/* Nombre de têtes */}
      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          🎭 NOMBRE DE TÊTES D'ATTENTION
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {[1, 2, 4].map(n => (
            <button key={n} onClick={() => setNHeads(n)}
              className="px-2 py-2 rounded-lg text-sm font-bold border"
              style={{
                background: config.n_heads === n ? `${HEAD_COLORS[n-1]}20` : 'var(--bg-elevated)',
                borderColor: config.n_heads === n ? HEAD_COLORS[n-1] : 'var(--border-default)',
                color: config.n_heads === n ? HEAD_COLORS[n-1] : 'var(--text-tertiary)',
              }}>{n}</button>
          ))}
        </div>
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
          Plus de têtes = plus de "perspectives" parallèles
        </p>
      </div>

      {/* d_model */}
      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          DIMENSION MODÈLE (d_model) : {config.d_model}
        </label>
        <div className="grid grid-cols-4 gap-1">
          {[4, 8, 16, 32].filter(d => d % config.n_heads === 0).map(d => (
            <button key={d} onClick={() => setConfig({ ...config, d_model: d })}
              className="px-1 py-1 rounded text-xs font-mono border"
              style={{
                background: config.d_model === d ? 'var(--tint-info-bg)' : 'var(--bg-elevated)',
                borderColor: config.d_model === d ? C.encoder : 'var(--border-default)',
                color: config.d_model === d ? C.encoder : 'var(--text-tertiary)',
              }}>{d}</button>
          ))}
        </div>
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
          d_head = d_model / n_heads = {config.d_model / config.n_heads}
        </p>
      </div>

      {/* d_ff */}
      <div>
        <label className="text-xs font-semibold mb-1.5 block" style={{ color:'var(--text-tertiary)' }}>
          FEED-FORWARD (d_ff) : {config.d_ff}
        </label>
        <div className="grid grid-cols-4 gap-1">
          {[8, 16, 32, 64].map(d => (
            <button key={d} onClick={() => setConfig({ ...config, d_ff: d })}
              className="px-1 py-1 rounded text-xs font-mono border"
              style={{
                background: config.d_ff === d ? '#2a0d4a' : 'var(--bg-elevated)',
                borderColor: config.d_ff === d ? C.ffn : 'var(--border-default)',
                color: config.d_ff === d ? C.ffn : 'var(--text-tertiary)',
              }}>{d}</button>
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
          background: loading ? 'var(--bg-card)' : `linear-gradient(135deg, ${C.encoder}30, ${C.encoder}50)`,
          border: `1px solid ${loading ? 'var(--border-default)' : C.encoder}`,
          color: loading ? 'var(--text-muted)' : C.encoder,
        }}>
        {loading ? '⚙️ Calcul...' : <><Zap size={14}/> Lancer la démo Transformer</>}
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// NarrationPanel
// ═══════════════════════════════════════════════════════════════
function NarrationPanel({ data, scene, sceneIndex }) {
  if (!data || !scene) return null;

  return (
    <div className="rounded-xl border" style={{ background:'var(--bg-base)', borderColor: `${C.encoder}40` }}>
      <div className="px-4 py-2.5 border-b flex items-center justify-between"
        style={{ background:'var(--tint-info-bg)', borderColor: `${C.encoder}40` }}>
        <div className="flex items-center gap-2">
          <BookOpen size={14} style={{ color: C.encoder }} />
          <span className="text-sm font-semibold" style={{ color: C.encoder }}>
            📝 Narration Transformer
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
                style={{ color: scene.color || C.encoder }}>
                {scene.section}
              </p>
              <h3 className="text-base font-bold mt-1" style={{ color:'var(--text-primary)' }}>
                {scene.title}
              </h3>
            </div>

            <p className="text-sm leading-relaxed mb-3" style={{ color:'var(--text-secondary)' }}
              dangerouslySetInnerHTML={{ __html: scene.text }}/>

            {scene.showCycleOverview && <div className="my-4"><CycleOverview/></div>}
            {scene.showAnatomy        && <div className="my-4"><TransformerAnatomy/></div>}
            {scene.showMultiHead && (
              <div className="my-4">
                <MultiHeadView
                  heads={scene.multiHeadData}
                  qLabels={scene.qLabels}
                  kLabels={scene.kLabels}
                  color={scene.color}
                />
              </div>
            )}

            {scene.formula && (
              <div className="rounded-lg p-3 my-3" style={{ background:'var(--bg-card)' }}>
                <BlockMath math={scene.formula}/>
              </div>
            )}

            {scene.attentionMatrix && (
              <div className="my-3 flex justify-center">
                <AttentionHeatmap
                  weights={scene.attentionMatrix}
                  qLabels={scene.qLabels}
                  kLabels={scene.kLabels}
                  title={scene.attentionTitle}
                  color={scene.color}
                />
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
                borderLeft:`2px solid ${scene.color || C.encoder}`
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
// Construction du scénario Transformer
// ═══════════════════════════════════════════════════════════════
function buildTransformerScenario(data) {
  if (!data) return [];
  const { config, weights, embedding, encoder_steps, decoder_steps,
          prediction, loss_info, backward,
          y_true_norm, y_true_hours, feat_names, slices_2d } = data;
  const { d_model, n_heads, d_head, batch_size, seq_length, n_features } = config;

  const scenes = [];
  const seqLabels = Array.from({length: batch_size}, (_, i) => `Séq ${String.fromCharCode(65 + i)}`);
  const tLabels   = Array.from({length: seq_length}, (_, i) => `t${i+1}`);
  const dimLabels = Array.from({length: d_model}, (_, i) => `d${i+1}`);
  const headLabels = Array.from({length: d_head},  (_, i) => `h${i+1}`);

  // ═══ INTRO ═══
  scenes.push({
    id: 'intro',
    section: 'Introduction',
    title: '🤖 Bienvenue dans la démonstration Transformer',
    color: C.encoder,
    text: `Le <b>Transformer</b> a révolutionné le deep learning en 2017.
           Contrairement aux RNN/LSTM qui traitent les séquences <b>séquentiellement</b>,
           le Transformer regarde <b>tous les pas de temps en parallèle</b> grâce à l'<b>attention</b>.<br/><br/>
           Configuration : <b>${n_heads} tête${n_heads>1?'s':''}</b> · <b>d_model=${d_model}</b> · <b>d_head=${d_head}</b><br/>
           Batch <b>${batch_size}×${seq_length}×${n_features}</b> issu d'un mini-dataset synthétique pédagogique.`,
    note: `Article fondateur : "Attention is all you need" (Vaswani et al., 2017). Aujourd'hui à la base de GPT, BERT, etc.`,
  });

  // ═══ CYCLE ═══
  scenes.push({
    id: 'cycle',
    section: 'Plan',
    title: '🗺️ Cycle d\'apprentissage',
    color: 'var(--accent-blue)',
    text: `Comme tout réseau, le Transformer suit le cycle Forward → Loss → Backward → Update.
           La spécificité est <b>ce qui se passe dans le forward</b>.`,
    showCycleOverview: true,
  });

  // ═══ ANATOMIE ═══
  scenes.push({
    id: 'anatomy',
    section: 'Anatomie',
    title: '🧬 Anatomie d\'un Transformer',
    color: C.encoder,
    text: `Comprenons d'abord les <b>4 mécanismes clés</b> qui composent un Transformer.`,
    showAnatomy: true,
  });

  // ═══ EMBEDDING + POSITIONAL ENCODING ═══
  scenes.push({
    id: 'embed-1',
    section: '① Embedding',
    title: '📐 Étape 1 : Embedding des features',
    color: C.embed,
    text: `Les <b>${n_features} features brutes</b> (volt, rotate, etc.) sont projetées
           dans un espace de dimension plus grande <b>d_model = ${d_model}</b>.<br/>
           C'est comme transformer chaque heure en un "vecteur sémantique" plus riche.`,
    formula: String.raw`X_{embed} = X \cdot W_{embed} + b_{embed}, \quad X_{embed} \in \mathbb{R}^{${batch_size} \times ${seq_length} \times ${d_model}}`,
    matrices: [
      {
        title: 'X (séq A)',
        subtitle: 'features brutes',
        data: data.input_tensor_3d[0],
        rowLabels: tLabels,
        colLabels: feat_names,
        color: 'var(--accent-blue)',
        small: true,
      },
      {
        title: 'W_embed',
        subtitle: 'projection',
        data: weights.W_embed,
        rowLabels: feat_names,
        colLabels: dimLabels,
        color: C.embed,
        small: true,
      },
      {
        title: 'X_embed (séq A)',
        subtitle: 'projeté',
        data: embedding.X_embed[0],
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.embed,
        small: true,
        highlight: true,
      },
    ],
  });

  scenes.push({
    id: 'embed-pe',
    section: '① Embedding',
    title: '🌊 Étape 2 : Positional Encoding (sinusoïdes)',
    color: C.pos,
    text: `Le Transformer regarde tous les pas en parallèle — il ne sait pas <b>quel pas vient avant l'autre</b> !
           On ajoute donc des <b>sinusoïdes spécifiques à chaque position</b> qui agissent comme une "horloge".`,
    formula: String.raw`PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos(...)`,
    matrices: [
      {
        title: 'PE',
        subtitle: 'positional encoding',
        data: embedding.PE,
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.pos,
        small: true,
      },
      {
        title: 'X_pos = X_embed + PE',
        subtitle: 'séq A',
        data: embedding.X_pos[0],
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.pos,
        small: true,
        highlight: true,
      },
    ],
    note: `Chaque position a sa "signature" unique de sin/cos à différentes fréquences. Cela permet au modèle d'apprendre des relations relatives entre positions.`,
  });

  // ═══ ENCODER : SELF-ATTENTION (Q, K, V) ═══
  scenes.push({
    id: 'encoder-attn-qkv',
    section: '② Encoder Attention',
    title: '🧠 Étape 3 : Multi-Head Self-Attention — calcul de Q, K, V',
    color: C.attn,
    text: `Chaque pas de temps va se demander : <b>"qui dois-je écouter parmi les autres pas ?"</b><br/>
           On commence par calculer <b>3 versions</b> de l'entrée X_pos :<br/>
           • <b>Q (Query)</b> = "ce que je cherche" — bleu<br/>
           • <b>K (Key)</b> = "ce que je peux offrir" — vert<br/>
           • <b>V (Value)</b> = "ce qu'on récupère" — jaune`,
    formula: String.raw`Q = X_{pos} W^Q, \quad K = X_{pos} W^K, \quad V = X_{pos} W^V`,
    matrices: [
      {
        title: 'Q (séq A)',
        subtitle: 'queries',
        data: encoder_steps.self_attn.Q_full[0],
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.query,
        small: true,
      },
      {
        title: 'K (séq A)',
        subtitle: 'keys',
        data: encoder_steps.self_attn.K_full[0],
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.key,
        small: true,
      },
      {
        title: 'V (séq A)',
        subtitle: 'values',
        data: encoder_steps.self_attn.V_full[0],
        rowLabels: tLabels,
        colLabels: dimLabels,
        color: C.value,
        small: true,
      },
    ],
    note: `Q, K, V sont juste des projections linéaires de X_pos par 3 matrices différentes. Chacune a une fonction sémantique distincte dans le calcul d'attention.`,
  });

  // Si plusieurs têtes : scène multi-head
  if (n_heads > 1) {
    scenes.push({
      id: 'encoder-multi-head',
      section: '② Encoder Attention',
      title: `🎭 Étape 4 : Découpage en ${n_heads} têtes parallèles`,
      color: C.query,
      text: `Au lieu de faire UNE attention sur d_model=${d_model}, on découpe Q, K, V en
             <b>${n_heads} têtes</b> de taille d_head=${d_head}.<br/>
             Chaque tête fait sa propre attention <b>en parallèle</b>.<br/>
             Comme ${n_heads} cerveaux qui regardent les mêmes données mais sous des angles différents.`,
      formula: String.raw`\text{head}_i = \text{Attention}(Q_i, K_i, V_i), \quad i = 1, ..., ${n_heads}`,
      showMultiHead: true,
      multiHeadData: encoder_steps.self_attn.heads,
      qLabels: tLabels,
      kLabels: tLabels,
      note: `Chaque tête peut apprendre un type de relation différente (proximité, contraste, périodicité...).`,
    });
  }

  // Score & Softmax pour la première tête (visualisation)
  scenes.push({
    id: 'encoder-attn-scores',
    section: '② Encoder Attention',
    title: `🔢 Étape ${n_heads > 1 ? '5' : '4'} : Calcul des scores Q · K^T / √d_head`,
    color: C.attn,
    text: `Pour chaque <b>paire (Q_t, K_s)</b>, on calcule le <b>produit scalaire</b>.
           Plus il est élevé, plus le pas t doit "écouter" le pas s.<br/>
           On divise par √d_head pour stabiliser les gradients.`,
    formula: String.raw`\text{scores}_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d_{head}}} = \frac{Q_i \cdot K_j}{\sqrt{${d_head}}}`,
    attentionMatrix: encoder_steps.self_attn.heads[0].scores[0].map(row =>
      row.map(v => Math.tanh(v / 3) * 0.5 + 0.5)  // Normalisation visuelle entre 0-1
    ),
    qLabels: tLabels,
    kLabels: tLabels,
    attentionTitle: 'Scores bruts (tête 1, séq A) — avant softmax',
    note: `Les scores ne sont pas encore des probabilités. La prochaine étape (softmax) les normalise.`,
  });

  scenes.push({
    id: 'encoder-attn-softmax',
    section: '② Encoder Attention',
    title: `✨ Étape ${n_heads > 1 ? '6' : '5'} : Softmax → poids d'attention`,
    color: C.attn,
    text: `On applique <b>softmax</b> sur chaque ligne des scores → on obtient des <b>probabilités</b>
           (positives, somme = 1) qui indiquent à quel point chaque Q "écoute" chaque K.`,
    formula: String.raw`\alpha_{i,j} = \frac{\exp(\text{score}_{i,j})}{\sum_k \exp(\text{score}_{i,k})}`,
    attentionMatrix: encoder_steps.self_attn.heads[0].attn_weights[0],
    qLabels: tLabels,
    kLabels: tLabels,
    attentionTitle: '🎯 Poids d\'attention (tête 1, séq A) — après softmax',
    note: `Chaque ligne somme à 100%. Une cellule à 50% signifie "j'accorde la moitié de mon attention à cette position".`,
  });

  scenes.push({
    id: 'encoder-attn-output',
    section: '② Encoder Attention',
    title: `📤 Étape ${n_heads > 1 ? '7' : '6'} : Application aux V → sortie de l'attention`,
    color: C.attn,
    text: `On multiplie les <b>poids d'attention</b> par <b>V (values)</b>.
           Chaque position récupère une combinaison pondérée des valeurs de toutes les positions.<br/>
           ${n_heads > 1 ? `Puis on <b>concatène les ${n_heads} têtes</b> et on projette par W_o.` : ''}`,
    formula: String.raw`\text{output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V`,
    matrices: [{
      title: 'Sortie attention (séq A)',
      subtitle: 'après concat + W_o',
      data: encoder_steps.self_attn.output[0],
      rowLabels: tLabels,
      colLabels: dimLabels,
      color: C.attn,
      small: true,
      highlight: true,
    }],
  });

  // Add & Norm
  scenes.push({
    id: 'encoder-norm1',
    section: '③ Add & Norm',
    title: '🔗 Étape : Connexion résiduelle + Layer Normalization',
    color: C.norm,
    text: `On <b>additionne l'entrée X_pos</b> à la sortie de l'attention (skip connection),
           puis on normalise. Ça stabilise l'apprentissage et permet d'empiler plein de couches.`,
    formula: String.raw`\text{Norm1} = \text{LayerNorm}(X_{pos} + \text{Attention}(X_{pos}))`,
    matrices: [{
      title: 'Norm1 (séq A)',
      data: encoder_steps.norm1[0],
      rowLabels: tLabels,
      colLabels: dimLabels,
      color: C.norm,
      small: true,
      highlight: true,
    }],
    note: `LayerNorm normalise chaque ligne (chaque pas de temps) à moyenne 0, variance 1.`,
  });

  // FFN
  scenes.push({
    id: 'encoder-ffn',
    section: '④ Feed-Forward',
    title: '🔧 Étape : Feed-Forward Network',
    color: C.ffn,
    text: `Chaque position passe par un MLP à 2 couches avec ReLU. <b>Indépendamment</b> des autres positions.<br/>
           C'est le seul endroit non-linéaire ! L'attention est linéaire.`,
    formula: String.raw`\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2`,
    matrices: [{
      title: 'FFN output (séq A)',
      data: encoder_steps.ffn_out[0],
      rowLabels: tLabels,
      colLabels: dimLabels,
      color: C.ffn,
      small: true,
      highlight: true,
    }],
    note: `Un autre Add & Norm suit pour donner la sortie finale de l'encoder.`,
  });

  // Encoder output
  scenes.push({
    id: 'encoder-output',
    section: '⑤ Sortie Encoder',
    title: '🎯 Sortie finale de l\'encoder',
    color: C.encoder,
    text: `L'encoder a transformé X en <b>une représentation riche et contextualisée</b>
           de la séquence d'entrée. Cette sortie va alimenter le decoder via la cross-attention.`,
    matrices: [{
      title: 'Encoder output (séq A)',
      subtitle: 'représentation finale',
      data: encoder_steps.output[0],
      rowLabels: tLabels,
      colLabels: dimLabels,
      color: C.encoder,
      small: true,
      highlight: true,
    }],
  });

  // ═══ DECODER ═══
  scenes.push({
    id: 'decoder-target',
    section: '⑥ Decoder',
    title: '🎯 Decoder : le token cible',
    color: C.decoder,
    text: `Le decoder commence avec un <b>token cible appris</b> — un vecteur unique qui
           sera "rempli" d'information par les couches d'attention.<br/>
           Pour ton problème de régression, il y a <b>un seul token</b> qui produira la prédiction.`,
    matrices: [{
      title: 'Target token',
      subtitle: '1 vecteur appris',
      data: [weights.decoder.target_token],
      rowLabels: ['target'],
      colLabels: dimLabels,
      color: C.decoder,
      small: true,
    }],
    note: `Dans un Transformer de NLP (traduction), le token serait un <SOS> (Start Of Sequence). Pour la régression, c'est juste un point d'ancrage.`,
  });

  scenes.push({
    id: 'decoder-masked',
    section: '⑦ Masked Self-Attn',
    title: '🎭 Decoder : Masked Self-Attention',
    color: C.attn,
    text: `Le decoder fait sa propre auto-attention sur la séquence cible.
           Le "masque" empêche de regarder le futur (pas de tricherie).<br/>
           Avec un seul token cible, c'est une attention triviale, mais en NLP avec plusieurs tokens
           c'est essentiel.`,
    formula: String.raw`\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) V`,
    note: `Le masque M vaut -∞ pour les positions futures, ce qui annule leur attention après softmax.`,
  });

  scenes.push({
    id: 'decoder-cross',
    section: '⑨ Cross-Attention',
    title: '🔗 ⭐ La pièce maîtresse : Cross-Attention',
    color: C.attn,
    text: `<b>C'est ici que la magie opère !</b><br/>
           Le decoder utilise sa propre représentation comme <b>Q (queries)</b>,
           mais récupère <b>K et V depuis l'encoder</b>.<br/>
           → Le decoder demande à l'encoder : <i>"Qu'est-ce qui est important dans ce que tu as appris ?"</i>`,
    formula: String.raw`\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d}}\right) V_{enc}`,
    attentionMatrix: decoder_steps.cross_attn.heads[0].attn_weights[0],
    qLabels: ['target'],
    kLabels: tLabels,
    attentionTitle: '🔗 Cross-attention (tête 1, séq A) : que regarde le decoder dans l\'encoder ?',
    note: `Cette ligne d'attention montre les ${seq_length} pas de la séquence d'entrée que le token cible "observe" pour produire sa prédiction.`,
  });

  scenes.push({
    id: 'decoder-output',
    section: '⑩ Sortie Decoder',
    title: '🎯 Sortie finale du decoder',
    color: C.decoder,
    text: `Après FFN + Add&Norm finale, on obtient la représentation finale du token cible.`,
    matrices: [{
      title: 'Decoder output (séq A)',
      data: decoder_steps.output[0],
      rowLabels: ['target'],
      colLabels: dimLabels,
      color: C.decoder,
      small: true,
      highlight: true,
    }],
  });

  // ═══ PRÉDICTION ═══
  scenes.push({
    id: 'predict',
    section: 'Prédiction',
    title: '🎯 Calcul de la prédiction Ŷ',
    color: C.output,
    text: `On projette la sortie du decoder par <b>W_out</b> pour obtenir la RUL prédite.`,
    formula: String.raw`\hat{Y} = \text{decoder\_output} \cdot W_{out} + b_{out}`,
    matrices: [{
      title: 'Ŷ (norm)',
      data: [prediction.Y_pred_norm],
      rowLabels: ['ŷ'],
      colLabels: seqLabels,
      color: C.output,
      highlight: true,
    }],
    note: `En heures : ${prediction.Y_pred_hours.map(v => v.toFixed(1)+'h').join(', ')}`,
  });

  // ═══ LOSS ═══
  scenes.push({
    id: 'loss',
    section: 'Loss',
    title: '📉 Calcul de la fonction de coût',
    color: 'var(--accent-orange)',
    text: `On compare nos prédictions avec les vraies valeurs.`,
    formula: String.raw`\mathcal{L} = \frac{1}{B} \sum (\hat{y}_i - y_i)^2 = ${loss_info.loss.toFixed(4)}`,
    matrices: [
      { title: 'Y vrai', data: [y_true_norm], rowLabels: ['y'], colLabels: seqLabels, color: 'var(--success)', small: true },
      { title: 'Ŷ prédit', data: [prediction.Y_pred_norm], rowLabels: ['ŷ'], colLabels: seqLabels, color: C.output, small: true },
      { title: 'Erreurs', data: [loss_info.per_seq_error_norm], rowLabels: ['err'], colLabels: seqLabels, color: 'var(--accent-orange)', small: true, highlight: true },
    ],
  });

  // ═══ BACKPROP ═══
  scenes.push({
    id: 'backprop',
    section: 'Backprop',
    title: '🔄 Rétropropagation à travers le Transformer',
    color: C.output,
    text: `Le gradient remonte à travers <b>toutes les attention heads, FFN, layer norms</b>.<br/>
           C'est ce qui permet d'ajuster simultanément toutes les matrices Q, K, V, FFN du modèle.`,
    formula: String.raw`\nabla W = \frac{\partial \mathcal{L}}{\partial W}`,
    matrices: [
      {
        title: 'Norme des gradients',
        subtitle: 'par bloc',
        data: [[
          backward.norms.embedding,
          backward.norms.enc_attn,
          backward.norms.enc_ffn,
          backward.norms.encoder,
          backward.norms.dec_masked,
          backward.norms.dec_cross,
          backward.norms.dec_ffn,
          backward.norms.decoder,
          backward.norms.dW_out,
        ]],
        rowLabels: ['||∇||'],
        colLabels: ['emb', 'enc_a', 'enc_f', 'enc', 'd_msk', 'd_cr', 'd_ffn', 'dec', 'W_o'],
        color: C.output,
        decimals: 4,
        small: true,
        highlight: true,
      },
    ],
    note: `Les gradients diminuent en remontant — c'est normal. La connexion résiduelle (Add & Norm) atténue ce phénomène par rapport à un RNN classique.`,
  });

  // ═══ FIN ═══
  scenes.push({
    id: 'end',
    section: 'Conclusion',
    title: '🎉 Fin de la démo Transformer',
    color: 'var(--success)',
    text: `Tu as vu un cycle complet d'un Transformer encoder + decoder avec
           multi-head attention.<br/><br/>
           <b>Différences principales avec un RNN/LSTM :</b><br/>
           • RNN/LSTM : séquentiel · état caché unique · vanishing gradient<br/>
           • Transformer : <b>parallèle</b> · attention · positional encoding · résiduelle`,
    note: `Pour PdM, le Transformer est intéressant pour les longues séquences (capture des dépendances lointaines), mais demande plus de données pour bien généraliser.`,
  });

  return scenes;
}

// ═══════════════════════════════════════════════════════════════
// COMPOSANT PRINCIPAL
// ═══════════════════════════════════════════════════════════════
export default function TransformerDemoPanel({ onSwitchTo }) {
  const [config, setConfig] = useState({
    d_model:       8,
    n_heads:       2,
    d_ff:          16,
    batch_size:    4,
    seq_length:    3,
    learning_rate: 0.01,
    seed:          7,
  });
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const [sceneIndex, setSceneIndex] = useState(0);
  const [autoPlay, setAutoPlay]     = useState(false);
  const [speed, setSpeed]           = useState(3500);

  const scenes = useMemo(() => buildTransformerScenario(data), [data]);
  const currentScene = scenes[sceneIndex];

  // Calculer activeStage selon la scène pour la TransformerBox
  const { activeStage, mode } = useMemo(() => {
    if (!currentScene) return { activeStage: null, mode: 'idle' };
    const id = currentScene.id;
    if (id === 'predict')          return { activeStage: null, mode: 'predict' };
    if (id === 'backprop')         return { activeStage: null, mode: 'backprop' };
    if (id.startsWith('embed'))    return { activeStage: 'encoder-input', mode: 'forward' };
    if (id.startsWith('encoder-attn'))   return { activeStage: 'encoder-attn', mode: 'forward' };
    if (id === 'encoder-multi-head')     return { activeStage: 'encoder-attn', mode: 'forward' };
    if (id === 'encoder-norm1')          return { activeStage: 'encoder-norm1', mode: 'forward' };
    if (id === 'encoder-ffn')            return { activeStage: 'encoder-ffn', mode: 'forward' };
    if (id === 'encoder-output')         return { activeStage: 'encoder-norm2', mode: 'forward' };
    if (id === 'decoder-target')         return { activeStage: 'decoder-target', mode: 'forward' };
    if (id === 'decoder-masked')         return { activeStage: 'decoder-masked', mode: 'forward' };
    if (id === 'decoder-cross')          return { activeStage: 'decoder-cross', mode: 'forward' };
    if (id === 'decoder-output')         return { activeStage: 'decoder-norm2', mode: 'forward' };
    return { activeStage: null, mode: 'idle' };
  }, [currentScene]);

  const runDemo = async () => {
    setLoading(true); setError(null); setSceneIndex(0); setAutoPlay(false);
    try {
      const res = await fetch(`${API}/api/transformer_demo/run`, {
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
            style={{ background:'linear-gradient(135deg, var(--tint-info-bg), #1a3a5c)', border:`1px solid ${C.encoder}40` }}>
            <Network size={18} style={{ color: C.encoder }} />
          </div>
          <div>
            <h2 className="text-lg font-semibold" style={{ color:'var(--text-primary)' }}>
              Démo Transformer — Mini-dataset synthétique
            </h2>
            <p className="text-xs" style={{ color:'var(--text-tertiary)' }}>
              Encoder + Decoder · Multi-Head Attention · Positional Encoding · Cross-Attention
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
              style={{ background:'var(--tint-info-bg)', color: C.encoder }}>
              Transformer
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
            onRun={runDemo} loading={loading} onSwitchTo={onSwitchTo}/>
        </div>

        <div className="col-span-9 space-y-4">
          {!data && !loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <Network size={32} style={{ color:'var(--text-muted)' }} className="mx-auto mb-3"/>
              <p className="text-sm" style={{ color:'var(--text-tertiary)' }}>
                Configure à gauche puis clique <b>Lancer la démo Transformer</b>
              </p>
            </div>
          )}

          {loading && (
            <div className="rounded-xl border p-8 text-center"
              style={{ background:'var(--bg-base)', borderColor:'var(--border-default)' }}>
              <p className="text-sm animate-pulse" style={{ color: C.encoder }}>
                ⚙️ Calcul Transformer (multi-head attention)...
              </p>
            </div>
          )}

          {data && (
            <>
              <TransformerBox config={data.config}
                activeStage={activeStage} mode={mode}
                sceneId={currentScene?.id || ''}/>

              <div className="rounded-xl border p-3" style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <button onClick={() => setAutoPlay(!autoPlay)}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5"
                    style={{ background:'var(--bg-elevated)', border:`1px solid ${C.encoder}40`, color: C.encoder }}>
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
                    style={{ background:'var(--bg-elevated)', border:`1px solid ${C.encoder}40`, color: C.encoder }}>
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
                          background: speed === s ? 'var(--tint-info-bg)' : 'var(--bg-elevated)',
                          borderColor: speed === s ? C.encoder : 'var(--border-default)',
                          color: speed === s ? C.encoder : 'var(--text-muted)',
                        }}>
                        {['🐢','🚶','🏃','⚡'][i]}
                      </button>
                    ))}
                  </div>
                  <div className="flex-1"/>
                  <span className="text-xs font-mono px-2 py-1 rounded"
                    style={{ background:'var(--bg-elevated)', color: C.encoder }}>
                    {sceneIndex + 1} / {scenes.length}
                  </span>
                </div>
                <div className="w-full h-1 rounded overflow-hidden" style={{ background:'var(--bg-deep)' }}>
                  <div className="h-full transition-all duration-300"
                    style={{
                      width: `${((sceneIndex + 1) / scenes.length) * 100}%`,
                      background: 'linear-gradient(90deg,var(--accent-blue),var(--accent-green),var(--accent-orange),#f06292,var(--accent-purple))',
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