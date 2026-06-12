/**
 * ArchitectureVisualizer.jsx — Schéma vivant Cevital (LSTM/GRU + Embedding).
 *
 * Différence majeure vs version Azure (1 input) :
 *   • 2 inputs en parallèle :
 *       - Branche HAUTE  : X_num  (lookback, n_features=9)
 *       - Branche BASSE  : X_comp (1 idx) → Embedding(dim) → Flatten → RepeatVector
 *   • Fusion CONCAT au milieu
 *   • Couches LSTM/GRU empilées (jusqu'à 3)
 *   • Dense final = 1 neurone (RUL en jours)
 *
 * Style préservé du visualiseur précédent :
 *   - Cercles "neurones" avec dropout en pointillés
 *   - Pile batch sur la gauche
 *   - Légende en bas
 *   - Couleurs via variables CSS uniquement
 */
import { Eye } from 'lucide-react';

// Couleurs accents (réutilisées du panel)
const ARCH_COLORS = {
  LSTM:        'var(--accent-blue)',
  GRU:         'var(--accent-green)',
  RNN:         'var(--accent-orange)',
  Transformer: 'var(--accent-purple)',
};
const N_FEATURES = 9;            // Cevital : 9 features modèle
const MAX_DISPLAY_NEURONS = 6;   // Cap visuel

export default function ArchitectureVisualizer({
  arch         = 'LSTM',
  numLayers    = 2,
  units        = [64, 32, 32],
  dropouts     = [0.2, 0.15, 0.1],
  batchSize    = 32,
  lookback     = 30,
  embeddingDim = 8,
  numClassesComp = 100,
}) {
  const archColor  = ARCH_COLORS[arch] || 'var(--accent-blue)';
  const embedColor = 'var(--accent-purple)';
  const pinkColor  = 'var(--accent-pink)';

  // ─── Géométrie SVG ──────────────────────────────────────────
  const W = 1020, H = 420;
  const xBatch     = 70;
  const xInputNum  = 165;
  const xInputComp = 165;
  const yMidNum    = 130;
  const yMidComp   = 290;
  const yConcat    = (yMidNum + yMidComp) / 2;
  const xEmbedding = 280;
  const xConcat    = 380;
  const xLayers    = [490, 620, 750];   // max 3 couches affichées
  const xDense     = 870;
  const xOutput    = 970;

  const layersToShow = Math.min(3, Math.max(1, numLayers));
  const concatFeatures = N_FEATURES + (Number(embeddingDim) || 0);

  return (
    <div className="rounded-xl border p-4"
         style={{ background:'var(--bg-card)', borderColor:'var(--border-default)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Eye size={14} style={{ color: archColor }} />
          <h3 className="text-sm font-bold" style={{ color: archColor }}>
            🎨 Architecture en direct — {arch}
          </h3>
        </div>
        <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
          Aperçu avant entraînement · 2 entrées (X_num + X_comp)
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minHeight: 300 }}>

        {/* Définitions flèches */}
        <defs>
          <marker id="arrow-arch" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 z" fill={archColor}/>
          </marker>
          <marker id="arrow-embed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 z" fill={embedColor}/>
          </marker>
          <marker id="arrow-pink" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 z" fill={pinkColor}/>
          </marker>
        </defs>

        {/* ════════════════════════════════════════════════ */}
        {/* BRANCHE 1 : INPUT NUMÉRIQUE (en haut)             */}
        {/* ════════════════════════════════════════════════ */}
        <text x={xBatch} y={50} textAnchor="middle"
              fill="var(--text-tertiary)" fontSize="11" fontWeight="bold"
              fontFamily="monospace">
          BATCH = {batchSize}
        </text>
        <BatchStack x={xBatch} y={yMidNum} batchSize={batchSize} color={archColor}/>

        <text x={xInputNum} y={yMidNum - 78} textAnchor="middle"
              fill={archColor} fontSize="11" fontWeight="bold">
          INPUT_NUM
        </text>
        <text x={xInputNum} y={yMidNum - 63} textAnchor="middle"
              fill="var(--text-muted)" fontSize="10">
          {lookback}j × {N_FEATURES} feat.
        </text>
        <rect x={xInputNum - 28} y={yMidNum - 50}
              width="56" height="100"
              fill={`color-mix(in srgb, ${archColor} 25%, var(--bg-card))`}
              stroke={archColor} strokeWidth="1.5" rx="4"/>
        {/* lignes intérieures pour suggérer le tenseur (lookback × features) */}
        {[0.25, 0.5, 0.75].map(f => (
          <line key={f}
                x1={xInputNum - 28} y1={yMidNum - 50 + f * 100}
                x2={xInputNum + 28} y2={yMidNum - 50 + f * 100}
                stroke={archColor} strokeOpacity="0.35"/>
        ))}

        {/* Flèche INPUT_NUM → CONCAT */}
        <line x1={xInputNum + 28} y1={yMidNum}
              x2={xConcat - 32}   y2={yConcat - 22}
              stroke={archColor} strokeWidth="1.5"
              markerEnd="url(#arrow-arch)"/>

        {/* ════════════════════════════════════════════════ */}
        {/* BRANCHE 2 : INPUT COMPOSANT (en bas)              */}
        {/* ════════════════════════════════════════════════ */}
        <text x={xInputComp} y={yMidComp - 38} textAnchor="middle"
              fill={embedColor} fontSize="11" fontWeight="bold">
          INPUT_COMP
        </text>
        <text x={xInputComp} y={yMidComp - 23} textAnchor="middle"
              fill="var(--text-muted)" fontSize="10">
          1 idx ∈ [0, {numClassesComp - 1}]
        </text>
        <rect x={xInputComp - 20} y={yMidComp - 10}
              width="40" height="35"
              fill={`color-mix(in srgb, ${embedColor} 25%, var(--bg-card))`}
              stroke={embedColor} strokeWidth="1.5" rx="3"/>
        <text x={xInputComp} y={yMidComp + 14} textAnchor="middle"
              fill={embedColor} fontSize="13" fontWeight="bold" fontFamily="monospace">
          c
        </text>

        {/* Embedding box */}
        <g>
          <rect x={xEmbedding - 45} y={yMidComp - 22}
                width="90" height="44"
                fill={`color-mix(in srgb, ${embedColor} 18%, var(--bg-card))`}
                stroke={embedColor} strokeWidth="1.8" rx="6"/>
          <text x={xEmbedding} y={yMidComp - 4} textAnchor="middle"
                fill={embedColor} fontSize="11" fontWeight="bold">
            Embedding
          </text>
          <text x={xEmbedding} y={yMidComp + 11} textAnchor="middle"
                fill={embedColor} fontSize="10" fontFamily="monospace">
            dim = {embeddingDim}
          </text>
        </g>

        {/* Flèche INPUT_COMP → Embedding */}
        <line x1={xInputComp + 20} y1={yMidComp + 7}
              x2={xEmbedding - 45} y2={yMidComp}
              stroke={embedColor} strokeWidth="1.5"
              markerEnd="url(#arrow-embed)"/>

        {/* Flèche Embedding → CONCAT (avec mention Flatten + Repeat) */}
        <line x1={xEmbedding + 45} y1={yMidComp}
              x2={xConcat - 32}    y2={yConcat + 22}
              stroke={embedColor} strokeWidth="1.5"
              strokeDasharray="4 3"
              markerEnd="url(#arrow-embed)"/>
        <text x={(xEmbedding + xConcat) / 2 + 12}
              y={(yMidComp + yConcat) / 2 + 4}
              fill={embedColor} fontSize="9" fontStyle="italic"
              textAnchor="middle">
          Flatten · RepeatVector({lookback})
        </text>

        {/* ════════════════════════════════════════════════ */}
        {/* CONCAT (fusion 2 branches)                        */}
        {/* ════════════════════════════════════════════════ */}
        <g>
          <rect x={xConcat - 32} y={yConcat - 22}
                width="64" height="44"
                fill="var(--bg-elevated)"
                stroke="var(--accent-orange)" strokeWidth="2"
                strokeDasharray="3 2" rx="5"/>
          <text x={xConcat} y={yConcat - 5} textAnchor="middle"
                fill="var(--accent-orange)" fontSize="10" fontWeight="bold">
            CONCAT
          </text>
          <text x={xConcat} y={yConcat + 11} textAnchor="middle"
                fill="var(--text-muted)" fontSize="9" fontFamily="monospace">
            ({concatFeatures})
          </text>
        </g>

        {/* Flèche CONCAT → Couche 1 */}
        <line x1={xConcat + 32} y1={yConcat}
              x2={xLayers[0] - 35} y2={yConcat}
              stroke={archColor} strokeWidth="1.5"
              markerEnd="url(#arrow-arch)"/>

        {/* ════════════════════════════════════════════════ */}
        {/* COUCHES RÉCURRENTES (LSTM/GRU empilées)           */}
        {/* ════════════════════════════════════════════════ */}
        {Array.from({ length: layersToShow }).map((_, i) => (
          <LayerVisualization key={i}
            x={xLayers[i]} y={yConcat}
            architecture={arch}
            archColor={archColor}
            units={units[i] ?? 32}
            dropout={dropouts[i] ?? 0.2}
            isLast={i === layersToShow - 1}/>
        ))}

        {/* Flèches entre couches */}
        {Array.from({ length: layersToShow - 1 }).map((_, i) => (
          <line key={i}
                x1={xLayers[i] + 35} y1={yConcat}
                x2={xLayers[i+1] - 35} y2={yConcat}
                stroke={archColor} strokeWidth="1.5"
                markerEnd="url(#arrow-arch)"/>
        ))}

        {/* ════════════════════════════════════════════════ */}
        {/* DENSE FINAL                                       */}
        {/* ════════════════════════════════════════════════ */}
        <g>
          <circle cx={xDense} cy={yConcat} r="20"
                  fill={`color-mix(in srgb, ${pinkColor} 30%, var(--bg-card))`}
                  stroke={pinkColor} strokeWidth="2"/>
          <text x={xDense} y={yConcat + 4} textAnchor="middle"
                fill={pinkColor} fontSize="11" fontWeight="bold">
            DENSE
          </text>
          <text x={xDense} y={yConcat + 38} textAnchor="middle"
                fill="var(--text-muted)" fontSize="9" fontFamily="monospace">
            1 (RUL · j)
          </text>
        </g>

        {/* Flèche dernière couche → Dense */}
        <line x1={xLayers[layersToShow - 1] + 35} y1={yConcat}
              x2={xDense - 20} y2={yConcat}
              stroke={archColor} strokeWidth="1.5"
              markerEnd="url(#arrow-arch)"/>

        {/* OUTPUT ŷ */}
        <line x1={xDense + 20} y1={yConcat}
              x2={xOutput - 10} y2={yConcat}
              stroke={pinkColor} strokeWidth="1.5"
              markerEnd="url(#arrow-pink)"/>
        <text x={xOutput} y={yConcat + 6} textAnchor="middle"
              fill={pinkColor} fontSize="22" fontWeight="bold" fontFamily="monospace">
          ŷ
        </text>

        {/* Légende en bas */}
        <g transform={`translate(${W/2 - 280}, ${H - 18})`}>
          <rect x={0} y={-9} width={560} height={18} rx={8}
                fill="var(--bg-card)" stroke="var(--border-default)"/>
          <circle cx={20} cy={0} r={4} fill={`color-mix(in srgb, ${archColor} 30%, transparent)`} stroke={archColor}/>
          <text x={30} y={3} fontSize={9} fill="var(--text-tertiary)" fontFamily="monospace">neurone actif</text>
          <circle cx={150} cy={0} r={4} fill="var(--bg-card)" stroke="var(--text-muted)" strokeDasharray="2,2"/>
          <text x={160} y={3} fontSize={9} fill="var(--text-tertiary)" fontFamily="monospace">neurone dropé</text>
          <text x={280} y={3} fontSize={9} fill="var(--text-tertiary)" fontFamily="monospace">📦 batch = {batchSize}</text>
          <text x={400} y={3} fontSize={9} fill={archColor} fontFamily="monospace">⏱ window = {lookback}j</text>
        </g>
      </svg>

      {/* Note pédagogique */}
      <p className="text-xs mt-3 px-3 py-2 rounded leading-relaxed"
         style={{
           background:  'var(--bg-elevated)',
           color:       'var(--text-tertiary)',
           borderLeft:  `3px solid ${pinkColor}`,
         }}>
        💡 <b style={{ color: pinkColor }}>Dense final = 1 neurone fixe</b> (non configurable)
        — c'est imposé par le problème : on prédit <b>une seule valeur</b>, le RUL en jours.
        L'embedding composant (taille <b style={{ color: embedColor }}>{embeddingDim}</b>)
        encode l'identité de chaque pièce (parmi {numClassesComp}) dans un espace continu
        que le réseau apprend en même temps que les poids LSTM/GRU.
      </p>
    </div>
  );
}


// ─── Pile batch (visualisation des séquences empilées en 3D) ───
function BatchStack({ x, y, batchSize, color }) {
  const DISPLAY = Math.min(batchSize, 6);
  const offset  = 4;
  return (
    <g>
      {Array.from({ length: DISPLAY }).map((_, i) => (
        <rect key={i}
          x={x - 22 + i * offset}
          y={y - 45 + i * offset}
          width="44" height="90"
          fill={`color-mix(in srgb, ${color} 15%, var(--bg-card))`}
          stroke={color} strokeWidth="1"
          opacity={1 - i * 0.1}
          rx="3"/>
      ))}
    </g>
  );
}


// ─── Couche LSTM/GRU (cercles neurones + récurrence + dropout) ─
function LayerVisualization({ x, y, architecture, archColor, units, dropout, isLast }) {
  const displayCount = Math.min(units, MAX_DISPLAY_NEURONS);
  const neuronRadius = 8;
  const spacingY     = 22;

  // Aléatoire mais déterministe par couche (sinon Math.random change à chaque rerender)
  const droppedSet = new Set();
  const seed = displayCount + Math.round(dropout * 100);
  for (let i = 0; i < displayCount; i++) {
    // mini PRNG déterministe
    const x_ = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453;
    const r  = x_ - Math.floor(x_);
    if (r < dropout) droppedSet.add(i);
  }

  return (
    <g>
      {/* Label */}
      <text x={x} y={y - 88} textAnchor="middle"
            fill={archColor} fontSize="11" fontWeight="bold">
        {architecture}
      </text>
      <text x={x} y={y - 73} textAnchor="middle"
            fill="var(--text-muted)" fontSize="9" fontFamily="monospace">
        {units} unit. · dr = {dropout.toFixed(2)}
      </text>

      {/* Neurones */}
      {Array.from({ length: displayCount }).map((_, i) => {
        const cy = y - ((displayCount - 1) * spacingY) / 2 + i * spacingY;
        const isDropped = droppedSet.has(i);
        return (
          <circle key={i}
            cx={x} cy={cy} r={neuronRadius}
            fill={isDropped ? 'var(--bg-card)' : `color-mix(in srgb, ${archColor} 30%, transparent)`}
            stroke={isDropped ? 'var(--text-muted)' : archColor}
            strokeWidth="1.5"
            opacity={isDropped ? 0.45 : 1}
            strokeDasharray={isDropped ? "2 2" : "none"}/>
        );
      })}

      {/* "..." si > MAX */}
      {units > MAX_DISPLAY_NEURONS && (
        <text x={x} y={y + ((displayCount - 1) * spacingY) / 2 + 22}
              textAnchor="middle"
              fill="var(--text-muted)" fontSize="9" fontFamily="monospace">
          ... (+{units - MAX_DISPLAY_NEURONS})
        </text>
      )}

      {/* Récurrence (icone ↻) sauf sur la dernière couche return_sequences=False */}
      {!isLast && (
        <text x={x + 28} y={y + 5}
              fill={archColor} fontSize="11" fontStyle="italic">
          ↻ rec.
        </text>
      )}
    </g>
  );
}
