/**
 * CevitalLogo.jsx — Logo CEVITAL inline (SVG vectoriel, thème-aware).
 *
 * Pas d'asset image : le logo est entièrement défini en SVG inline pour :
 *   - rester net à toute taille
 *   - respecter automatiquement la palette CEVITAL via les variables CSS
 *     (--brand-primary = bleu, --brand-secondary = jaune)
 *
 * Usage :
 *   <CevitalLogo size={36} />
 *   <CevitalLogo size={48} withText />   // affiche aussi "CEVITAL" à droite
 */
import React from 'react';

export default function CevitalLogo({
  size = 36,
  withText = false,
  className = '',
}) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* ─── Marque SVG ──────────────────────────────────────── */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 64 64"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Logo ML Platform"
      >
        {/* Fond arrondi bleu */}
        <rect x="2" y="2" width="60" height="60" rx="14" fill="var(--brand-primary)" />

        {/* Connexions couche input→cachée */}
        <line x1="14" y1="20" x2="32" y2="12" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.8"/>
        <line x1="14" y1="32" x2="32" y2="12" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.5"/>
        <line x1="14" y1="44" x2="32" y2="32" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.8"/>
        <line x1="14" y1="44" x2="32" y2="52" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.5"/>
        <line x1="14" y1="20" x2="32" y2="32" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.4"/>
        <line x1="14" y1="32" x2="32" y2="52" stroke="var(--brand-secondary)" strokeWidth="1.8" strokeOpacity="0.4"/>

        {/* Connexions couche cachée→output */}
        <line x1="32" y1="12" x2="50" y2="32" stroke="#ffffff" strokeWidth="1.8" strokeOpacity="0.9"/>
        <line x1="32" y1="32" x2="50" y2="32" stroke="#ffffff" strokeWidth="1.8" strokeOpacity="0.9"/>
        <line x1="32" y1="52" x2="50" y2="32" stroke="#ffffff" strokeWidth="1.8" strokeOpacity="0.9"/>

        {/* Nœuds input (jaunes) */}
        <circle cx="14" cy="20" r="4" fill="var(--brand-secondary)"/>
        <circle cx="14" cy="32" r="4" fill="var(--brand-secondary)"/>
        <circle cx="14" cy="44" r="4" fill="var(--brand-secondary)"/>

        {/* Nœuds cachés (blancs) */}
        <circle cx="32" cy="12" r="4" fill="#ffffff" opacity="0.95"/>
        <circle cx="32" cy="32" r="4" fill="#ffffff" opacity="0.95"/>
        <circle cx="32" cy="52" r="4" fill="#ffffff" opacity="0.95"/>

        {/* Nœud output (jaune, plus grand) */}
        <circle cx="50" cy="32" r="5.5" fill="var(--brand-secondary)"/>
      </svg>

      {/* ─── Wordmark optionnel ──────────────────────────────── */}
      {withText && (
        <div className="leading-none">
          <p
            className="font-bold tracking-wider"
            style={{
              color: 'var(--brand-primary)',
              fontSize: size * 0.45,
              letterSpacing: '0.08em',
            }}
          >
            CEVITAL
          </p>
          <p
            className="font-mono uppercase"
            style={{
              color: 'var(--text-tertiary)',
              fontSize: size * 0.22,
              letterSpacing: '0.18em',
              marginTop: 2,
            }}
          >
            PdM Platform
          </p>
        </div>
      )}
    </div>
  );
}
