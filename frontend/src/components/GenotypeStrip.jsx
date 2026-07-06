/**
 * GenotypeStrip — the app's signature motif.
 *
 * Renders a latent vector as a band of vertical ticks: value above the
 * baseline reads on the green (genotype) channel, below it on the magenta
 * (phenotype) channel. This is the W-Vector Editor's canvas promoted to the
 * brand mark — the literal picture of "an image is an individual, its vector
 * is its DNA". Purely decorative here, so it's hidden from assistive tech.
 */

// Deterministic pseudo-values so the strip is stable across renders (no flicker).
function strand(count, seed) {
  const out = [];
  let s = seed;
  for (let i = 0; i < count; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    out.push((s / 0x7fffffff) * 2 - 1); // -1..1
  }
  return out;
}

export function GenotypeStrip({
  count = 96,
  seed = 7,
  height = 34,
  gap = 2,
  className = '',
  style,
}) {
  const values = strand(count, seed);
  const tick = 2;
  const step = tick + gap;
  const width = count * step;
  const mid = height / 2;

  return (
    <svg
      className={`genotype-strip ${className}`}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      style={style}
    >
      {values.map((v, i) => {
        const h = Math.max(1.5, Math.abs(v) * (mid - 2));
        const positive = v >= 0;
        return (
          <rect
            key={i}
            x={i * step}
            y={positive ? mid - h : mid}
            width={tick}
            height={h}
            rx={1}
            fill={positive ? 'var(--gfp)' : 'var(--mag)'}
          />
        );
      })}
    </svg>
  );
}
