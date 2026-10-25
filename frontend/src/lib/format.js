// Pure helpers: flare class from flux, durations, percentages.
export const iso = (s) => (typeof s === "number" ? s : Date.parse(s) / 1000);

export function classOf(f) {
  if (!isFinite(f) || f <= 0) return "–";
  if (f >= 1e-4) return "X" + (f / 1e-4).toFixed(1);
  if (f >= 1e-5) return "M" + (f / 1e-5).toFixed(1);
  if (f >= 1e-6) return "C" + (f / 1e-6).toFixed(1);
  if (f >= 1e-7) return "B" + (f / 1e-7).toFixed(1);
  return "A" + (f / 1e-8).toFixed(1);
}

export function dur(min) {
  const h = Math.floor(min / 60), r = Math.round(min % 60);
  return h ? (r ? `${h} h ${r} min` : `${h} h`) : `${r} min`;
}

export const pct = (p) => Math.round(p * 100) + "%";
export const isMX = (f) => /^[MX]/.test(f.class);
