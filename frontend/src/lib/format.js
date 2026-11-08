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

/** "1 h 40 min", "5 h", "3 days": how old data is, in words. */
export function ago(hours) {
  if (hours >= 48) return `${Math.round(hours / 24)} days`;
  return dur(Math.max(1, Math.round(hours * 60)));
}

export const letterOf = (cls) => (/^[ABCMX]/.test(String(cls)) ? String(cls)[0] : "A");
