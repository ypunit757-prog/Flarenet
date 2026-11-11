// Thin wrapper over the browser Notification API. Every call is guarded: some browsers (and iOS Safari outside an
// installed web app) have no Notification at all, and a blocked or private window can throw.

export const supported = () => typeof Notification !== "undefined";
export const permission = () => (supported() ? Notification.permission : "unsupported");

export async function request() {
  if (!supported()) return "unsupported";
  try { return (await Notification.requestPermission()) || Notification.permission; } catch { return permission(); }
}

/** Shows one notification; returns true if the browser accepted it. */
export function show({ title, body, tag }) {
  if (permission() !== "granted") return false;
  try { new Notification(title, { body, tag }); return true; } catch { return false; }
}

const KEY_S = "flarenet-alert-settings", KEY_M = "flarenet-alert-memory";
export function load(key) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null; } catch { return null; } }
export function save(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage blocked */ } }
export const KEYS = { settings: KEY_S, memory: KEY_M };
