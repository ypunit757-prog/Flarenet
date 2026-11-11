import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULTS, dataProblem, decide, normalizeSettings } from "./alerts.js";
import * as N from "./notify.js";

/**
 * Browser notifications for the forecast on screen. They work only while a FlareNet tab is open: a web page cannot
 * run in the background. For alerts with the page closed, use the e-mail alerts (docs/DEPLOY.md).
 */
export function useBrowserAlerts(D, onToast) {
  const [settings, setSettingsState] = useState(() => normalizeSettings(N.load(N.KEYS.settings) || DEFAULTS));
  const [perm, setPerm] = useState(N.permission());
  const [last, setLast] = useState(null);       // what the last check did, for the settings page
  const memory = useRef(N.load(N.KEYS.memory));
  const seenData = useRef(null);
  const toast = useRef(onToast);
  toast.current = onToast;

  const setSettings = useCallback((patch) => {
    setSettingsState((s) => { const n = normalizeSettings({ ...s, ...patch }); N.save(N.KEYS.settings, n); return n; });
  }, []);

  useEffect(() => {
    if (!D) return;
    // Always track the state, even with alerts off, so turning them on later does not announce an old alert.
    const isNew = seenData.current !== D;   // a changed setting must not announce what was already on screen
    seenData.current = D;
    const r = decide(D, settings, memory.current);
    memory.current = r.memory;
    N.save(N.KEYS.memory, r.memory);
    setLast({ at: Date.now(), skipped: r.skipped, sent: r.skipped ? [] : r.notify.map((n) => n.kind) });
    if (!isNew || !settings.enabled || r.skipped) return;
    for (const n of r.notify) {
      if (!N.show(n)) toast.current?.(`${n.title}. ${n.body}`);
    }
  }, [D, settings]);

  const enable = useCallback(async () => {
    const p = await N.request();
    setPerm(p);
    setSettings({ enabled: p === "granted" });
    return p;
  }, [setSettings]);

  const test = useCallback(() => {
    const ok = N.show({ title: "FlareNet: test notification", body: "This is a test. No flare has been forecast or observed.", tag: "test" });
    if (!ok) toast.current?.("FlareNet test: notifications are blocked or unsupported in this browser.");
    return ok;
  }, []);

  return { settings, setSettings, permission: perm, enable, test, last, problem: dataProblem(D) };
}
