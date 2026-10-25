import { createContext, useCallback, useContext, useMemo, useState } from "react";

// ---- time zone formatting ----
const TimeCtx = createContext(null);
export function TimeProvider({ children }) {
  const [tz, setTz] = useState("UTC");
  const value = useMemo(() => {
    const fmt = (u, o) => new Intl.DateTimeFormat("en-GB", { timeZone: tz === "UTC" ? "UTC" : undefined, hour12: false, ...o }).format(new Date(u * 1000));
    const sfx = tz === "UTC" ? " UTC" : "";
    return {
      tz, setTz, fmt,
      fmtDT: (u) => fmt(u, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) + sfx,
      fmtClock: (u) => fmt(u, { hour: "2-digit", minute: "2-digit" }) + sfx,
    };
  }, [tz]);
  return <TimeCtx.Provider value={value}>{children}</TimeCtx.Provider>;
}
export const useTime = () => useContext(TimeCtx);

// ---- toasts ----
const ToastCtx = createContext(() => {});
export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const push = useCallback((msg) => {
    const id = Math.random();
    setItems((x) => [...x, { id, msg }]);
    setTimeout(() => setItems((x) => x.filter((i) => i.id !== id)), 5000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div id="toasts" aria-live="polite">{items.map((i) => <div className="toast" key={i.id}>{i.msg}</div>)}</div>
    </ToastCtx.Provider>
  );
}
export const useToast = () => useContext(ToastCtx);
