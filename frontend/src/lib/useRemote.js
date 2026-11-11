import { useCallback, useEffect, useRef, useState } from "react";
import { getJSON } from "./remote.js";

/** Loads one JSON document. `url` null means "not available here" and gives status "off" without any request. */
export function useRemote(url) {
  const [s, setS] = useState({ status: url ? "loading" : "off", data: null, error: null });
  const seq = useRef(0);
  const load = useCallback(async () => {
    if (!url) { setS({ status: "off", data: null, error: null }); return; }
    const my = ++seq.current;
    setS((x) => ({ ...x, status: "loading", error: null }));
    try {
      const data = await getJSON(url);
      if (my === seq.current) setS({ status: "ok", data, error: null });
    } catch (e) {
      if (my === seq.current) setS({ status: "error", data: null, error: e });
    }
  }, [url]);
  useEffect(() => { load(); return () => { seq.current++; }; }, [load]);
  return { ...s, reload: load };
}
