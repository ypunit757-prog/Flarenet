// Where the extra data comes from. The forecast itself is loaded by useForecastData; the pages below add the alert history
// and daily forward test (served by the Render API) and the model comparison (a static file next to the site).

const env = (typeof import.meta !== "undefined" && import.meta.env) || {};
/** The Render API, or "" when the site was built without VITE_API_URL. */
export const API = String(env.VITE_API_URL || "").replace(/\/+$/, "");
const BASE = env.BASE_URL || "/";

/** URL of an API route such as "alerts?limit=50", or null when no API is configured. */
export const apiUrl = (path) => (API ? `${API}/api/${path}` : null);
/** URL of a file published next to the site, such as model_comparison.json. */
export const staticUrl = (name) => `${BASE}${name}`;

/** Fetch JSON; throws an Error whose message says what went wrong (never returns an HTML error page as data). */
export async function getJSON(url, { signal, fetcher = globalThis.fetch } = {}) {
  const r = await fetcher(url, { cache: "no-store", signal });
  if (!r.ok) throw Object.assign(new Error(r.status === 404 ? "not found" : `HTTP ${r.status}`), { status: r.status });
  const type = r.headers?.get?.("content-type") || "";
  if (type && !/json/i.test(type)) throw new Error("the server did not return JSON");
  try { return await r.json(); } catch { throw new Error("the response was not valid JSON"); }
}
