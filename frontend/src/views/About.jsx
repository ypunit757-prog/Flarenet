import Chip from "../components/Chip.jsx";

export default function About({ store }) {
  const D = store.data;
  return (
    <div className="stack" style={{ maxWidth: 900 }}>
      <section className="card prose">
        <h2>What FlareNet does</h2>
        <p>FlareNet reads the Sun's X-ray flux, one value per minute. The <b>nowcast</b> names the flare class happening right now. The <b>forecast</b> scores the chance of an M or X flare starting in the next six hours. The nowcast and the network's share of the forecast come from one shared neural network.{D && D.model.stack && D.model.stack.used && <> The forecast score also blends in a gradient-boosted tree model that reads the last three days of X-ray history (how active the Sun has been, how long since the last M flare); the first three days of the chart have too little history behind them and use the network alone.</>}</p>
        <h2>Flare classes</h2>
        <div className="wrapx">
          <table>
            <thead><tr><th>Class</th><th>Peak 0.1–0.8 nm flux</th><th>Typical effect</th></tr></thead>
            <tbody>
              {[["A", "< 1e-7 W/m²", "Background"], ["B", "1e-7 to 1e-6", "Negligible"], ["C", "1e-6 to 1e-5", "Minor"], ["M", "1e-5 to 1e-4", "Brief radio blackouts at the poles"], ["X", "≥ 1e-4", "Strong, can cause wide radio blackouts"]].map(([c, f, e]) => (
                <tr key={c}><td><Chip cls={c} /></td><td className="mono">{f}</td><td>{e}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <h2>How to read the metrics</h2>
        <ul>
          <li><b>TSS</b> (True Skill Statistic): hit rate minus false-alarm rate. The standard score for rare events because it does not reward always saying "no flare".</li>
          <li><b>POD / FAR</b>: share of flares caught, and share of alerts that were wrong.</li>
          <li><b>Lead time</b>: how long before the flare started the first alert came.</li>
          <li><b>AUC, HSS, Brier</b>: ranking quality, skill against chance, and probability accuracy.</li>
        </ul>
        <h2>Limits</h2>
        <ul>
          <li>It sees only X-ray flux. It has no sunspot or magnetic-field data, which operational forecasters use.</li>
          <li>It forecasts flares, not their effects. Geomagnetic storms from coronal mass ejections can arrive one to three days later and are not predicted here.</li>
          <li>The threshold maximises TSS on validation data. {D && D.model.calibrated ? "The score was calibrated on validation data, but it is only as good as that period resembles the next one." : "The score is not a calibrated chance."}</li>
          <li>All results are from GOES. ISRO Aditya-L1 and other missions are not tested yet.</li>
          <li>Flare times come from an NOAA-style rule on the flux and can differ by minutes from NOAA's list.</li>
        </ul>
        {D && <p className="muted">Data: {D.source.name} ({D.source.kind}), exported {D.generated || "n/a"}.</p>}
      </section>
    </div>
  );
}
