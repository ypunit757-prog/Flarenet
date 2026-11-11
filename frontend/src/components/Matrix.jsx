/** A 2 x 2 confusion matrix: rows are what happened, columns what the model said. cm = [[TN, FP], [FN, TP]]. */
export default function Matrix({ cm, label = "Confusion matrix" }) {
  if (!Array.isArray(cm) || cm.length !== 2 || cm.some((r) => !Array.isArray(r) || r.length !== 2)) return null;
  const [[tn, fp], [fn, tp]] = cm;
  const n = Math.max(1, tn + fp + fn + tp);
  const cell = (v, kind, name) => (
    <td className={`cm ${kind}`}>
      <b className="num">{Number(v).toLocaleString("en-GB")}</b>
      <span>{name}</span>
      <i className="num">{((v / n) * 100).toFixed(1)}% of all</i>
    </td>
  );
  return (
    <div className="wrapx">
      <table className="matrix" aria-label={label}>
        <thead><tr><td /><th scope="col">No alert</th><th scope="col">Alert</th></tr></thead>
        <tbody>
          <tr><th scope="row">No flare followed</th>{cell(tn, "good", "correct quiet")}{cell(fp, "bad", "false alarm")}</tr>
          <tr><th scope="row">M/X flare followed</th>{cell(fn, "bad", "missed")}{cell(tp, "good", "caught")}</tr>
        </tbody>
      </table>
    </div>
  );
}
