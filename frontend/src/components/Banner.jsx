export default function Banner({ kind = "info", children }) {
  return <div className={`banner ${kind}`} role="note"><span>{children}</span></div>;
}
