export default function Note({ kind = "info", children }) {
  return <p className={`note ${kind}`} role="note">{children}</p>;
}
