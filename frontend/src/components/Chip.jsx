export default function Chip({ cls }) {
  return <span className={`chip ${String(cls)[0]}`}>{cls}</span>;
}
