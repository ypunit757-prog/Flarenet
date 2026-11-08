import { letterOf } from "../lib/format.js";

/** A flare class such as M2.0: the letter with a bar in the class colour, so colour is never the only signal. */
export default function ClassTag({ cls }) {
  return <span className={`tag ${letterOf(cls)}`}>{cls}</span>;
}
