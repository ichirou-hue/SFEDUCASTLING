export default function EvalBar({ diff = 0, flipped, height }) {
  const MAX = 10;
  const signed = Number.isFinite(diff) ? diff : 0;
  const clamped = Math.max(-MAX, Math.min(MAX, signed));

  const signedBottom = flipped ? -clamped : clamped;
  const fillPct = 50 + (signedBottom / MAX) * 50;

  const label = (() => {
    const v = Math.round(clamped);
    if (v === 0) return "";
    return v > 0 ? `+${v}` : `${v}`;
  })();

  return (
    <div className="eval-bar" style={{ height: height || undefined }}>
      <div
        className="eval-bar-white"
        style={{ height: `${fillPct}%`, top: "auto", bottom: 0 }}
      >
        {label && <span className="eval-bar-label">{label}</span>}
      </div>
    </div>
  );
}