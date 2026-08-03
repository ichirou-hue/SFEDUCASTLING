export default function EvalBar({ evaluation }) {
  const whitePercent = (() => {
    if (!evaluation) return 50;
    if (evaluation.type === "mate") return evaluation.value > 0 ? 100 : 0;
    const cp = evaluation.value;
    return 50 - 35 * Math.tanh(cp / 400);
  })();

  const label = (() => {
    if (!evaluation) return "";
    if (evaluation.type === "mate") return `M${Math.abs(evaluation.value)}`;
    const cp = evaluation.value;
    const sign = cp >= 0 ? "+" : "";
    return `${sign}${(cp / 100).toFixed(1)}`;
  })();

  return (
    <div className="eval-bar">
      <div className="eval-bar-white" style={{ height: `${whitePercent}%` }} />
    </div>
  );
}
