interface Props {
  score: number;
}

export default function ConfidenceBadge({ score }: Props) {
  const pct = Math.round(score * 100);

  let color: string;
  let label: string;

  if (score >= 0.9) {
    color = "bg-emerald-500/20 text-emerald-400 ring-emerald-500/30";
    label = "High";
  } else if (score >= 0.7) {
    color = "bg-amber-500/20 text-amber-400 ring-amber-500/30";
    label = "Good";
  } else if (score >= 0.4) {
    color = "bg-orange-500/20 text-orange-400 ring-orange-500/30";
    label = "Partial";
  } else {
    color = "bg-red-500/20 text-red-400 ring-red-500/30";
    label = "Uncertain";
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${color}`}
    >
      {label} &middot; {pct}%
    </span>
  );
}
