interface Props {
  previewUrl: string;
}

export default function LoadingSpinner({ previewUrl }: Props) {
  return (
    <div className="space-y-5">
      {/* Image preview with scanning line */}
      <div className="relative overflow-hidden rounded-2xl">
        <img
          src={previewUrl}
          alt="Captured book cover"
          className="w-full rounded-2xl object-cover opacity-60"
        />
        {/* Scanning line */}
        <div className="pointer-events-none absolute inset-x-0 animate-scan h-0.5 bg-gradient-to-r from-transparent via-sky-400 to-transparent shadow-[0_0_15px_3px_rgba(56,189,248,0.4)]" />
      </div>

      {/* Status text */}
      <div className="flex items-center justify-center gap-3 text-slate-400">
        <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
          />
        </svg>
        <span className="text-sm">Identifying book&hellip;</span>
      </div>
    </div>
  );
}
