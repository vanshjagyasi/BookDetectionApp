interface Props {
  message: string;
  onRetry: () => void;
}

export default function ErrorMessage({ message, onRetry }: Props) {
  return (
    <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-6 text-center">
      <svg
        className="mx-auto mb-3 h-10 w-10 text-red-400"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.5}
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
        />
      </svg>
      <p className="mb-4 text-red-300">{message}</p>
      <button
        onClick={onRetry}
        className="rounded-xl bg-red-500/20 px-6 py-2.5 text-sm font-medium text-red-300 transition hover:bg-red-500/30"
      >
        Try Again
      </button>
    </div>
  );
}
