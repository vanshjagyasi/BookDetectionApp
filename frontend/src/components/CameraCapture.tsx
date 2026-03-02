import { useRef } from "react";

interface Props {
  onCapture: (file: File) => void;
  disabled?: boolean;
}

const MAX_SIZE_MB = 20;

export default function CameraCapture({ onCapture, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input so the same file can be re-selected
    e.target.value = "";

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      alert(`Image is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is ${MAX_SIZE_MB} MB.`);
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    onCapture(file);
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => inputRef.current?.click()}
      className="group flex w-full flex-col items-center gap-4 rounded-3xl border-2 border-dashed border-slate-700 bg-slate-900/50 p-10 transition hover:border-sky-500/50 hover:bg-slate-800/50 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
    >
      {/* Camera icon */}
      <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-sky-500/10 text-sky-400 transition group-hover:bg-sky-500/20">
        <svg className="h-10 w-10" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z"
          />
        </svg>
      </div>

      <div className="text-center">
        <p className="text-lg font-semibold text-slate-200">Scan a Book Cover</p>
        <p className="mt-1 text-sm text-slate-500">
          Point your camera at the front, back, or spine
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleChange}
        className="hidden"
      />
    </button>
  );
}
