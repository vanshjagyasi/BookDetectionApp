import BookResultCard from "./components/BookResultCard";
import CameraCapture from "./components/CameraCapture";
import ErrorMessage from "./components/ErrorMessage";
import LoadingSpinner from "./components/LoadingSpinner";
import { useBookDetection } from "./hooks/useBookDetection";

export default function App() {
  const { status, result, error, previewUrl, detect, reset } =
    useBookDetection();

  return (
    <div className="mx-auto flex min-h-[100dvh] max-w-md flex-col px-4 py-6">
      {/* Header */}
      <header className="mb-8 text-center">
        <h1 className="text-2xl font-bold tracking-tight text-white">
          <span className="text-sky-400">Book</span>Scanner
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          AI-powered book identification
        </p>
      </header>

      {/* Main content */}
      <main className="flex flex-1 flex-col justify-center">
        {status === "idle" && <CameraCapture onCapture={detect} />}

        {status === "loading" && previewUrl && (
          <LoadingSpinner previewUrl={previewUrl} />
        )}

        {status === "success" && result && (
          <BookResultCard
            book={result.book}
            extractionNotes={result.extraction_notes}
            previewUrl={previewUrl}
            onReset={reset}
          />
        )}

        {status === "error" && (
          <div className="space-y-5">
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Scanned cover"
                className="w-full rounded-2xl object-cover opacity-40"
              />
            )}
            <ErrorMessage message={error ?? "Unknown error"} onRetry={reset} />
          </div>
        )}
      </main>
    </div>
  );
}
