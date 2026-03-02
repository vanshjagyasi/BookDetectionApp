import { useState } from "react";
import type { BookInfo } from "../types";
import ConfidenceBadge from "./ConfidenceBadge";

interface Props {
  book: BookInfo;
  extractionNotes: string | null;
  previewUrl: string | null;
  onReset: () => void;
}

function Stars({ rating }: { rating: number }) {
  return (
    <span className="inline-flex gap-0.5 text-amber-400">
      {[1, 2, 3, 4, 5].map((i) => (
        <svg
          key={i}
          className={`h-4 w-4 ${i <= Math.round(rating) ? "fill-current" : "fill-slate-700 text-slate-700"}`}
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      ))}
      <span className="ml-1 text-sm text-slate-400">{rating.toFixed(1)}</span>
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="mt-0.5 text-sm text-slate-200">{children}</dd>
    </div>
  );
}

export default function BookResultCard({ book, extractionNotes, previewUrl, onReset }: Props) {
  const [synopsisExpanded, setSynopsisExpanded] = useState(false);

  return (
    <div className="space-y-5">
      {/* Preview image */}
      {previewUrl && (
        <img
          src={previewUrl}
          alt="Scanned cover"
          className="w-full rounded-2xl object-cover"
        />
      )}

      {/* Result card */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
        {/* Header: title + confidence */}
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-xl font-bold text-white truncate">
              {book.title ?? "Unknown Title"}
            </h2>
            {book.author && (
              <p className="mt-0.5 text-sm text-slate-400">by {book.author}</p>
            )}
          </div>
          <ConfidenceBadge score={book.confidence_score} />
        </div>

        {/* Metadata grid */}
        <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3">
          {book.isbn && <Field label="ISBN">{book.isbn}</Field>}
          {book.publisher && <Field label="Publisher">{book.publisher}</Field>}
          {book.publication_year && <Field label="Year">{book.publication_year}</Field>}
          {book.genre && <Field label="Genre">{book.genre}</Field>}
          {book.price != null && (
            <Field label="Price">${book.price.toFixed(2)}</Field>
          )}
          {book.rating != null && (
            <Field label="Rating">
              <Stars rating={book.rating} />
            </Field>
          )}
        </dl>

        {/* Tags */}
        {book.tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {book.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-400 ring-1 ring-inset ring-sky-500/20"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Synopsis */}
        {book.synopsis && (
          <div className="mt-4">
            <p
              className={`text-sm leading-relaxed text-slate-400 ${
                !synopsisExpanded ? "line-clamp-3" : ""
              }`}
            >
              {book.synopsis}
            </p>
            <button
              onClick={() => setSynopsisExpanded(!synopsisExpanded)}
              className="mt-1 text-xs font-medium text-sky-400 hover:text-sky-300"
            >
              {synopsisExpanded ? "Show less" : "Show more"}
            </button>
          </div>
        )}

        {/* Debug notes */}
        {extractionNotes && (
          <p className="mt-4 text-xs text-slate-600">{extractionNotes}</p>
        )}
      </div>

      {/* Scan another */}
      <button
        onClick={onReset}
        className="w-full rounded-2xl bg-sky-500/10 py-3 text-sm font-semibold text-sky-400 transition hover:bg-sky-500/20"
      >
        Scan Another Book
      </button>
    </div>
  );
}
