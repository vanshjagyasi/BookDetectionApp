import { useCallback, useRef, useState } from "react";
import { detectBook } from "../api/detectBook";
import type { DetectionResponse } from "../types";

type Status = "idle" | "loading" | "success" | "error";

export function useBookDetection() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const urlRef = useRef<string | null>(null);

  const detect = useCallback(async (file: File) => {
    // Revoke previous preview
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);

    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setPreviewUrl(url);
    setStatus("loading");
    setResult(null);
    setError(null);

    try {
      const res = await detectBook(file);
      setResult(res);
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, previewUrl, detect, reset };
}
