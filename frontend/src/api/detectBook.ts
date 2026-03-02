import type { DetectionResponse } from "../types";

const STATUS_MESSAGES: Record<number, string> = {
  413: "Image is too large (max 20 MB).",
  415: "Unsupported image format. Use JPEG, PNG, WEBP, or GIF.",
  422: "Could not read this image — the file may be corrupt.",
  500: "Server error — please try again.",
};

export async function detectBook(file: File): Promise<DetectionResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/v1/detect-book", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const message =
        body?.detail ??
        body?.error?.message ??
        STATUS_MESSAGES[response.status] ??
        "Something went wrong.";
      throw new Error(message);
    }

    return await response.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out — the server took too long.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}
