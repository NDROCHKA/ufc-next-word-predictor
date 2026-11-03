"use strict";

/**
 * HTTP adapter around the Python inference service.
 * Express never knows how PyTorch tokenizes text or constructs a model.
 */
function createPredictionService({ baseUrl, fetchImpl = globalThis.fetch }) {
  if (typeof fetchImpl !== "function") {
    throw new Error("A Fetch implementation is required.");
  }

  async function request(path, options = {}) {
    const response = await fetchImpl(`${baseUrl}${path}`, {
      ...options,
      headers: {
        "content-type": "application/json",
        ...options.headers,
      },
      // A stuck model service must not leave the browser waiting forever.
      signal: AbortSignal.timeout(10_000),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = body.detail || `Model service returned HTTP ${response.status}.`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  return {
    predict(text) {
      return request("/predict", {
        method: "POST",
        body: JSON.stringify({ text }),
      });
    },

    async health() {
      try {
        const details = await request("/health");
        return { ok: details.status === "ready", ...details };
      } catch (error) {
        return { ok: false, status: "unavailable", message: error.message };
      }
    },
  };
}

module.exports = { createPredictionService };

