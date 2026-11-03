"use strict";

const MAX_INPUT_LENGTH = 500;

function validatePredictionPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return "Prediction payload must be an object.";
  }
  if (typeof payload.text !== "string" || !payload.text.trim()) {
    return "Enter at least one word before requesting a prediction.";
  }
  if (payload.text.length > MAX_INPUT_LENGTH) {
    return `Sentence must be ${MAX_INPUT_LENGTH} characters or fewer.`;
  }
  if (payload.requestId != null && typeof payload.requestId !== "string") {
    return "requestId must be a string.";
  }
  return null;
}

function registerPredictionHandlers({ socket, predictionService }) {
  socket.on("prediction:request", async (payload) => {
    const validationError = validatePredictionPayload(payload);
    if (validationError) {
      socket.emit("prediction:error", {
        requestId: payload?.requestId ?? null,
        message: validationError,
      });
      return;
    }

    try {
      const result = await predictionService.predict(payload.text.trim());
      // socket.emit targets only the user who requested the prediction.
      socket.emit("prediction:result", {
        requestId: payload.requestId ?? null,
        ...result,
      });
    } catch (error) {
      console.error("Prediction failed:", error.message);
      socket.emit("prediction:error", {
        requestId: payload.requestId ?? null,
        message: error.message || "Prediction service is unavailable.",
      });
    }
  });
}

module.exports = { registerPredictionHandlers, validatePredictionPayload };

