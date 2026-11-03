"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { validatePredictionPayload } = require("../server/socket/registerPredictionHandlers");

test("prediction payload requires non-empty text", () => {
  assert.match(validatePredictionPayload({ text: "   " }), /at least one word/i);
});

test("valid prediction payload is accepted", () => {
  assert.equal(
    validatePredictionPayload({ text: "khabib nurmagomedov", requestId: "one" }),
    null,
  );
});

