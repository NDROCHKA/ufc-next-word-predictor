"use strict";

// Composition root: infrastructure is created here, while prediction logic and
// socket event handling live in separate modules that can be tested in isolation.
const http = require("node:http");
const express = require("express");
const { Server } = require("socket.io");

const { createPredictionService } = require("./server/services/predictionService");
const { registerPredictionHandlers } = require("./server/socket/registerPredictionHandlers");

const PORT = Number(process.env.PORT || 3000);
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8000";

// Browsers treat localhost and 127.0.0.1 as different origins. Support both
// local development URLs so the Socket.IO handshake works whichever one the
// user opens. Production deployments can provide a comma-separated list.
const CLIENT_ORIGINS = (
  process.env.CLIENT_ORIGINS ||
  process.env.CLIENT_ORIGIN ||
  "http://localhost:5173,http://127.0.0.1:5173"
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const app = express();
const httpServer = http.createServer(app);
const io = new Server(httpServer, {
  cors: { origin: CLIENT_ORIGINS },
});
const predictionService = createPredictionService({ baseUrl: MODEL_SERVICE_URL });

app.use(express.json());

app.get("/api/health", async (_request, response) => {
  const modelService = await predictionService.health();
  response.status(modelService.ok ? 200 : 503).json({
    gateway: "ready",
    modelService,
  });
});

io.on("connection", (socket) => {
  console.log(`Socket connected: ${socket.id}`);
  registerPredictionHandlers({ socket, predictionService });

  socket.on("disconnect", (reason) => {
    console.log(`Socket disconnected: ${socket.id} (${reason})`);
  });
});

httpServer.listen(PORT, () => {
  console.log(`Express + Socket.IO listening on http://localhost:${PORT}`);
  console.log(`Model service expected at ${MODEL_SERVICE_URL}`);
});

