"use strict";

const { io } = require("../client/node_modules/socket.io-client");

const socket = io("http://127.0.0.1:3000", {
  transports: ["websocket"],
  reconnection: false,
  // Match the browser origin used by the documented local application URL.
  extraHeaders: { Origin: "http://127.0.0.1:5173" },
});

const timeout = setTimeout(() => {
  console.error("Timed out waiting for a Socket.IO prediction.");
  socket.disconnect();
  process.exit(1);
}, 15_000);

socket.on("connect", () => {
  socket.emit("prediction:request", {
    text: "khabib",
    requestId: "e2e-check",
  });
});

socket.on("prediction:result", (result) => {
  clearTimeout(timeout);
  console.log(JSON.stringify(result, null, 2));
  const bothCorrect =
    result.predictions?.birnn?.word === "nurmagomedov" &&
    result.predictions?.textcnn?.word === "nurmagomedov";
  socket.disconnect();
  process.exit(bothCorrect ? 0 : 1);
});

socket.on("prediction:error", (error) => {
  clearTimeout(timeout);
  console.error(error);
  socket.disconnect();
  process.exit(1);
});

