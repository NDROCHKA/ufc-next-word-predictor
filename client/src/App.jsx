import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./App.css";

// One client is shared by the component for its entire lifetime. autoConnect is
// disabled so listeners can be registered before the first server event arrives.
// Use the same hostname the user used to open React. This avoids mixing
// localhost and 127.0.0.1, which browsers correctly treat as different origins.
const socketServerUrl =
  import.meta.env.VITE_SOCKET_URL ??
  `${window.location.protocol}//${window.location.hostname}:3000`;
const socket = io(socketServerUrl, { autoConnect: false });
const AUTO_PREDICT_DELAY_MS = 350;

const EXAMPLE_PROMPTS = [
  "khabib",
  "conor mcgregor faced",
  "islam makhachev competed in a",
  "alex pereira versus",
];

function formatScore(value) {
  return `${Math.round(value * 100)}% model score`;
}

function PredictionCard({ label, description, prediction, accent, onSelect }) {
  return (
    <article className={`prediction-card ${accent}`}>
      <div className="model-heading">
        <div>
          <span className="model-label">{label}</span>
          <p>{description}</p>
        </div>
        <span className="score">{formatScore(prediction.confidence)}</span>
      </div>

      <button
        className="word-choice"
        type="button"
        onClick={() => onSelect(prediction.word)}
        aria-label={`Append ${prediction.word}, predicted by ${label}`}
      >
        <span>{prediction.word}</span>
        <span aria-hidden="true">＋</span>
      </button>

      <div className="alternatives" aria-label={`${label} alternatives`}>
        {prediction.alternatives.slice(1).map((choice) => (
          <button key={choice.word} type="button" onClick={() => onSelect(choice.word)}>
            {choice.word}
          </button>
        ))}
      </div>
    </article>
  );
}

function App() {
  const [sentence, setSentence] = useState("");
  const [connectionState, setConnectionState] = useState("connecting");
  const [predictions, setPredictions] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState("");
  // A ref remembers the latest request without causing a re-render. It prevents
  // an older, slower response from replacing a newer prediction.
  const latestRequestId = useRef(null);

  useEffect(() => {
    function handleConnect() {
      setConnectionState("connected");
    }

    function handleDisconnect() {
      setConnectionState("disconnected");
      setIsPredicting(false);
      latestRequestId.current = null;
    }

    function handleResult(result) {
      if (result.requestId !== latestRequestId.current) return;
      setPredictions(result.predictions);
      setIsPredicting(false);
      setError("");
    }

    function handleError(result) {
      if (result.requestId && result.requestId !== latestRequestId.current) return;
      setError(result.message);
      setIsPredicting(false);
    }

    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("prediction:result", handleResult);
    socket.on("prediction:error", handleError);
    socket.connect();

    return () => {
      // Remove the exact callbacks registered above. This avoids duplicated
      // messages if React mounts the component again during development.
      socket.off("connect", handleConnect);
      socket.off("disconnect", handleDisconnect);
      socket.off("prediction:result", handleResult);
      socket.off("prediction:error", handleError);
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    const trimmed = sentence.trim();

    // Changing the text immediately invalidates any response still travelling
    // back from the server. Only the request for the current text may update UI.
    const requestId = crypto.randomUUID();
    latestRequestId.current = requestId;

    if (!trimmed) {
      return undefined;
    }

    if (connectionState !== "connected" || !socket.connected) {
      return undefined;
    }

    // Wait for a brief typing pause. Without this debounce, typing five letters
    // would unnecessarily run both neural networks five times.
    const timer = window.setTimeout(() => {
      setIsPredicting(true);
      setError("");
      socket.emit("prediction:request", { text: trimmed, requestId });
    }, AUTO_PREDICT_DELAY_MS);

    return () => window.clearTimeout(timer);
  }, [sentence, connectionState]);

  function appendWord(word) {
    // Updating the sentence automatically schedules the following prediction,
    // allowing the user to build a sentence one suggested word at a time.
    setSentence((current) => `${current.trim()} ${word}`.trim());
    setPredictions(null);
    setError("");
  }

  function selectExample(prompt) {
    setSentence(prompt);
    setPredictions(null);
    setError("");
  }

  return (
    <main className="app-shell">
      <nav className="topbar">
        <div className="brand-mark" aria-hidden="true">U</div>
        <div className="brand-copy">
          <strong>Octagon Oracle</strong>
          <span>UFC next-word laboratory</span>
        </div>
        <div className={`connection-pill ${connectionState}`}>
          <span className="status-dot" />
          {connectionState}
        </div>
      </nav>

      <section className="hero">
        <p className="eyebrow">CSC 433 · Neural networks</p>
        <h1>Finish the sentence.<br /><em>Compare the minds.</em></h1>
        <p className="intro">
          Two PyTorch models trained on modern UFC bouts suggest the next word.
          Choose the prediction that fits your sentence best.
        </p>
      </section>

      <section className="workspace" aria-label="UFC next-word predictor">
        <div className="prediction-input">
          <label htmlFor="sentence">Your incomplete sentence</label>
          <textarea
            id="sentence"
            value={sentence}
            onChange={(event) => {
              setSentence(event.target.value);
              setPredictions(null);
              setIsPredicting(false);
              setError("");
            }}
            maxLength={500}
            placeholder="Start typing: khabib nurmagomedov faced"
            rows={3}
          />
          <div className="automatic-status" aria-live="polite">
            <span className={isPredicting ? "pulse" : ""} />
            {connectionState !== "connected"
              ? "Waiting for Socket.IO connection"
              : isPredicting
                ? "Both models are predicting…"
                : sentence.trim()
                  ? "Predictions update after you pause typing"
                  : "Type any UFC phrase to begin"}
          </div>
        </div>

        <div className="example-row">
          <span>Try an example</span>
          {EXAMPLE_PROMPTS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => selectExample(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        {error && <div className="error-message" role="alert">{error}</div>}

        {predictions ? (
          <div className="results" aria-live="polite">
            <div className="results-heading">
              <span>Choose a word to append</span>
              <span>Both answers are shown independently</span>
            </div>
            <div className="prediction-grid">
              <PredictionCard
                label="BiRNN"
                description="Reads the typed context in two directions"
                prediction={predictions.birnn}
                accent="amber"
                onSelect={appendWord}
              />
              <PredictionCard
                label="TextCNN"
                description="Finds strong local word patterns"
                prediction={predictions.textcnn}
                accent="red"
                onSelect={appendWord}
              />
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="octagon" aria-hidden="true">VS</div>
            <div>
              <strong>Your two predictions will appear here</strong>
              <p>The displayed score is the model's softmax output, not a guarantee.</p>
            </div>
          </div>
        )}
      </section>

      <footer>
        <span>BiRNN + TextCNN</span>
        <span>React · Socket.IO · Express · PyTorch</span>
        <span>Modern UFC corpus: 2018–2025</span>
      </footer>
    </main>
  );
}

export default App;

