import { useState, useRef, useEffect } from "react";
import PerfumeCard from "./components/PerfumeCard";
import "./App.css";

const CLOUDS = [
  { w: 500, h: 340, left: "5%",  top: "5%",  color: "rgba(180,218,255,0.55)", dur: "22s", tx: "40px",  ty: "-20px" },
  { w: 420, h: 300, left: "60%", top: "0%",  color: "rgba(255,192,220,0.45)", dur: "28s", tx: "-30px", ty: "25px"  },
  { w: 360, h: 260, left: "30%", top: "55%", color: "rgba(210,190,255,0.4)",  dur: "19s", tx: "25px",  ty: "-30px" },
  { w: 300, h: 220, left: "75%", top: "60%", color: "rgba(255,210,230,0.4)",  dur: "25s", tx: "-20px", ty: "20px"  },
  { w: 280, h: 200, left: "15%", top: "75%", color: "rgba(190,225,255,0.45)", dur: "32s", tx: "35px",  ty: "-15px" },
];

const SPARKLES = Array.from({ length: 55 }, (_, i) => ({
  id: i,
  left: `${Math.random() * 100}%`,
  top: `${Math.random() * 100}%`,
  size: `${Math.random() * 3 + 1}px`,
  dur: `${Math.random() * 4 + 2}s`,
  delay: `${Math.random() * 6}s`,
  op: (Math.random() * 0.55 + 0.25).toFixed(2),
}));

function SparkleBackground() {
  return (
    <div className="cloud-bg">
      {CLOUDS.map((c, i) => (
        <div key={i} className="cloud" style={{
          width: c.w, height: c.h, left: c.left, top: c.top,
          background: c.color, "--dur": c.dur, "--tx": c.tx, "--ty": c.ty,
          "--delay": `${i * 1.5}s`,
        }} />
      ))}
      {SPARKLES.map((s) => (
        <div key={s.id} className="sparkle" style={{
          left: s.left, top: s.top, width: s.size, height: s.size,
          "--dur": s.dur, "--delay": s.delay, "--op": s.op,
        }} />
      ))}
    </div>
  );
}

const STEPS = [
  "Extracting mood from your input…",
  "Identifying scent accords…",
  "Searching fragrance database…",
  "Ranking your perfect matches…",
];

export default function App() {
  const [mode, setMode] = useState("image");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [results, setResults] = useState([]);
  const [moods, setMoods] = useState([]);
  const [accords, setAccords] = useState([]);
  const [error, setError] = useState("");
  const fileInputRef = useRef();

  // Cycle through loading steps
  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => setLoadingStep((s) => Math.min(s + 1, STEPS.length - 1)), 4000);
    return () => clearInterval(t);
  }, [loading]);

  function handleImageChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResults([]);
    setMoods([]);
    setAccords([]);
    setLoading(true);
    setLoadingStep(0);

    try {
      const form = new FormData();
      form.append("input_type", mode);
      form.append("user_id", "anonymous");

      if (mode === "image") {
        if (!imageFile) { setError("Please upload an image."); setLoading(false); return; }
        form.append("image", imageFile);
      } else {
        if (!text.trim()) { setError("Please enter a mood description."); setLoading(false); return; }
        form.append("text", text.trim());
      }

      const res = await fetch("/recommend", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          const payload = JSON.parse(part.slice(6));
          if (payload.type === "moods")   setMoods(payload.moods);
          if (payload.type === "accords") setAccords(payload.accords);
          if (payload.type === "result")  setResults(payload.recommendations);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <SparkleBackground />

      <header className="header">
        <h1 className="title">✦ Scentmatch</h1>
        <p className="subtitle">Upload a mood image or describe how you feel — we'll find your perfect scent.</p>
      </header>

      <div className="workspace">
        {/* ── Left panel ── */}
        <aside className="panel-left">
          <form onSubmit={handleSubmit} style={{ display: "contents" }}>
            <div className="toggle">
              <button type="button" className={`toggle-btn ${mode === "image" ? "active" : ""}`} onClick={() => setMode("image")}>
                🖼 Image
              </button>
              <button type="button" className={`toggle-btn ${mode === "text" ? "active" : ""}`} onClick={() => setMode("text")}>
                ✍ Text
              </button>
            </div>

            {mode === "image" && (
              <div className="drop-zone" onClick={() => fileInputRef.current.click()}>
                {imagePreview
                  ? <img src={imagePreview} alt="preview" className="preview-img" />
                  : <span className="drop-hint"><span className="drop-hint-icon">🌙</span>Click to upload a mood image</span>
                }
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageChange} hidden />
              </div>
            )}

            {mode === "text" && (
              <textarea
                className="text-input"
                placeholder="e.g. a misty forest at dawn, dewy and ethereal…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
              />
            )}

            {/* Streamed moods */}
            {moods.length > 0 && (
              <div className="mood-strip">
                <span className="mood-strip-label">✦ Moods detected</span>
                {moods.map((m) => <span key={m} className="mood-tag">{m}</span>)}
              </div>
            )}

            {/* Streamed accords */}
            {accords.length > 0 && (
              <div className="mood-strip">
                <span className="mood-strip-label">✦ Scent accords</span>
                {accords.map((a) => <span key={a} className="mood-tag">{a}</span>)}
              </div>
            )}

            {error && <p className="error">{error}</p>}

            <button className="submit-btn" type="submit" disabled={loading}>
              {loading ? "Searching the stars…" : "✦ Find My Scent"}
            </button>
          </form>
        </aside>

        {/* ── Right panel ── */}
        <section className="panel-right">
          {loading ? (
            <div className="loading-state">
              <div className="loading-orb" />
              <div className="loading-steps">
                {STEPS.map((s, i) => (
                  <p key={s} className={`loading-step ${i === loadingStep ? "active" : ""}`}>{s}</p>
                ))}
              </div>
            </div>
          ) : results.length > 0 ? (
            <>
              <h2 className="results-title">Your Top Picks ✦</h2>
              <div className="cards-grid">
                {results.map((p, i) => <PerfumeCard key={p.perfume_id ?? i} perfume={p} />)}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">🌸</span>
              <p>Your recommendations will appear here</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
