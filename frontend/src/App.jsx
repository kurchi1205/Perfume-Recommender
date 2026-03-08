import { useState, useRef } from "react";
import PerfumeCard from "./components/PerfumeCard";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("image");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const fileInputRef = useRef();

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
    setLoading(true);

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
      const data = await res.json();
      setResults(data.recommendations || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">✦ Scentmatch</h1>
        <p className="subtitle">Upload a mood image or describe how you feel — we'll find your perfect scent.</p>
      </header>

      <div className="workspace">
        {/* Left panel — input */}
        <aside className="panel-left">
          <form onSubmit={handleSubmit} style={{ display: "contents" }}>
            <div className="toggle">
              <button
                type="button"
                className={`toggle-btn ${mode === "image" ? "active" : ""}`}
                onClick={() => setMode("image")}
              >
                🖼 Image
              </button>
              <button
                type="button"
                className={`toggle-btn ${mode === "text" ? "active" : ""}`}
                onClick={() => setMode("text")}
              >
                ✍ Text
              </button>
            </div>

            {mode === "image" && (
              <div className="drop-zone" onClick={() => fileInputRef.current.click()}>
                {imagePreview ? (
                  <img src={imagePreview} alt="preview" className="preview-img" />
                ) : (
                  <span className="drop-hint">
                    <span className="drop-hint-icon">📷</span>
                    Click to upload a mood image
                  </span>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  hidden
                />
              </div>
            )}

            {mode === "text" && (
              <textarea
                className="text-input"
                placeholder="e.g. a rainy autumn evening, cozy and warm..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
              />
            )}

            {error && <p className="error">{error}</p>}

            <button className="submit-btn" type="submit" disabled={loading}>
              {loading ? "Finding your scent..." : "✦ Find My Scent"}
            </button>
          </form>
        </aside>

        {/* Right panel — results */}
        <section className="panel-right">
          {loading ? (
            <div className="loading-state">
              <div className="spinner" />
              <p>Analyzing your mood...</p>
            </div>
          ) : results.length > 0 ? (
            <>
              <h2 className="results-title">Your Top Picks</h2>
              <div className="cards-grid">
                {results.map((p, i) => (
                  <PerfumeCard key={p.perfume_id ?? i} perfume={p} />
                ))}
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
