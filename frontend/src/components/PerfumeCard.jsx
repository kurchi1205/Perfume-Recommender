import { useState, useEffect } from "react";
import "./PerfumeCard.css";

const PEXELS_KEY = import.meta.env.VITE_PEXELS_KEY;

function extractId(perfumePageUrl) {
  return perfumePageUrl?.match(/-(\d+)\.html$/)?.[1] || null;
}

function deriveImageUrl(perfumePageUrl) {
  const id = extractId(perfumePageUrl);
  return id ? `https://www.fragrantica.com/mdimg/perfume-social-cards/en-p_c_${id}.jpeg` : null;
}

function fimgsUrl(perfumePageUrl) {
  const id = extractId(perfumePageUrl);
  return id ? `https://fimgs.net/mdimg/perfume/social.${id}.jpg` : null;
}

async function fetchPexelsImage(query) {
  if (!PEXELS_KEY) return null;
  try {
    const res = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=1&orientation=square`,
      { headers: { Authorization: PEXELS_KEY } }
    );
    const data = await res.json();
    return data.photos?.[0]?.src?.medium || null;
  } catch {
    return null;
  }
}

export default function PerfumeCard({ perfume }) {
  const { name, brand, image_url, main_accords = [], url } = perfume;
  const derived = image_url || deriveImageUrl(url);
  const [imgSrc, setImgSrc] = useState(derived || null);
  const [step, setStep] = useState(0); // 0=primary, 1=fimgs, 2=pexels, 3=placeholder

  async function handleImgError() {
    if (step === 0) {
      setStep(1);
      setImgSrc(fimgsUrl(url));
    } else if (step === 1) {
      setStep(2);
      const accord = main_accords.length
        ? main_accords[Math.floor(Math.random() * main_accords.length)]
        : "perfume";
      const pexelsUrl = await fetchPexelsImage(`${accord} perfume bottle`);
      setImgSrc(pexelsUrl);
    } else {
      setStep(3);
      setImgSrc(null);
    }
  }

  return (
    <div className="card">
      <div className="card-img-wrap">
        {imgSrc ? (
          <img src={imgSrc} alt={name} className="card-img" onError={handleImgError} />
        ) : (
          <div className="card-img-placeholder">
            <span>🌸</span>
            <span>No image</span>
          </div>
        )}
      </div>

      <div className="card-body">
        <p className="card-brand">{brand}</p>
        <h3 className="card-name">{name}</h3>

        {main_accords.length > 0 && (
          <div className="card-accords">
            {main_accords.slice(0, 5).map((accord) => (
              <span key={accord} className="accord-tag">{accord}</span>
            ))}
          </div>
        )}

        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer" className="card-link">
            View on Fragrantica →
          </a>
        )}
      </div>
    </div>
  );
}
