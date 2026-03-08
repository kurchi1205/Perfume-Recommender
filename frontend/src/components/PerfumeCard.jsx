import "./PerfumeCard.css";

export default function PerfumeCard({ perfume }) {
  const { name, brand, image_url, main_accords = [], url } = perfume;

  return (
    <div className="card">
      <div className="card-img-wrap">
        {image_url ? (
          <img src={image_url} alt={name} className="card-img" />
        ) : (
          <div className="card-img-placeholder">🌸</div>
        )}
      </div>

      <div className="card-body">
        <p className="card-brand">{brand}</p>
        <h3 className="card-name">{name}</h3>

        {main_accords.length > 0 && (
          <div className="card-accords">
            {main_accords.slice(0, 5).map((accord) => (
              <span key={accord} className="accord-tag">
                {accord}
              </span>
            ))}
          </div>
        )}

        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="card-link"
          >
            View on Fragrantica →
          </a>
        )}
      </div>
    </div>
  );
}
