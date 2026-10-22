import { useEffect, useState } from "react";
import { fetchRatingBandProgress, getAccessToken } from "../api.js";
import "./RatingBandProgress.css";

function ratingPoints(value) {
  const category = new Intl.PluralRules("ru").select(value);
  const word = category === "one" ? "пункт" : category === "few" ? "пункта" : "пунктов";
  return `${value.toLocaleString("ru-RU")} ${word}`;
}

export default function RatingBandProgress({ compact = false, userId = null }) {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const load = () => {
      if (!getAccessToken()) {
        setProgress(null);
        return;
      }
      fetchRatingBandProgress()
        .then((data) => {
          if (!cancelled) setProgress(data);
        })
        .catch(() => {
          if (!cancelled) setProgress(null);
        });
    };

    load();
    window.addEventListener("sfedu-assessment-updated", load);
    return () => {
      cancelled = true;
      window.removeEventListener("sfedu-assessment-updated", load);
    };
  }, [userId]);

  if (!progress?.current_band) return null;

  const current = progress.current_band;
  const next = progress.next_band;
  const label = next
    ? `Прогресс от уровня «${current.title}» до уровня «${next.title}»`
    : "Достигнут максимальный рейтинговый уровень";

  return (
    <section
      className={`rating-band-progress${compact ? " rating-band-progress--compact" : ""}`}
      aria-label="Прогресс рейтингового уровня"
    >
      <div className="rating-band-progress__header">
        <div className="rating-band-progress__title">
          <span>Ваш уровень</span>
          <strong>{current.title}</strong>
        </div>
        <div className="rating-band-progress__rating">
          {progress.rating_available ? progress.rating.toLocaleString("ru-RU") : "—"}
          <small>{progress.provisional ? "предварительно" : "рейтинг"}</small>
        </div>
      </div>

      <div
        className="rating-band-progress__track"
        role="progressbar"
        aria-valuenow={Math.round(progress.progress_percent)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <span style={{ width: `${progress.progress_percent}%` }} />
      </div>

      <div className="rating-band-progress__caption">
        {!progress.rating_available ? (
          "Пройдите тест уровня, чтобы уточнить прогресс"
        ) : progress.is_max_band ? (
          "Достигнут максимальный рейтинговый уровень"
        ) : (
          <>
            До уровня «{next.title}»: осталось {ratingPoints(progress.points_to_next)}
          </>
        )}
      </div>
    </section>
  );
}
