import { useCallback, useEffect, useState } from "react";
import { fetchAchievements, getAccessToken } from "../api.js";
import "./AchievementsWidget.css";

/**
 * Витрина разовых достижений. Данные приходят с сервера, поэтому клиент не
 * решает сам, открыта награда или нет.
 */
export default function AchievementsWidget() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const load = useCallback(async () => {
    if (!getAccessToken()) {
      setLoading(false);
      return;
    }
    try {
      const data = await fetchAchievements();
      setItems(Array.isArray(data) ? data : []);
      setError(false);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      if (cancelled) return;
      await load();
    };

    refresh();
    window.addEventListener("sfedu-game-finished", refresh);
    return () => {
      cancelled = true;
      window.removeEventListener("sfedu-game-finished", refresh);
    };
  }, [load]);

  if (!getAccessToken()) return null;

  if (loading) {
    return (
      <section className="achievements achievements--status" role="status">
        Загружаем достижения…
      </section>
    );
  }

  if (error) {
    return (
      <section className="achievements achievements--status" role="status">
        <p>Не удалось загрузить достижения.</p>
        <button type="button" onClick={load}>
          Повторить
        </button>
      </section>
    );
  }

  const unlockedCount = items.filter((item) => item.unlocked).length;

  return (
    <section className="achievements" aria-labelledby="achievements-title">
      <header className="achievements__heading">
        <div>
          <div className="user-profile-kicker">Награды</div>
          <h2 id="achievements-title">Достижения</h2>
        </div>
        <span className="achievements__count">
          {unlockedCount} из {items.length}
        </span>
      </header>

      {items.length === 0 ? (
        <p className="achievements__empty">
          Сыграйте первую партию, чтобы открыть первую награду.
        </p>
      ) : (
        <div className="achievements__grid">
          {items.map((item) => {
            const reward = item.unlocked ? item.awarded || item.reward : item.reward;
            return (
              <article
                key={item.code}
                className={`achievement-card ${
                  item.unlocked ? "is-unlocked" : "is-locked"
                }`}
                aria-label={`${item.title}: ${
                  item.unlocked ? "получено" : "не открыто"
                }`}
              >
                <span className="achievement-card__icon" aria-hidden="true">
                  {item.icon}
                </span>
                <div className="achievement-card__body">
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </div>
                <div className="achievement-card__footer">
                  <span className="achievement-card__reward">
                    +{reward} GigaCoin
                  </span>
                  <span className="achievement-card__status">
                    {item.unlocked ? "Получено" : "Не открыто"}
                  </span>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
