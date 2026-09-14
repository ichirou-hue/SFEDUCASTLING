import { useEffect, useState } from "react";
import { fetchCurrentUser } from "../api.js";
import ProgressWidget from "./ProgressWidget.jsx";
import "./UserProfilePage.css";

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  }).format(date);
}

export default function UserProfilePage({ user, onUserChange }) {
  const [profile, setProfile] = useState(user || null);
  const [loading, setLoading] = useState(Boolean(user));
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      try {
        setLoading(true);
        setError("");
        const freshUser = await fetchCurrentUser();
        if (cancelled) return;
        setProfile(freshUser);
        onUserChange?.(freshUser);
      } catch (err) {
        if (cancelled) return;
        setError(
          err?.response?.data?.detail ||
            err?.message ||
            "Не удалось обновить данные профиля.",
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadProfile();
    return () => {
      cancelled = true;
    };
  }, [onUserChange]);

  if (!profile && !loading) {
    return (
      <main className="user-profile-page">
        <section className="user-profile-empty">
          Войдите в аккаунт, чтобы открыть профиль и прогресс обучения.
        </section>
      </main>
    );
  }

  return (
    <main className="user-profile-page">
      <section className="user-profile-card">
        <div className="user-profile-avatar" aria-hidden="true">
          {profile?.is_admin ? "♛" : "♟"}
        </div>

        <div className="user-profile-main">
          <div className="user-profile-heading">
            <div>
              <div className="user-profile-kicker">Профиль пользователя</div>
              <h1>{profile?.login || "Пользователь"}</h1>
            </div>
            {profile?.is_admin && (
              <span className="user-profile-badge">Администратор</span>
            )}
          </div>

          <div className="user-profile-stats">
            <div className="user-profile-stat">
              <span>Шахматный уровень</span>
              <strong>{profile?.elo ?? "Не определён"}</strong>
            </div>
            <div className="user-profile-stat">
              <span>Email</span>
              <strong>{profile?.email || "Не указан"}</strong>
            </div>
            <div className="user-profile-stat">
              <span>В системе с</span>
              <strong>{formatDate(profile?.created_at)}</strong>
            </div>
          </div>

          {error && <div className="user-profile-warning">{error}</div>}
        </div>
      </section>

      <section className="user-profile-progress-section">
        <div className="user-profile-section-heading">
          <div>
            <div className="user-profile-kicker">Обучение</div>
            <h2>Прогресс</h2>
          </div>
          <p>
            Здесь объединены оба учебных трека: курс и шахматные задачи.
          </p>
        </div>

        <ProgressWidget />
      </section>
    </main>
  );
}
