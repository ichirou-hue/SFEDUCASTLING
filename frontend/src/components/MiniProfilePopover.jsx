import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchCurrentUser,
  fetchLearningProgress,
  fetchTrainingProgress,
} from "../api.js";
import "./MiniProfilePopover.css";

function safeNumber(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

export default function MiniProfilePopover({ user, onUserChange }) {
  const [open, setOpen] = useState(false);
  const [profile, setProfile] = useState(user || null);
  const [training, setTraining] = useState(null);
  const [learning, setLearning] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [dirty, setDirty] = useState(true);

  useEffect(() => {
    setProfile(user || null);
  }, [user]);

  const loadPreview = useCallback(async () => {
    if (loading) return;

    try {
      setLoading(true);
      setError("");

      const [freshUser, trainingData, learningData] = await Promise.all([
        fetchCurrentUser(),
        fetchTrainingProgress(),
        fetchLearningProgress(),
      ]);

      setProfile(freshUser);
      setTraining(trainingData);
      setLearning(learningData);
      setLoaded(true);
      setDirty(false);
      onUserChange?.(freshUser);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Не удалось загрузить прогресс.",
      );
    } finally {
      setLoading(false);
    }
  }, [loading, onUserChange]);

  useEffect(() => {
    if (open && (!loaded || dirty)) {
      loadPreview();
    }
  }, [open, loaded, dirty, loadPreview]);

  useEffect(() => {
    const markProgressDirty = () => {
      if (open) {
        setDirty(true);
      } else {
        setDirty(true);
      }
    };

    window.addEventListener("sfedu-progress-updated", markProgressDirty);
    return () =>
      window.removeEventListener("sfedu-progress-updated", markProgressDirty);
  }, [open]);

  const modules = training?.modules || {};
  const courseCompleted = safeNumber(modules.completed);
  const courseTotal = safeNumber(modules.total);
  const solvedPuzzles = safeNumber(learning?.solved);
  const streak = safeNumber(training?.streak);

  const handleBlur = (event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setOpen(false);
    }
  };

  return (
    <div
      className="mini-profile"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={handleBlur}
    >
      <Link
        to="/profile"
        className="user-name mini-profile__trigger"
        title="Открыть профиль и прогресс"
      >
        {profile?.is_admin ? "♛ " : "👤 "}
        {profile?.login || user?.login || "Пользователь"}
      </Link>

      <div
        className={`mini-profile__popover ${open ? "mini-profile__popover--open" : ""}`}
        role="status"
        aria-hidden={!open}
      >
        <div className="mini-profile__arrow" aria-hidden="true" />

        <div className="mini-profile__heading">
          <div className="mini-profile__avatar" aria-hidden="true">
            {profile?.is_admin ? "♛" : "♟"}
          </div>
          <div className="mini-profile__identity">
            <strong>{profile?.login || user?.login || "Пользователь"}</strong>
            <span>Elo {profile?.elo ?? "—"}</span>
          </div>
        </div>

        {loading && !loaded ? (
          <div className="mini-profile__message">Загружаем прогресс…</div>
        ) : error && !loaded ? (
          <div className="mini-profile__message mini-profile__message--error">
            {error}
          </div>
        ) : (
          <>
            <div className="mini-profile__stats">
              <div className="mini-profile__row">
                <span>Курс</span>
                <strong>
                  {courseCompleted}/{courseTotal}
                </strong>
              </div>
              <div className="mini-profile__row">
                <span>Пазлы</span>
                <strong>{solvedPuzzles}</strong>
              </div>
              <div className="mini-profile__row mini-profile__row--streak">
                <span>Серия</span>
                <strong>🔥 {streak} дн.</strong>
              </div>
            </div>

            {error && (
              <div className="mini-profile__message mini-profile__message--warning">
                Данные могут быть неактуальны.
              </div>
            )}

            <Link to="/profile" className="mini-profile__details-link">
              Открыть полный профиль →
            </Link>
          </>
        )}
      </div>
    </div>
  );
}
