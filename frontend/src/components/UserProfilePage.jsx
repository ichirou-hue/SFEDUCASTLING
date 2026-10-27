import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchCurrentUser,
  fetchLinkedChessAccounts,
  getAccessToken,
  linkChessAccount,
  unlinkChessAccount,
} from "../api.js";
import ProgressWidget from "./ProgressWidget.jsx";
import RatingBandProgress from "./RatingBandProgress.jsx";
import UserRecordsWidget from "./UserRecordsWidget.jsx";
import PhaseAccuracyWidget from "./PhaseAccuracyWidget.jsx";
import SiteFooter from "./SiteFooter.jsx";
import GigacoinShop from "./GigacoinShop.jsx";
import "./UserProfilePage.css";

function formatDate(value) {
  if (!value) return "0";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "0";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  }).format(date);
}

function platformLabel(platform) {
  return platform === "chesscom" ? "Chess.com" : "Lichess";
}

/*
 * Привязка публичного шахматного аккаунта (soft link, без OAuth).
 * Бэкенд сам подтягивает рейтинг и/games/ профиля при привязке.
 */
function LinkedAccountsSection() {
  const [accounts, setAccounts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState("");
  const [platform, setPlatform] = useState("lichess");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);

  const load = useCallback(async () => {
    // Без токена эндпоинт вернёт 401 — не дёргаем его для гостей.
    if (!getAccessToken()) {
      setAccounts([]);
      setLoading(false);
      return;
    }
    try {
      const data = await fetchLinkedChessAccounts();
      setAccounts(data.items || []);
    } catch {
      // Секция не должна ломать профиль при временном сбое API.
      setAccounts([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleLink = async (event) => {
    event.preventDefault();
    const name = username.trim();
    if (!name || busy) return;
    setBusy(true);
    setNotice(null);
    try {
      const data = await linkChessAccount(name, platform);
      setAccounts((prev) => {
        const next = prev.filter((item) => item.platform !== data.account.platform);
        return [...next, data.account];
      });
      setUsername("");
      setNotice({ type: "ok", text: `Аккаунт ${data.account.username} привязан.` });
    } catch (err) {
      setNotice({
        type: "error",
        text:
          err?.response?.data?.detail ||
          err?.message ||
          "Не удалось привязать аккаунт. Проверьте ник и попробуйте снова.",
      });
    } finally {
      setBusy(false);
    }
  };

  const handleUnlink = async (acc) => {
    if (busy) return;
    setBusy(true);
    setNotice(null);
    try {
      await unlinkChessAccount(acc.platform);
      setAccounts((prev) => prev.filter((item) => item.platform !== acc.platform));
      setNotice({ type: "ok", text: `Аккаунт ${acc.username} отвязан.` });
    } catch (err) {
      setNotice({
        type: "error",
        text: err?.response?.data?.detail || err?.message || "Не удалось отвязать аккаунт.",
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="user-profile-section">
      <div className="user-profile-section-heading">
        <div>
          <div className="user-profile-kicker">Шахматный профиль</div>
          <h2>Привязанные аккаунты</h2>
        </div>
        <p>Привяжите публичный профиль Lichess или Chess.com.</p>
      </div>

      {notice && (
        <div
          className={`user-profile-warning ${
            notice.type === "ok" ? "user-profile-warning--ok" : ""
          }`}
        >
          {notice.text}
        </div>
      )}

      {loading ? (
        <p className="user-profile-muted">Загружаем список аккаунтов…</p>
      ) : accounts.length > 0 ? (
        <div className="user-profile-accounts">
          {accounts.map((acc) => (
            <div className="user-profile-account" key={`${acc.platform}-${acc.username}`}>
              <div className="user-profile-account-main">
                <strong>{acc.username}</strong>
                <span>
                  {platformLabel(acc.platform)}
                  {acc.rating != null ? ` · ${acc.rating_type || "рейтинг"} ${acc.rating}` : ""}
                  {!acc.rating_usable ? " · рейтинг уточняется" : ""}
                </span>
              </div>
              <button
                type="button"
                className="user-profile-account-remove"
                onClick={() => handleUnlink(acc)}
                disabled={busy}
              >
                Отвязать
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="user-profile-muted">Аккаунты пока не привязаны.</p>
      )}

      <form className="user-profile-account-form" onSubmit={handleLink}>
        <select
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
          aria-label="Платформа"
        >
          <option value="lichess">Lichess</option>
          <option value="chesscom">Chess.com</option>
        </select>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Имя пользователя"
          aria-label="Имя пользователя на платформе"
        />
        <button type="submit" disabled={busy || !username.trim()}>
          {busy ? "Сохраняем…" : "Привязать"}
        </button>
      </form>
    </section>
  );
}

export default function UserProfilePage({ user, onUserChange }) {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(user || null);
  const [loading, setLoading] = useState(Boolean(user));
  const [error, setError] = useState("");

  useEffect(() => {
    if (user) setProfile(user);
  }, [user]);

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
      <section className={`user-profile-card user-profile-card--accent-${profile?.equipped_profile_accent || "default"}`}>
        <div className={`user-profile-avatar user-profile-avatar--${profile?.equipped_avatar_frame || "default"}`} aria-hidden="true">
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
            <div className="user-profile-stat user-profile-stat--coins">
              <span>Баланс GigaCoin</span>
              <strong>{Number(profile?.gigacoins || 0).toLocaleString("ru-RU")}</strong>
            </div>
          </div>

          <div className="user-profile-actions">
            {/* Кнопка нужна только ученикам, которые ещё не проходили
                проверку уровня: после submit бэкенд ставит
                assessment_completed_at, и public() отдаёт
                assessment_completed: true - кнопка прячется. Администраторам
                стартовая оценка не требуется (бэкенд отвечает 403), поэтому
                им кнопку не показываем вообще. Во время загрузки свежих
                данных кнопку тоже не показываем, чтобы она не мигала тем,
                кто тест уже прошёл (начальный profile из кэша). */}
            {!loading &&
              !profile?.assessment_completed &&
              profile?.role !== "admin" && (
              <button
                type="button"
                className="user-profile-level-test-btn"
                onClick={() => navigate("/level-test")}
              >
                Пройти проверку уровня
              </button>
            )}
          </div>

          {error && <div className="user-profile-warning">{error}</div>}
        </div>
      </section>

      <GigacoinShop onUserChange={(updated) => {
        setProfile(updated);
        onUserChange?.(updated);
      }} />

      <UserRecordsWidget />
      <PhaseAccuracyWidget />

      <section className="user-profile-rating-section">
        <div className="user-profile-section-heading">
          <div>
            <div className="user-profile-kicker">Ваш результат</div>
            <h2>Прогресс уровня</h2>
          </div>
          <p>Путь от текущего уровня к следующему по рейтингу.</p>
        </div>
        <RatingBandProgress userId={profile?.id ?? user?.id} />
      </section>

      <LinkedAccountsSection />

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
      <SiteFooter />
    </main>
  );
}
