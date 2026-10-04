import { useCallback, useEffect, useState } from "react";
import {
  fetchAdminUser,
  fetchAdminUserStats,
  fetchAdminUsers,
} from "../api.js";
import "./AdminPage.css";

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

function roleLabel(role) {
  return role === "admin" ? "Администратор" : "Ученик";
}

function Stat({ label, value }) {
  return (
    <div className="admin-stat">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

export default function AdminPage() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setError("");
        const data = await fetchAdminUsers();
        if (cancelled) return;
        setUsers(data.users || []);
      } catch (err) {
        if (cancelled) return;
        setError(
          err?.response?.data?.detail ||
            err?.message ||
            "Не удалось загрузить список пользователей.",
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadDetail = useCallback(async (userId) => {
    setDetail(null);
    setDetailError("");
    setDetailLoading(true);
    try {
      const [profileData, statsData] = await Promise.all([
        fetchAdminUser(userId),
        fetchAdminUserStats(userId),
      ]);
      setDetail({ ...profileData, ...statsData });
    } catch (err) {
      setDetailError(
        err?.response?.data?.detail ||
          err?.message ||
          "Не удалось загрузить статистику пользователя.",
      );
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const selectUser = (userId) => {
    setSelectedId(userId);
    loadDetail(userId);
  };

  const selectedUsers = detail?.user
    ? users.map((u) => (u.id === detail.user.id ? { ...u, ...detail.user } : u))
    : users;

  const training = detail?.training;
  const learning = detail?.learning;
  const weaknesses = detail?.weaknesses;

  return (
    <main className="admin-page">
      <section className="admin-heading">
        <div>
          <div className="admin-kicker">Администрирование</div>
          <h1>Пользователи и прогресс обучения</h1>
        </div>
        <p>Выберите пользователя, чтобы увидеть его профиль и статистику.</p>
      </section>

      {error && <div className="admin-warning">{error}</div>}

      {loading ? (
        <section className="admin-empty">Загружаем список пользователей…</section>
      ) : users.length === 0 ? (
        <section className="admin-empty">Пользователей пока нет.</section>
      ) : (
        <div className="admin-layout">
          <aside className="admin-users-panel">
            {selectedUsers.map((u) => (
              <button
                key={u.id}
                type="button"
                className={`admin-user ${
                  selectedId === u.id ? "admin-user--active" : ""
                }`}
                onClick={() => selectUser(u.id)}
              >
                <span className="admin-user__avatar" aria-hidden="true">
                  {u.is_admin ? "♛" : "♟"}
                </span>
                <span className="admin-user__main">
                  <strong>{u.login}</strong>
                  <span className="admin-user__sub">{roleLabel(u.role)}</span>
                </span>
                <span className="admin-user__meta">
                  {u.skill_band != null ? `гр. ${u.skill_band}` : "гр. —"}
                </span>
              </button>
            ))}
          </aside>

          <div className="admin-detail">
            {detailLoading && <div className="admin-empty">Загружаем статистику…</div>}
            {detailError && <div className="admin-warning">{detailError}</div>}

            {!detailLoading && detail && !detailError && (
              <>
                <section className="admin-card">
                  <div className="admin-user-profile-head">
                    <div className="admin-user-profile-avatar" aria-hidden="true">
                      {detail.user.is_admin ? "♛" : "♟"}
                    </div>
                    <div className="admin-user-profile-main">
                      <div className="admin-user-profile-heading">
                        <div>
                          <div className="admin-kicker">Профиль пользователя</div>
                          <h2>{detail.user.login}</h2>
                        </div>
                        {detail.user.is_admin && (
                          <span className="admin-badge">Администратор</span>
                        )}
                      </div>
                      <div className="admin-stats">
                        <Stat label="Роль" value={roleLabel(detail.user.role)} />
                        <Stat label="Шахматный уровень" value={detail.user.elo} />
                        <Stat label="Группа навыка" value={detail.user.skill_band} />
                        <Stat label="До онбординга" value={detail.user.prior_band} />
                        <Stat label="Оценка рейтинга" value={detail.user.rating_estimate} />
                        <Stat label="Email" value={detail.user.email} />
                        <Stat label="В системе с" value={formatDate(detail.user.created_at)} />
                        {detail.user.rating_scale && (
                          <Stat label="Шкала рейтинга" value={detail.user.rating_scale} />
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <div className="admin-tracks">
                  <section className="admin-card">
                    <h3>Учебный курс</h3>
                    <div className="admin-stats">
                      <Stat label="Завершено модулей" value={`${training?.modules?.completed ?? 0} / ${training?.modules?.total ?? 0}`} />
                      <Stat label="Прогресс" value={`${training?.modules?.percent ?? 0}%`} />
                      <Stat label="Точность" value={training?.attempts?.accuracy != null ? `${training.attempts.accuracy}%` : "—"} />
                      <Stat label="Попыток" value={training?.attempts?.total ?? 0} />
                      <Stat label="Верных" value={training?.attempts?.correct ?? 0} />
                      <Stat label="Серия дней" value={training?.streak ?? 0} />
                    </div>
                  </section>

                  <section className="admin-card">
                    <h3>Тактические паззлы</h3>
                    <div className="admin-stats">
                      <Stat label="Попыток" value={learning?.attempts ?? 0} />
                      <Stat label="Верных" value={learning?.correct_attempts ?? 0} />
                      <Stat label="Точность" value={learning?.accuracy != null ? `${learning.accuracy}%` : "—"} />
                      <Stat label="Паззлов решено" value={learning?.solved ?? 0} />
                      <Stat label="Паззлов взято" value={learning?.attempted ?? 0} />
                    </div>
                  </section>
                </div>

                {weaknesses && (
                  <section className="admin-card">
                    <h3>Сильные и слабые темы</h3>
                    {weaknesses.weak_topics?.length || weaknesses.strong_topics?.length ? (
                      <div className="admin-topics">
                        <div className="admin-topics-group">
                          <span className="admin-topics-title admin-topics-title--weak">
                            Слабые
                          </span>
                          <div className="admin-topic-chips">
                            {(weaknesses.weak_topics || []).map((t) => (
                              <span key={t.slug} className="admin-topic-chip admin-topic-chip--weak">
                                {t.title}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="admin-topics-group">
                          <span className="admin-topics-title">Сильные</span>
                          <div className="admin-topic-chips">
                            {(weaknesses.strong_topics || []).map((t) => (
                              <span key={t.slug} className="admin-topic-chip">
                                {t.title}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="admin-muted">
                        Пока недостаточно попыток для оценки тем.
                      </p>
                    )}
                  </section>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}