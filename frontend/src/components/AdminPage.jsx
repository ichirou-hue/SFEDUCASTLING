import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchAdminUser,
  fetchAdminUserStats,
  fetchAdminUsers,
  updateUserRole,
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

function isRecent(value, days) {
  if (!value) return false;
  const ts = new Date(value).getTime();
  if (Number.isNaN(ts)) return false;
  return Date.now() - ts <= days * 24 * 60 * 60 * 1000;
}

function Stat({ label, value }) {
  return (
    <div className="admin-stat">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

export default function AdminPage({ user, onUserChange }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [sortBy, setSortBy] = useState("id");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [roleSaving, setRoleSaving] = useState(false);
  const [confirmRoleId, setConfirmRoleId] = useState(null);

  const currentUserId = user?.id;

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await fetchAdminUsers();
      setUsers(data.users || []);
    } catch (err) {
      const status = err?.response?.status;
      setError(
        status === 401 || status === 403
          ? "Недостаточно прав для просмотра пользователей."
          : err?.response?.data?.detail ||
              err?.message ||
              "Не удалось загрузить список пользователей.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  // Дашборд считается на клиенте из списка пользователей — отдельный эндпоинт не нужен.
  const dashboard = useMemo(() => {
    const total = users.length;
    const admins = users.filter((u) => u.role === "admin").length;
    const learners = total - admins;
    const new7 = users.filter((u) => isRecent(u.created_at, 7)).length;
    const new30 = users.filter((u) => isRecent(u.created_at, 30)).length;
    const withElo = users.filter((u) => u.elo != null);
    const avgElo = withElo.length
      ? Math.round(withElo.reduce((sum, u) => sum + Number(u.elo), 0) / withElo.length)
      : null;
    return { total, admins, learners, new7, new30, avgElo };
  }, [users]);

  const visibleUsers = useMemo(() => {
    const query = search.trim().toLowerCase();
    let list = users;
    if (query) {
      list = list.filter((u) => u.login.toLowerCase().includes(query));
    }
    if (roleFilter !== "all") {
      list = list.filter((u) => u.role === roleFilter);
    }
    const sorted = [...list];
    sorted.sort((a, b) => {
      switch (sortBy) {
        case "login":
          return a.login.localeCompare(b.login, "ru");
        case "created_desc":
          return String(b.created_at || "").localeCompare(String(a.created_at || ""));
        case "elo_desc":
          return (b.elo ?? -1) - (a.elo ?? -1);
        default:
          return a.id - b.id;
      }
    });
    return sorted;
  }, [users, search, roleFilter, sortBy]);

  const loadDetail = useCallback(async (userId) => {
    setDetail(null);
    setDetailError("");
    setDetailLoading(true);
    try {
      const [profileData, statsData] = await Promise.all([
        fetchAdminUser(userId),
        fetchAdminUserStats(userId),
      ]);
      setDetail({
        ...statsData,
        user: statsData?.user ?? profileData?.user ?? null,
      });
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
    setConfirmRoleId(null);
    loadDetail(userId);
  };

  const changeRole = async (target, role) => {
    setRoleSaving(true);
    setNotice("");
    setError("");
    try {
      const data = await updateUserRole(target.id, role);
      const updated = data?.user;
      if (updated) {
        setUsers((prev) =>
          prev.map((u) => (u.id === updated.id ? { ...u, ...updated } : u)),
        );
        setDetail((prev) =>
          prev?.user?.id === updated.id
            ? { ...prev, user: { ...prev.user, ...updated } }
            : prev,
        );
        // Сменили роль сами себе — обновляем глобальный user в app.jsx.
        if (updated.id === currentUserId) {
          onUserChange?.(updated);
        }
      }
      setNotice(`Роль пользователя «${target.login}» обновлена: ${roleLabel(role)}.`);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Не удалось изменить роль пользователя.",
      );
    } finally {
      setRoleSaving(false);
      setConfirmRoleId(null);
    }
  };

  const training = detail?.training;
  const learning = detail?.learning;
  const weaknesses = detail?.weaknesses;
  const detailUser = detail?.user;
  const detailIsSelf = detailUser && detailUser.id === currentUserId;

  return (
    <main className="admin-page">
      <section className="admin-heading">
        <div>
          <div className="admin-kicker">Администрирование</div>
          <h1>Пользователи и прогресс обучения</h1>
        </div>
        <p>Дашборд, управление ролями и статистика каждого ученика.</p>
      </section>

      <section className="admin-summary" aria-label="Сводка по пользователям">
        <Stat label="Всего пользователей" value={dashboard.total} />
        <Stat label="Администраторы" value={dashboard.admins} />
        <Stat label="Ученики" value={dashboard.learners} />
        <Stat label="Новых за 7 дней" value={dashboard.new7} />
        <Stat label="Новых за 30 дней" value={dashboard.new30} />
        <Stat
          label="Средний Elo"
          value={dashboard.avgElo != null ? dashboard.avgElo : "—"}
        />
      </section>

      {error && <div className="admin-warning">{error}</div>}
      {notice && <div className="admin-notice admin-notice--success">{notice}</div>}

      {loading ? (
        <section className="admin-empty">Загружаем список пользователей…</section>
      ) : users.length === 0 ? (
        <section className="admin-empty">Пользователей пока нет.</section>
      ) : (
        <div className="admin-layout">
          <aside className="admin-users-panel">
            <div className="admin-toolbar">
              <input
                className="admin-search"
                type="search"
                placeholder="Поиск по логину…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Поиск по логину"
              />
              <select
                className="admin-select"
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
                aria-label="Фильтр по роли"
              >
                <option value="all">Все роли</option>
                <option value="admin">Администраторы</option>
                <option value="learner">Ученики</option>
              </select>
              <select
                className="admin-select"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                aria-label="Сортировка"
              >
                <option value="id">По ID</option>
                <option value="login">По логину</option>
                <option value="created_desc">Сначала новые</option>
                <option value="elo_desc">По Elo</option>
              </select>
              <button
                type="button"
                className="admin-btn"
                onClick={loadUsers}
                disabled={loading}
              >
                Обновить
              </button>
            </div>

            <div className="admin-count">
              Показано {visibleUsers.length} из {users.length}
            </div>

            {visibleUsers.length === 0 ? (
              <p className="admin-muted admin-panel-hint">
                Ничего не найдено — измените поиск или фильтр.
              </p>
            ) : (
              visibleUsers.map((u) => (
                <button
                  key={u.id}
                  type="button"
                  className={`admin-user ${
                    selectedId === u.id ? "admin-user--active" : ""
                  }`}
                  onClick={() => selectUser(u.id)}
                >
                  <span className="admin-user__avatar" aria-hidden="true">
                    {u.is_admin || u.role === "admin" ? "♛" : "♟"}
                  </span>
                  <span className="admin-user__main">
                    <strong>{u.login}</strong>
                    <span className="admin-user__sub">{roleLabel(u.role)}</span>
                  </span>
                  <span className="admin-user__meta">
                    {u.skill_band != null ? `гр. ${u.skill_band}` : "гр. —"}
                  </span>
                </button>
              ))
            )}
          </aside>

          <div className="admin-detail">
            {detailLoading && <div className="admin-empty">Загружаем статистику…</div>}
            {detailError && <div className="admin-warning">{detailError}</div>}

            {!detailLoading && !detail && !detailError && (
              <section className="admin-empty">
                Выберите пользователя слева, чтобы увидеть профиль и статистику.
              </section>
            )}

            {!detailLoading && detailUser && !detailError && (
              <>
                <section className="admin-card">
                  <div className="admin-user-profile-head">
                    <div className="admin-user-profile-avatar" aria-hidden="true">
                      {detailUser.role === "admin" ? "♛" : "♟"}
                    </div>
                    <div className="admin-user-profile-main">
                      <div className="admin-user-profile-heading">
                        <div>
                          <div className="admin-kicker">Профиль пользователя</div>
                          <h2>{detailUser.login}</h2>
                        </div>
                        {detailUser.role === "admin" && (
                          <span className="admin-badge">Администратор</span>
                        )}
                      </div>
                      <div className="admin-stats">
                        <Stat label="Роль" value={roleLabel(detailUser.role)} />
                        <Stat label="Шахматный уровень" value={detailUser.elo} />
                        <Stat label="Группа навыка" value={detailUser.skill_band} />
                        <Stat label="До онбординга" value={detailUser.prior_band} />
                        <Stat label="Оценка рейтинга" value={detailUser.rating_estimate} />
                        <Stat label="Email" value={detailUser.email} />
                        <Stat label="В системе с" value={formatDate(detailUser.created_at)} />
                        {detailUser.rating_scale && (
                          <Stat label="Шкала рейтинга" value={detailUser.rating_scale} />
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <section className="admin-card">
                  <h3>Управление ролью</h3>
                  {detailIsSelf ? (
                    <p className="admin-muted">
                      Это ваш аккаунт: снять роль администратора у самого себя нельзя.
                    </p>
                  ) : (
                    <div className="admin-role-row">
                      <p className="admin-muted">
                        Текущая роль: <strong>{roleLabel(detailUser.role)}</strong>
                      </p>
                      <div className="admin-role-actions">
                        {detailUser.role === "admin" ? (
                          confirmRoleId === detailUser.id ? (
                            <>
                              <span className="admin-muted">
                                Снять роль администратора?
                              </span>
                              <button
                                type="button"
                                className="admin-btn admin-btn--danger"
                                disabled={roleSaving}
                                onClick={() => changeRole(detailUser, "learner")}
                              >
                                {roleSaving ? "Сохраняем…" : "Да, снять"}
                              </button>
                              <button
                                type="button"
                                className="admin-btn"
                                disabled={roleSaving}
                                onClick={() => setConfirmRoleId(null)}
                              >
                                Отмена
                              </button>
                            </>
                          ) : (
                            <button
                              type="button"
                              className="admin-btn admin-btn--danger"
                              onClick={() => setConfirmRoleId(detailUser.id)}
                            >
                              Снять администратора
                            </button>
                          )
                        ) : (
                          <button
                            type="button"
                            className="admin-btn admin-btn--primary"
                            disabled={roleSaving}
                            onClick={() => changeRole(detailUser, "admin")}
                          >
                            {roleSaving ? "Сохраняем…" : "Сделать администратором"}
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </section>

                <div className="admin-tracks">
                  <section className="admin-card">
                    <h3>Учебный курс</h3>
                    <div className="admin-stats">
                      <Stat
                        label="Завершено модулей"
                        value={`${training?.modules?.completed ?? 0} / ${training?.modules?.total ?? 0}`}
                      />
                      <Stat label="Прогресс" value={`${training?.modules?.percent ?? 0}%`} />
                      <Stat
                        label="Точность"
                        value={
                          training?.attempts?.accuracy != null
                            ? `${training.attempts.accuracy}%`
                            : "—"
                        }
                      />
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
                      <Stat
                        label="Точность"
                        value={
                          learning?.accuracy != null ? `${learning.accuracy}%` : "—"
                        }
                      />
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
                              <span
                                key={t.slug}
                                className="admin-topic-chip admin-topic-chip--weak"
                              >
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
