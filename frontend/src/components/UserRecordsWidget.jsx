import { useEffect, useState } from "react";
import { fetchLearningRecords, getAccessToken } from "../api.js";
import "./UserRecordsWidget.css";

const CHART_MODES = {
  puzzles: {
    label: "Задачи",
    total: (day) => day.correct + day.incorrect,
    segments: [
      { key: "correct", label: "Решено верно", color: "#4f8b65" },
      { key: "incorrect", label: "Ошибки", color: "#d5a65b" },
    ],
  },
  games: {
    label: "Партии",
    total: (day) => day.wins + day.draws + day.losses,
    segments: [
      { key: "wins", label: "Победы", color: "#4f8b65" },
      { key: "draws", label: "Ничьи", color: "#d5a65b" },
      { key: "losses", label: "Поражения", color: "#bd7168" },
    ],
  },
};

function formatDay(value) {
  const date = new Date(`${value}T12:00:00`);
  return new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short" })
    .format(date)
    .replace(".", "");
}

function formatRange(start, end) {
  const startDate = new Date(`${start}T12:00:00`);
  const endDate = new Date(`${end}T12:00:00`);
  const month = new Intl.DateTimeFormat("ru-RU", { month: "short" })
    .format(endDate)
    .replace(".", "");
  if (startDate.getMonth() === endDate.getMonth()) {
    return `${startDate.getDate()}–${endDate.getDate()} ${month}`;
  }
  return `${formatDay(start)} – ${formatDay(end)}`;
}

function dayNumber(value) {
  return new Date(`${value}T12:00:00`).getDate();
}

function pluralize(value, forms) {
  const category = new Intl.PluralRules("ru").select(value);
  return `${value} ${forms[category] || forms.other}`;
}

function MetricCard({ icon, label, value, detail, accent }) {
  return (
    <article className={`user-record-card user-record-card--${accent}`}>
      <div className="user-record-card__top">
        <span className="user-record-card__icon" aria-hidden="true">{icon}</span>
        <span className="user-record-card__label">{label}</span>
      </div>
      <strong className="user-record-card__value">{value}</strong>
      <span className="user-record-card__detail">{detail}</span>
    </article>
  );
}

function ActivityChart({ activity, mode }) {
  const config = CHART_MODES[mode];
  const rows = activity?.[mode] || [];
  const maxTotal = Math.max(1, ...rows.map(config.total));
  const total = rows.reduce((sum, row) => sum + config.total(row), 0);

  return (
    <div className="user-record-chart" role="img" aria-label={`Активность: ${config.label.toLowerCase()} за последние 14 дней`}>
      <div className="user-record-chart__plot">
        {total === 0 && (
          <div className="user-record-chart__empty">
            Здесь появится твоя динамика
          </div>
        )}
        {rows.map((row) => {
          const dayTotal = config.total(row);
          const title = `${formatDay(row.date)} · ${config.segments
            .map((segment) => `${segment.label.toLowerCase()}: ${row[segment.key]}`)
            .join(" · ")}`;
          return (
            <div className="user-record-chart__column" key={row.date} title={title}>
              <span className="user-record-chart__count">
                {dayTotal > 0 ? dayTotal : ""}
              </span>
              <div className="user-record-chart__track">
                <div
                  className={`user-record-chart__bar${dayTotal ? " user-record-chart__bar--active" : ""}`}
                  style={{ height: `${Math.max(dayTotal ? 5 : 2, (dayTotal / maxTotal) * 100)}%` }}
                >
                  {config.segments.map((segment) => {
                    const amount = row[segment.key];
                    if (!amount || !dayTotal) return null;
                    return (
                      <span
                        key={segment.key}
                        className="user-record-chart__segment"
                        style={{
                          height: `${(amount / dayTotal) * 100}%`,
                          backgroundColor: segment.color,
                        }}
                      />
                    );
                  })}
                </div>
              </div>
              <span className="user-record-chart__date">{dayNumber(row.date)}</span>
            </div>
          );
        })}
      </div>
      <div className="user-record-chart__footer">
        <div className="user-record-legend">
          {config.segments.map((segment) => (
            <span key={segment.key}>
              <i style={{ backgroundColor: segment.color }} />{segment.label}
            </span>
          ))}
        </div>
        <strong>{pluralize(total, mode === "games"
          ? { one: "партия", few: "партии", many: "партий", other: "партии" }
          : { one: "задача", few: "задачи", many: "задач", other: "задачи" })}</strong>
      </div>
    </div>
  );
}

export default function UserRecordsWidget() {
  const [data, setData] = useState(null);
  const [mode, setMode] = useState("puzzles");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    if (!getAccessToken()) {
      setLoading(false);
      return undefined;
    }
    fetchLearningRecords()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <section className="user-records user-records--loading">Загружаем рекорды…</section>;
  }
  if (!data) {
    return <section className="user-records user-records--loading">Не удалось загрузить рекорды. Попробуйте обновить страницу.</section>;
  }

  const records = data.records;
  return (
    <section className="user-records" aria-labelledby="user-records-title">
      <header className="user-records__heading">
        <div>
          <div className="user-profile-kicker">Твои достижения</div>
          <h2 id="user-records-title">Личные рекорды</h2>
        </div>
        <span className="user-records__range">
          {formatRange(data.activity.starts_on, data.activity.ends_on)}
        </span>
      </header>

      <div className="user-records__metrics">
        <MetricCard
          icon="♛"
          label="Лучший результат недели"
          value={records.weekly_best_opponent_elo != null ? `${records.weekly_best_opponent_elo}` : "—"}
          detail={records.weekly_best_opponent_elo != null ? "ELO побеждённого бота" : "победи бота, чтобы установить рекорд"}
          accent="gold"
        />
        <MetricCard
          icon="♜"
          label="Победная серия"
          value={records.longest_win_streak}
          detail={records.longest_win_streak ? "партий подряд · личный максимум" : "победы появятся после первой партии"}
          accent="green"
        />
        <MetricCard
          icon="♞"
          label="Серия задач"
          value={records.best_puzzle_streak}
          detail={records.best_puzzle_streak ? "решено подряд · личный максимум" : "реши задачу, чтобы начать серию"}
          accent="blue"
        />
      </div>

      <div className="user-records__activity">
        <div className="user-records__activity-heading">
          <div>
            <h3>Твой ритм</h3>
            <p>Занятия день за днём</p>
          </div>
          <div className="user-records__tabs" role="tablist" aria-label="Тип активности">
            {Object.entries(CHART_MODES).map(([key, config]) => (
              <button
                type="button"
                role="tab"
                aria-selected={mode === key}
                className={mode === key ? "is-active" : ""}
                key={key}
                onClick={() => setMode(key)}
              >
                {config.label}
              </button>
            ))}
          </div>
        </div>
        <ActivityChart activity={data.activity} mode={mode} />
      </div>
    </section>
  );
}
