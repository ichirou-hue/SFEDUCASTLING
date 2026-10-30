import { useState } from "react";
import {
  fetchChessVerificationHistory,
  fetchLinkedChessAccounts,
  syncChessAccount,
  unlinkChessAccount,
  verifyChessAccount,
} from "../api.js";
import "./ChessAccountVerificationPanel.css";

const PLATFORM_LABELS = {
  lichess: "Lichess",
  chesscom: "Chess.com",
};

const PLATFORM_ICONS = {
  lichess: "♞",
  chesscom: "♟",
};

const STATUS = {
  linked: { label: "Не подтверждён", tone: "neutral" },
  pending: { label: "Проверка начата", tone: "pending" },
  verified: { label: "Подтверждён", tone: "verified" },
  expired: { label: "Проверка истекла", tone: "warning" },
  revoked: { label: "Отозван", tone: "warning" },
  error: { label: "Ошибка проверки", tone: "error" },
};

function dateTime(value) {
  if (!value) return "—";
  try {
    return new Intl.DateTimeFormat("ru-RU", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function errorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;
  return (typeof detail === "string" && detail) || error?.message || fallback;
}

export default function ChessAccountVerificationPanel({
  account,
  oauthProviders,
  onChanged,
  onError,
  compact = false,
  profileMode = false,
}) {
  const [busy, setBusy] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [localError, setLocalError] = useState("");

  if (!account) return null;

  const platform = account.platform;
  const providerName = PLATFORM_LABELS[platform] || platform;
  const providerIcon = PLATFORM_ICONS[platform] || "♟";
  const state = STATUS[account.verification_status] || STATUS.linked;
  const verificationAvailable =
    platform === "lichess" || oauthProviders?.[platform]?.verification === true;

  const fail = (error, fallback) => {
    const message = errorMessage(error, fallback);
    setLocalError(message);
    onError?.(message);
  };

  const handleVerify = async () => {
    if (busy) return;
    setBusy("verify");
    setLocalError("");
    try {
      await verifyChessAccount(
        platform,
        `${window.location.pathname}${window.location.search}`,
      );
      const linked = await fetchLinkedChessAccounts();
      const refreshed = linked?.items?.find((item) => item.platform === platform);
      onChanged?.(refreshed || account);
      if (historyOpen) await loadHistory(true);
    } catch (error) {
      fail(error, "Не удалось подтвердить владение аккаунтом");
    } finally {
      setBusy("");
    }
  };

  const handleSync = async () => {
    if (busy) return;
    setBusy("sync");
    setLocalError("");
    try {
      const result = await syncChessAccount(platform);
      onChanged?.(result?.account || account);
    } catch (error) {
      fail(error, "Не удалось синхронизировать профиль");
    } finally {
      setBusy("");
    }
  };

  const handleUnlink = async () => {
    if (busy) return;
    const ok = window.confirm(
      `Отключить ${providerName} ${account.username} от GigaChess?`,
    );
    if (!ok) return;
    setBusy("unlink");
    setLocalError("");
    try {
      await unlinkChessAccount(platform);
      onChanged?.(null);
    } catch (error) {
      fail(error, "Не удалось отключить аккаунт");
    } finally {
      setBusy("");
    }
  };

  const loadHistory = async (force = false) => {
    if (historyLoading || (history.length && !force)) return;
    setHistoryLoading(true);
    try {
      const data = await fetchChessVerificationHistory(platform, 20);
      setHistory(data?.items || []);
    } catch (error) {
      fail(error, "Не удалось загрузить историю проверок");
    } finally {
      setHistoryLoading(false);
    }
  };

  const toggleHistory = async () => {
    const next = !historyOpen;
    setHistoryOpen(next);
    if (next) await loadHistory();
  };

  const verificationLabel =
    account.verification_status === "verified"
      ? profileMode
        ? "Проверить снова"
        : "Подтвердить повторно"
      : profileMode
        ? "Подтвердить"
        : `Подтвердить через ${providerName}`;

  return (
    <div
      className={`chess-account-card ${
        compact ? "chess-account-card--compact" : ""
      } ${profileMode ? "chess-account-card--profile" : ""}`}
    >
      <div className="chess-account-card__head">
        <div className="chess-account-card__identity">
          <div
            className={`chess-account-card__platform-icon chess-account-card__platform-icon--${platform}`}
            aria-hidden="true"
          >
            {providerIcon}
          </div>
          <div>
            <div className="chess-account-card__provider">{providerName}</div>
            <div className="chess-account-card__username">{account.username}</div>
          </div>
        </div>

        <span className={`chess-account-badge chess-account-badge--${state.tone}`}>
          {account.verification_status === "verified" ? "✓ " : ""}
          {state.label}
        </span>
      </div>

      <div className="chess-account-card__summary-row">
        <div className="chess-account-card__rating">
          {account.rating != null ? (
            <>
              <strong>{account.rating}</strong>
              <span>{account.rating_type || "рейтинг"}</span>
              <span className="chess-account-card__dot">•</span>
              <span>{account.games || 0} партий</span>
            </>
          ) : (
            <span>Подходящий рейтинг не найден</span>
          )}
        </div>

        <div className="chess-account-trust">
          <span className={account.rating_trusted ? "is-trusted" : "is-untrusted"}>
            {account.rating_trusted
              ? "✓ Рейтинг подтверждён"
              : account.rating_usable
                ? "Рейтинг ждёт подтверждения"
                : "Рейтинг пока не используется"}
          </span>
        </div>
      </div>

      {account.verification_failure_reason &&
        account.verification_status !== "verified" && (
          <div className="chess-account-card__warning">
            {account.verification_failure_reason}
          </div>
        )}

      <div className="chess-account-card__meta">
        {account.verified_at && (
          <span>Подтверждён: {dateTime(account.verified_at)}</span>
        )}
        {account.last_verified_at &&
          account.last_verified_at !== account.verified_at && (
            <span>Последняя проверка: {dateTime(account.last_verified_at)}</span>
          )}
        {account.updated_at && <span>Обновлён: {dateTime(account.updated_at)}</span>}
      </div>

      <div className="chess-account-card__actions">
        {verificationAvailable && (
          <button
            type="button"
            className="chess-account-action chess-account-action--primary"
            onClick={handleVerify}
            disabled={Boolean(busy)}
          >
            {busy === "verify" ? "Проверяем…" : verificationLabel}
          </button>
        )}

        <button
          type="button"
          className="chess-account-action"
          onClick={handleSync}
          disabled={Boolean(busy)}
        >
          {busy === "sync" ? "Обновляем…" : profileMode ? "Обновить" : "Синхронизировать"}
        </button>

        <button
          type="button"
          className="chess-account-action chess-account-action--quiet"
          onClick={toggleHistory}
          disabled={historyLoading}
        >
          {historyLoading
            ? "Загрузка…"
            : historyOpen
              ? "Скрыть историю"
              : profileMode
                ? "История"
                : "История проверок"}
        </button>

        <button
          type="button"
          className="chess-account-action chess-account-action--danger"
          onClick={handleUnlink}
          disabled={Boolean(busy)}
        >
          {busy === "unlink" ? "Отключаем…" : "Отключить"}
        </button>
      </div>

      {!verificationAvailable && account.verification_status !== "verified" && (
        <div className="chess-account-card__note">
          OAuth-подтверждение {providerName} пока недоступно. Публичный профиль
          подключён, но его рейтинг не считается доверенным.
        </div>
      )}

      {localError && <div className="chess-account-card__error">{localError}</div>}

      {historyOpen && (
        <div className="chess-account-history">
          <div className="chess-account-history__title">История подтверждений</div>
          {!history.length && !historyLoading && (
            <div className="chess-account-history__empty">Проверок ещё не было.</div>
          )}
          {history.map((item) => {
            const itemState = STATUS[item.status] || STATUS.linked;
            return (
              <div className="chess-account-history__item" key={item.id}>
                <span className={`chess-account-history__status is-${itemState.tone}`}>
                  {itemState.label}
                </span>
                <span>{item.method === "oauth" ? "OAuth" : item.method}</span>
                <span>{dateTime(item.completed_at || item.started_at)}</span>
                {item.failure_reason && (
                  <span className="chess-account-history__reason">{item.failure_reason}</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
