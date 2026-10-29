import { useEffect, useMemo, useState } from "react";
import {
  fetchLinkedChessAccounts,
  fetchOAuthProviders,
  linkChessAccount,
  startChessOAuth,
} from "../api.js";
import ChessAccountVerificationPanel from "./ChessAccountVerificationPanel.jsx";
import "./ChessAccountsSection.css";

const PROVIDERS = [
  { id: "lichess", label: "Lichess", icon: "♞" },
  { id: "chesscom", label: "Chess.com", icon: "♟" },
];

function errorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;
  return (typeof detail === "string" && detail) || error?.message || fallback;
}

export default function ChessAccountsSection({ user, embedded = false }) {
  const [accounts, setAccounts] = useState([]);
  const [capabilities, setCapabilities] = useState(null);
  const [inputs, setInputs] = useState({ lichess: "", chesscom: "" });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const byProvider = useMemo(
    () => Object.fromEntries(accounts.map((item) => [item.platform, item])),
    [accounts],
  );

  const verifiedCount = useMemo(
    () => accounts.filter((item) => item.verification_status === "verified").length,
    [accounts],
  );

  const reload = async () => {
    const data = await fetchLinkedChessAccounts();
    setAccounts(data?.items || []);
  };

  useEffect(() => {
    if (!user) return;
    let cancelled = false;

    Promise.allSettled([fetchLinkedChessAccounts(), fetchOAuthProviders()]).then(
      ([linked, providers]) => {
        if (cancelled) return;
        if (linked.status === "fulfilled") setAccounts(linked.value?.items || []);
        if (providers.status === "fulfilled") setCapabilities(providers.value);
      },
    );

    return () => {
      cancelled = true;
    };
  }, [user]);

  if (!user) return null;

  const linkPublic = async (provider) => {
    const username = inputs[provider].trim();
    if (!username || busy) return;
    setBusy(`link:${provider}`);
    setError("");
    try {
      await linkChessAccount(username, provider);
      setInputs((prev) => ({ ...prev, [provider]: "" }));
      await reload();
    } catch (e) {
      setError(errorMessage(e, "Не удалось добавить шахматный профиль"));
    } finally {
      setBusy("");
    }
  };

  const linkOAuth = async (provider) => {
    if (busy) return;
    setBusy(`oauth:${provider}`);
    setError("");
    try {
      await startChessOAuth(
        provider,
        "link",
        `${window.location.pathname}${window.location.search}`,
      );
      await reload();
    } catch (e) {
      setError(errorMessage(e, "OAuth-привязка не завершена"));
    } finally {
      setBusy("");
    }
  };

  return (
    <section
      className={`chess-accounts-section ${
        embedded ? "chess-accounts-section--embedded" : ""
      }`}
    >
      <div className="chess-accounts-section__head">
        <div className="chess-accounts-section__heading-main">
          <div className="chess-accounts-section__icon" aria-hidden="true">
            ♞
          </div>
          <div>
            <div className="chess-accounts-section__kicker">Шахматные платформы</div>
            <h2>Шахматные аккаунты</h2>
            <p>
              Добавьте публичный профиль или подтвердите владение через OAuth.
              Подтверждённый рейтинг можно использовать для стартовой оценки.
            </p>
          </div>
        </div>

        {accounts.length > 0 && (
          <div className="chess-accounts-section__summary" aria-label="Статус аккаунтов">
            <strong>{accounts.length}</strong>
            <span>
              {verifiedCount > 0
                ? `подключено · ${verifiedCount} подтверждено`
                : "подключено"}
            </span>
          </div>
        )}
      </div>

      {error && (
        <div className="chess-accounts-section__error" role="status">
          <span className="chess-accounts-section__error-icon" aria-hidden="true">
            !
          </span>
          <span>{error}</span>
          <button
            type="button"
            aria-label="Закрыть сообщение"
            onClick={() => setError("")}
          >
            ×
          </button>
        </div>
      )}

      <div className="chess-accounts-section__list">
        {PROVIDERS.map(({ id: provider, label, icon }) => {
          const account = byProvider[provider];

          if (account) {
            return (
              <ChessAccountVerificationPanel
                key={provider}
                account={account}
                oauthProviders={capabilities}
                profileMode={embedded}
                onChanged={async () => {
                  setError("");
                  await reload();
                }}
                onError={setError}
              />
            );
          }

          const oauthAvailable =
            provider === "lichess" || capabilities?.[provider]?.oauth === true;

          return (
            <div className="chess-account-add" key={provider}>
              <div className="chess-account-add__provider">
                <div
                  className={`chess-account-add__mark chess-account-add__mark--${provider}`}
                  aria-hidden="true"
                >
                  {icon}
                </div>
                <div className="chess-account-add__provider-text">
                  <strong>{label}</strong>
                  <span>Не подключён</span>
                </div>
              </div>

              <div className="chess-account-add__controls">
                <div className="chess-account-add__row">
                  <input
                    value={inputs[provider]}
                    placeholder={`Ник на ${label}`}
                    maxLength={64}
                    aria-label={`Ник на ${label}`}
                    onChange={(e) =>
                      setInputs((prev) => ({ ...prev, [provider]: e.target.value }))
                    }
                    onKeyDown={(e) => e.key === "Enter" && linkPublic(provider)}
                  />
                  <button
                    type="button"
                    className="chess-account-add__link-btn"
                    disabled={Boolean(busy) || !inputs[provider].trim()}
                    onClick={() => linkPublic(provider)}
                  >
                    {busy === `link:${provider}` ? "Добавляем…" : "Добавить"}
                  </button>
                </div>

                {oauthAvailable ? (
                  <button
                    type="button"
                    className="chess-account-add__oauth"
                    disabled={Boolean(busy)}
                    onClick={() => linkOAuth(provider)}
                  >
                    <span aria-hidden="true">↗</span>
                    {busy === `oauth:${provider}`
                      ? "Открываем авторизацию…"
                      : `Подключить и подтвердить через ${label}`}
                  </button>
                ) : (
                  <div className="chess-account-add__note">
                    <span aria-hidden="true">i</span>
                    OAuth {label} пока не настроен. Профиль, добавленный по нику,
                    останется неподтверждённым.
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
