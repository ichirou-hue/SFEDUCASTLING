import { useCallback, useEffect, useState } from "react";
import {
  equipGigacoinCosmetic,
  fetchGigacoinShop,
  getAccessToken,
  purchaseGigacoinCosmetic,
} from "../api.js";
import "./GigacoinShop.css";

const SLOT_TITLES = {
  avatar_frame: "Рамки аватара",
  profile_accent: "Акцент профиля",
};

function friendlyError(err, fallback) {
  const status = err?.response?.status;
  if (status === 404) {
    return "Текущий backend ещё не содержит API магазина. Обнови сервер до версии с GigaCoin и перезапусти его.";
  }
  if (status === 500) {
    return "Backend не смог прочитать магазин. Проверь, что применена миграция базы данных: alembic upgrade head.";
  }
  if (status === 401) return "Сессия истекла. Войди в аккаунт заново.";
  if (status === 409 || status === 403) {
    return err?.response?.data?.detail || fallback;
  }
  return fallback;
}

export default function GigacoinShop({ onUserChange }) {
  const [shop, setShop] = useState(null);
  const [busyId, setBusyId] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    if (!getAccessToken()) return;
    try {
      setShop(await fetchGigacoinShop());
      setError("");
    } catch (err) {
      setError(friendlyError(err, "Не удалось загрузить каталог магазина. Проверь подключение и попробуй ещё раз."));
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleAction = async (item) => {
    if (busyId) return;
    setBusyId(item.id);
    setNotice("");
    setError("");
    try {
      if (item.owned) {
        const result = await equipGigacoinCosmetic(item.id);
        onUserChange?.(result.user);
        setNotice(`${item.title}: теперь используется.`);
      } else {
        const result = await purchaseGigacoinCosmetic(item.id);
        onUserChange?.(result.user);
        setNotice(`${item.title} куплен. Его можно надеть в любой момент.`);
      }
      await load();
    } catch (err) {
      setError(friendlyError(err, "Не удалось выполнить действие. Попробуй ещё раз."));
    } finally {
      setBusyId("");
    }
  };

  if (!getAccessToken()) return null;

  const groups = Object.entries(SLOT_TITLES).map(([slot, title]) => ({
    slot,
    title,
    items: (shop?.items || []).filter((item) => item.slot === slot),
  }));

  return (
    <section className="gigacoin-shop" aria-labelledby="gigacoin-shop-title">
      <header className="gigacoin-shop__header">
        <div>
          <div className="user-profile-kicker">Косметика без влияния на игру</div>
          <h2 id="gigacoin-shop-title">Магазин GigaCoin</h2>
          <p>Покупки постоянные. Купленные предметы можно бесплатно переключать.</p>
        </div>
        <div className="gigacoin-shop__balance" aria-live="polite">
          <span>Баланс</span><strong>✦ {Number(shop?.balance ?? 0).toLocaleString("ru-RU")}</strong>
        </div>
      </header>

      {notice && <p className="gigacoin-shop__notice" role="status">{notice}</p>}
      {error && (
        <div className="gigacoin-shop__error-box" role="alert">
          <p className="gigacoin-shop__error">{error}</p>
          <button type="button" className="gigacoin-shop__retry" onClick={load}>Повторить</button>
        </div>
      )}
      {!shop && !error && <p className="gigacoin-shop__message">Загружаем предметы…</p>}

      {groups.map((group) => group.items.length > 0 && (
        <div className="gigacoin-shop__group" key={group.slot}>
          <h3>{group.title}</h3>
          <div className="gigacoin-shop__grid">
            {group.items.map((item) => (
              <article className={`gigacoin-shop__item gigacoin-shop__item--${item.id}`} key={item.id}>
                <div className="gigacoin-shop__preview" aria-hidden="true">
                  {group.slot === "avatar_frame" ? <span className="gigacoin-shop__avatar">♟</span> : <span>✦</span>}
                </div>
                <div className="gigacoin-shop__copy">
                  <h4>{item.title}</h4>
                  <p>{item.description}</p>
                  <strong className="gigacoin-shop__price">✦ {item.price}</strong>
                </div>
                <button
                  type="button"
                  onClick={() => handleAction(item)}
                  disabled={Boolean(busyId) || (!item.owned && Number(shop?.balance || 0) < item.price)}
                >
                  {busyId === item.id ? "Подождите…" : item.equipped ? "Надето" : item.owned ? "Надеть" : "Купить"}
                </button>
              </article>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
