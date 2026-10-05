import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { fetchAssessmentStatus } from "../api.js";
import "./Assessment.css";

export default function AssessmentNudge({ user }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [status, setStatus] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const load = async () => {
    if (!user || user.role !== "learner") {
      setStatus(null);
      setShowModal(false);
      return;
    }
    try {
      const next = await fetchAssessmentStatus();
      setStatus(next);
      if (next.required && location.pathname !== "/assessment") {
        const key = `sfedu-assessment-modal-dismissed-${user.id}`;
        if (!sessionStorage.getItem(key)) setShowModal(true);
      } else {
        setShowModal(false);
      }
    } catch {
      // Напоминание не должно ломать остальной интерфейс при временном сбое API.
    }
  };

  useEffect(() => { load(); }, [user?.id, user?.role, location.pathname]);

  useEffect(() => {
    const handler = () => load();
    window.addEventListener("sfedu-assessment-updated", handler);
    return () => window.removeEventListener("sfedu-assessment-updated", handler);
  }, [user?.id, user?.role, location.pathname]);

  if (!user || user.role !== "learner" || !status?.required || location.pathname === "/assessment") {
    return null;
  }

  const answered = status.active_test?.answered || 0;
  const total = status.active_test?.total || 20;
  const title = status.phase === "onboarding"
    ? "Определите свой шахматный уровень"
    : status.phase === "feedback"
      ? "Тест завершён — остался короткий отзыв"
      : answered > 0
        ? `Продолжите оценку уровня — ${answered} из ${total}`
        : "Анкета готова — осталось пройти персональный тест";

  const subtitle = status.phase === "onboarding"
    ? "8 коротких вопросов и 20 персональных задач помогут подобрать обучение под ваш уровень."
    : status.phase === "feedback"
      ? "Последний шаг: расскажите, что можно улучшить, или пропустите отзыв и посмотрите итог."
      : `Предварительный диапазон: ${status.rating_group || "определён"}. После задач останется короткий необязательный отзыв.`;

  const go = () => {
    setShowModal(false);
    navigate("/assessment");
  };

  const later = () => {
    sessionStorage.setItem(`sfedu-assessment-modal-dismissed-${user.id}`, "1");
    setShowModal(false);
  };

  return (
    <>
      <div className="assessment-nudge-banner">
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
        <button onClick={go}>{status.phase === "feedback" ? "Завершить" : answered > 0 ? "Продолжить" : "Пройти оценку"}</button>
      </div>

      {showModal && (
        <div className="assessment-modal-backdrop">
          <section className="assessment-modal">
            <div className="assessment-modal-icon">♞</div>
            <div className="assessment-kicker">Персонализация SFEDUCASTLING</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
            <div className="assessment-modal-actions">
              <button className="assessment-primary" onClick={go}>{status.phase === "feedback" ? "Завершить" : answered > 0 ? "Продолжить" : "Пройти сейчас"}</button>
              <button className="assessment-secondary" onClick={later}>Позже</button>
            </div>
            <small>Если отложить, напоминание останется под верхней панелью и появится снова в следующей сессии.</small>
          </section>
        </div>
      )}
    </>
  );
}
