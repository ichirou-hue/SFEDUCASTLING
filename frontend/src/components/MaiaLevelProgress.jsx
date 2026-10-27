import { useEffect, useState } from "react";
import { fetchMaiaAchievements } from "../api.js";

const MAIA_LEVELS = Array.from({ length: 19 }, (_, index) => 800 + index * 100);
const STORAGE_KEY = "sfedu-maia-level-wins-v1";

function readLevels(key) {
  try {
    const stored = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(stored)
      ? stored.filter((level) => MAIA_LEVELS.includes(Number(level))).map(Number)
      : [];
  } catch {
    return [];
  }
}

function mergeLevels(...lists) {
  return [...new Set(lists.flat().map(Number))]
    .filter((level) => MAIA_LEVELS.includes(level))
    .sort((a, b) => a - b);
}

function saveLevels(key, levels) {
  try {
    localStorage.setItem(key, JSON.stringify(levels));
  } catch {
    // Server history remains the source of truth when browser storage is unavailable.
  }
}

export default function MaiaLevelProgress({
  userId,
  currentLevel,
  lastWin,
  onSelectLevel,
}) {
  const storageKey = `${STORAGE_KEY}:${userId ?? "guest"}`;
  const [earnedLevels, setEarnedLevels] = useState(() => readLevels(storageKey));

  useEffect(() => {
    const cachedLevels = readLevels(storageKey);
    setEarnedLevels(cachedLevels);
    if (userId == null) return undefined;

    let active = true;
    fetchMaiaAchievements()
      .then((serverLevels) => {
        if (!active) return;
        const merged = mergeLevels(cachedLevels, serverLevels);
        saveLevels(storageKey, merged);
        setEarnedLevels(merged);
      })
      .catch(() => undefined);

    return () => {
      active = false;
    };
  }, [storageKey, userId]);

  useEffect(() => {
    if (!lastWin || !MAIA_LEVELS.includes(Number(lastWin.elo))) return;
    const winUserId = lastWin.userId == null ? null : String(lastWin.userId);
    const currentUserId = userId == null ? null : String(userId);
    if (winUserId !== currentUserId) return;

    setEarnedLevels((current) => {
      const merged = mergeLevels(current, [lastWin.elo]);
      saveLevels(storageKey, merged);
      return merged;
    });
  }, [lastWin, storageKey, userId]);

  return (
    <section
      className="maia-level-progress"
      aria-label={`Победы над Maia: ${earnedLevels.length} из ${MAIA_LEVELS.length}`}
    >
      <h3 className="maia-level-progress-title">
        Победы над Maia <span>· {earnedLevels.length}/{MAIA_LEVELS.length}</span>
      </h3>
      <div className="maia-level-progress-grid" aria-label="Уровни Maia">
        {MAIA_LEVELS.map((level) => {
          const earned = earnedLevels.includes(level);
          const current = Number(currentLevel) === level;
          return (
            <button
              key={level}
              type="button"
              className={`maia-level-chip${earned ? " is-earned" : ""}${current ? " is-current" : ""}`}
              aria-label={`${level} ELO${earned ? ", победа засчитана" : ""}${current ? ", выбран" : ""}`}
              aria-pressed={current}
              title={`${level} ELO${earned ? " · победа" : ""}`}
              onClick={() => onSelectLevel(level)}
            >
              {earned ? <span aria-hidden="true">✓</span> : level}
            </button>
          );
        })}
      </div>
    </section>
  );
}
