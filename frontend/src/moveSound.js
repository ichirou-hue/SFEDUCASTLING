/*
 * Звук хода для всех досок сайта.
 *
 * Один AudioContext на страницу: браузеры ограничивают их количество,
 * а звук может запрашиваться сразу с нескольких досок. Если контекст
 * создан не по клику (например, ход бота пришёл по сети), он может быть
 * suspended - поэтому перед проигрыванием пытаемся его разбудить.
 */
let audioCtx = null;

function getAudioContext() {
  const AudioContextClass =
    typeof window !== "undefined" &&
    (window.AudioContext || window.webkitAudioContext);

  if (!AudioContextClass) return null;

  if (!audioCtx) audioCtx = new AudioContextClass();

  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }

  return audioCtx;
}

export function playMoveSound() {
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1200, now);
  filter.frequency.exponentialRampToValueAtTime(300, now + 0.08);

  osc.type = "triangle";
  osc.frequency.setValueAtTime(180, now);
  osc.frequency.exponentialRampToValueAtTime(60, now + 0.06);

  gain.gain.setValueAtTime(0.5, now);
  gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);

  osc.connect(filter);
  filter.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(now + 0.12);
}
