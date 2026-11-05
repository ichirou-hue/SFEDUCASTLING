import { useEffect, useRef } from "react";

/*
 * Автопрокрутка ленты чата к последнему сообщению.
 *
 * Важно, что прокручивается только сам контейнер: scrollIntoView умеет
 * заодно двигать внешние скролл-контейнеры, из-за чего «дёргается» вся
 * раскладка страницы.
 *
 * Мини-доска react-chessboard получает финальную высоту уже после
 * первого кадра, поэтому одного scrollTo недостаточно: лента вырастает
 * позже, и нижняя часть сообщения остаётся обрезанной. Держим её
 * прижатой к низу, пока последнее сообщение дорисовывается.
 */
export default function useChatAutoScroll(messages) {
  const listRef = useRef(null);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return undefined;

    let frame = 0;
    let timer = 0;
    let observer = null;

    const scrollToEnd = (behavior = "smooth") => {
      list.scrollTo({ top: list.scrollHeight, behavior });
    };

    const scheduleScroll = (behavior = "smooth") => {
      cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => scrollToEnd(behavior));
    };

    scheduleScroll();

    const messageNodes = list.querySelectorAll(".chat-msg");
    const lastMessage = messageNodes[messageNodes.length - 1];

    if (typeof ResizeObserver === "function" && lastMessage) {
      observer = new ResizeObserver(() => scheduleScroll("auto"));
      observer.observe(lastMessage);
      // После дорисовки доски наблюдатель больше не нужен: иначе любое
      // изменение высоты окна будет само прокручивать ленту.
      timer = window.setTimeout(() => {
        observer?.disconnect();
        observer = null;
      }, 800);
    }

    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(timer);
      observer?.disconnect();
    };
  }, [messages]);

  return listRef;
}
