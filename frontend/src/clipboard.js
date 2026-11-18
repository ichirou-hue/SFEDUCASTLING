/*
 * Копирование текста в буфер обмена с резервным способом для страниц по
 * http, где navigator.clipboard недоступен. Возвращает true при успехе.
 */
export default async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Например, страница открыта по http - пробуем резервный способ.
    }
  }

  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "0";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    return copied;
  } catch {
    return false;
  }
}
