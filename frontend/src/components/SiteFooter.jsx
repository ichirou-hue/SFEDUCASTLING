// Сквозной футер сайта — внутри скролл-контейнера каждой страницы,
// поэтому всегда внизу контента, а не зафиксированной полосой.
//
// Ссылка «Документ для ознакомления» — временная заглушка (href="#")
// перед добавлением настоящего документа; откроется в новой вкладке.
// Паттерн как в форме регистрации (RegisterModal: REG_DOC_URL/REG_DOC_LABEL).
const DOC_URL = "#";
const DOC_LABEL = "Документ для ознакомления";

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <span className="site-footer-brand">GIGACHESS · обучение шахматам</span>
      <a
        className="site-footer-link"
        href={DOC_URL}
        target="_blank"
        rel="noopener noreferrer"
      >
        {DOC_LABEL}
      </a>
    </footer>
  );
}
