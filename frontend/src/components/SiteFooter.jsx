// Сквозной футер сайта - внутри скролл-контейнера каждой страницы,
// поэтому всегда внизу контента, а не зафиксированной полосой.
import { FOOTER_LEGAL_DOCS } from "../legalDocs.js";

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <span className="site-footer-brand">GIGACHESS · обучение шахматам</span>
      <nav className="site-footer-docs" aria-label="Правовые документы">
        {FOOTER_LEGAL_DOCS.map((doc) => (
          <a
            key={doc.href}
            className="site-footer-link"
            href={doc.href}
            target="_blank"
            rel="noopener noreferrer"
          >
            {doc.label}
          </a>
        ))}
      </nav>
    </footer>
  );
}
