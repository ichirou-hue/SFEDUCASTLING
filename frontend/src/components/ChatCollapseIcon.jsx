/*
 * Двойной шеврон для кнопки «Свернуть чат».
 *
 * Раньше здесь стоял текстовый символ »: его чернила смещены внутри
 * глифа, поэтому в круглой кнопке он выглядел прижатым к одной стороне.
 * SVG нарисован симметрично относительно viewBox и центрируется
 * флексом одинаково на всех страницах.
 */
export default function ChatCollapseIcon() {
  return (
    <svg
      className="chat-collapse-icon"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 7l5 5-5 5" />
      <path d="M13 7l5 5-5 5" />
    </svg>
  );
}
