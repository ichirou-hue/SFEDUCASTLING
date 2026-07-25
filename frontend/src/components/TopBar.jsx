export default function TopBar() {
  return (
    <div className="top-bar">
      <img src="/logo_sfedu_white.png" alt="ЮФУ" className="sfedu-logo" />
      <div className="title-block">
        <div className="title">
          <img src="/knight-s.svg" alt="S" className="t-knight-img" />
          <span className="t-sfedu">FEDU</span>
          <span className="t-castling">CASTLING</span>
        </div>
        <div className="subtitle">
          <span className="ai-dot"></span> Powered by GigaChat
        </div>
      </div>
      <span className="version-tag">
        <span className="crown">♛</span> AI CHESS PLATFORM
      </span>
    </div>
  )
}
