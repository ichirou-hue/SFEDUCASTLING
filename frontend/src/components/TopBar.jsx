export default function TopBar({ onRegister }) {
  return (
    <div className="top-bar">
      <div className="logo-wrapper">
        <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
      </div>
      <button className="reg-btn" onClick={onRegister}>Регистрация</button>
    </div>
  )
}
