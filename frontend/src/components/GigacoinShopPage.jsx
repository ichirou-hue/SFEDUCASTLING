import { Link } from "react-router-dom";
import GigacoinShop from "./GigacoinShop.jsx";
import SiteFooter from "./SiteFooter.jsx";
import "./GigacoinShopPage.css";

export default function GigacoinShopPage({ onUserChange }) {
  return (
    <main className="gigacoin-shop-page">
      <Link className="gigacoin-shop-page__back" to="/profile">
        <span aria-hidden="true">←</span> Профиль
      </Link>
      <GigacoinShop onUserChange={onUserChange} />
      <SiteFooter />
    </main>
  );
}
