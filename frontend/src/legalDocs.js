// Юридические документы сайта. PDF лежат в frontend/public/documents
// и раздаются по абсолютным путям как в dev-сервере, так и в production.
export const LEGAL_DOCS = {
  privacy: {
    href: "/documents/01_Politika_konfidentsialnosti.pdf",
    label: "Политика конфиденциальности",
  },
  consent: {
    href: "/documents/02_Soglasie_na_obrabotku_PD.pdf",
    label: "Согласие на обработку персональных данных",
  },
  terms: {
    href: "/documents/03_Uslovia_ispolzovania.pdf",
    label: "Условия использования",
  },
  processors: {
    href: "/documents/04_Prilozhenie_3_Obrabotchiki_PD.pdf",
    label: "Приложение № 3. Обработчики персональных данных",
  },
  browserStorage: {
    href: "/documents/05_Prilozhenie_4_Khranilische_brauzera.pdf",
    label: "Приложение № 4. Данные в хранилище браузера",
  },
  informationSecurity: {
    href: "/documents/06_Politika_informatsionnoy_bezopasnosti.pdf",
    label: "Политика информационной безопасности",
  },
};

// В футере показываем только ключевые документы, с которыми
// пользователь соглашается при регистрации.
export const FOOTER_LEGAL_DOCS = [
  LEGAL_DOCS.terms,
  LEGAL_DOCS.privacy,
  LEGAL_DOCS.consent,
];
