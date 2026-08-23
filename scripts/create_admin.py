"""Создание администратора (задача 64).

Использование (из корня проекта, с активированным venv):
    python -m scripts.create_admin <login> <password> [--email me@example.com]

Если логин уже существует — пользователь повышается до админа (is_admin=True),
пароль НЕ меняется. Для смены пароля используйте --set-password.
"""

import argparse
import asyncio
import getpass
import sys

from sqlalchemy import select

from backend.api_gateway.security import hash_password
from backend.db.session import async_session_factory
from backend.models.user import User


async def create_admin(login: str, password: str, email: str | None, set_password: bool) -> int:
    async with async_session_factory() as db:
        user = await db.scalar(select(User).where(User.login == login))
        if user is None:
            user = User(
                login=login,
                email=email,
                password_hash=hash_password(password),
                is_admin=True,
            )
            db.add(user)
            action = "создан"
        else:
            user.is_admin = True
            if email:
                user.email = email
            if set_password:
                user.password_hash = hash_password(password)
            action = "обновлён (is_admin=True)"
        await db.commit()
        print(f"Админ '{login}' {action}: id={user.id}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Создать/повысить администратора")
    parser.add_argument("login", help="логин администратора (3-32 символа)")
    parser.add_argument(
        "password",
        nargs="?",
        default=None,
        help="пароль (если не указан — спросит скрытым вводом)",
    )
    parser.add_argument("--email", default=None, help="email (необязательно)")
    parser.add_argument(
        "--set-password",
        action="store_true",
        help="для существующего пользователя также сменить пароль",
    )
    args = parser.parse_args()

    if not (3 <= len(args.login) <= 32):
        print("Ошибка: логин должен быть 3-32 символа", file=sys.stderr)
        return 1

    password = args.password or getpass.getpass("Пароль администратора: ")
    if len(password) < 8:
        print("Ошибка: пароль минимум 8 символов", file=sys.stderr)
        return 1

    return asyncio.run(create_admin(args.login, password, args.email, args.set_password))


if __name__ == "__main__":
    raise SystemExit(main())
