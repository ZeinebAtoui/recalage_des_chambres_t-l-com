import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from passlib.context import CryptContext

DB_PATH = Path(__file__).resolve().parent.parent / "app.db"

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 jours

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    if SECRET_KEY == "dev-secret-change-me-in-production":
        print("⚠️  JWT_SECRET_KEY non defini : cle de developpement utilisee (a ne pas utiliser en production).")


def get_password_hash(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def get_user_by_email(email: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


def create_user(email: str, password: str) -> dict:
    if get_user_by_email(email):
        raise ValueError("Un compte existe deja avec cet email.")

    hashed = get_password_hash(password)
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, hashed_password, created_at) VALUES (?, ?, ?)",
            (email, hashed, created_at),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "email": email, "created_at": created_at}


def authenticate_user(email: str, password: str) -> dict | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Retourne l'email (sub) contenu dans le token, leve jwt.PyJWTError si invalide/expire."""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload["sub"]
