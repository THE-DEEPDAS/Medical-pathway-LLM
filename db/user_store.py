import sqlite3
import json
from pathlib import Path
from typing import Optional
from models.user import UserProfile
from werkzeug.security import generate_password_hash, check_password_hash

class UserStore:
    def __init__(self):
        db_path = Path(__file__).parent / "users.db"
        self.conn = sqlite3.connect(str(db_path))
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    profile TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_user(self, user: UserProfile):
        password_hash = generate_password_hash(user.password)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO users (user_id, email, password_hash, profile) VALUES (?, ?, ?, ?)",
                (user.user_id, user.email, password_hash, user.json())
            )
        return user.user_id

    def get_user(self, user_id: str) -> UserProfile:
        cursor = self.conn.execute(
            "SELECT profile FROM users WHERE user_id = ?", 
            (user_id,)
        )
        if row := cursor.fetchone():
            return UserProfile.parse_raw(row[0])
        return None

    def verify_login(self, email: str, password: str) -> Optional[str]:
        cursor = self.conn.execute(
            "SELECT user_id, password_hash FROM users WHERE email = ?",
            (email,)
        )
        if row := cursor.fetchone():
            user_id, password_hash = row
            if check_password_hash(password_hash, password):
                return user_id
        return None
