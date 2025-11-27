"""Persistence layer for RCT Field Flow.
Stores user sessions and activity logs in a local SQLite database for
retrospective access after browser sessions end.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
import hashlib
import secrets
from cryptography.fernet import Fernet
import bcrypt

DB_DIR = Path(__file__).parent / "persistent_data"
DB_PATH = DB_DIR / "rct_field_flow.db"

SCHEMA = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            organization TEXT,
            first_access TIMESTAMP,
            last_access TIMESTAMP,
            user_id TEXT UNIQUE,
            username_hash TEXT,
            username_salt TEXT,
            encrypted_username TEXT,
            password_hash TEXT,
            name TEXT,
            consent_given INTEGER DEFAULT 0,
            consent_timestamp TIMESTAMP
        )
    """,
    "activities": """
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            user_id TEXT,
            timestamp TIMESTAMP,
            page TEXT,
            action TEXT,
            details TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """,
    "design_data": """
        CREATE TABLE IF NOT EXISTS design_data (
            username TEXT PRIMARY KEY,
            user_id TEXT,
            team_name TEXT,
            program_card TEXT,
            current_step INTEGER,
            workbook_json TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """,
    "randomization": """
        CREATE TABLE IF NOT EXISTS randomization (
            username TEXT PRIMARY KEY,
            user_id TEXT,
            total_units INTEGER,
            arms_json TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """,
}

# Optional encryption key loading
_FERNET: Optional[Fernet] = None
KEY_ENV = "PERSISTENCE_ENCRYPT_KEY"

def _init_fernet() -> None:
    global _FERNET
    key = os.getenv(KEY_ENV)
    if not key:
        # Attempt to load from file
        key_file = DB_DIR / "fernet.key"
        if key_file.exists():
            key = key_file.read_text().strip()
        else:
            # Generate new key and persist to file for reuse
            key = Fernet.generate_key().decode()
            key_file.write_text(key)
    try:
        _FERNET = Fernet(key.encode())
    except Exception:
        _FERNET = None  # Encryption disabled if key invalid

def _hash_username(username: str, salt: str) -> str:
    return hashlib.sha256((salt + username).encode()).hexdigest()

def _encrypt_username(username: str) -> Optional[str]:
    if _FERNET is None:
        return None
    return _FERNET.encrypt(username.encode()).decode()


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Initialize database schema and apply migrations for new columns."""
    _init_fernet()
    conn = _connect()
    try:
        for ddl in SCHEMA.values():
            conn.execute(ddl)
        # Ensure required columns exist (idempotent migration)
        def ensure_columns(table: str, columns: List[str]):
            cur = conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            for col, ddl_col in columns:
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_col}")
        ensure_columns("users", [
            ("user_id", "TEXT UNIQUE"),
            ("username_hash", "TEXT"),
            ("username_salt", "TEXT"),
            ("encrypted_username", "TEXT"),
            ("password_hash", "TEXT"),
            ("name", "TEXT"),
            ("consent_given", "INTEGER DEFAULT 0"),
            ("consent_timestamp", "TIMESTAMP"),
        ])
        ensure_columns("activities", [("user_id", "TEXT")])
        ensure_columns("design_data", [("user_id", "TEXT")])
        ensure_columns("randomization", [("user_id", "TEXT")])
        conn.commit()
    finally:
        conn.close()


def record_user_login(username: str, organization: Optional[str], consent: bool) -> None:
    """Insert or update a user row when temporary access granted (with hashing/encryption)."""
    now = datetime.utcnow().isoformat()
    salt = secrets.token_hex(16)
    hashed = _hash_username(username, salt)
    encrypted = _encrypt_username(username)
    user_id = secrets.token_hex(12)  # 96-bit identifier
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id, username_salt FROM users WHERE username = ?", (username,))
        existing = cur.fetchone()
        if existing:
            # Preserve existing user_id & salt for stability
            existing_user_id, existing_salt = existing
            hashed = _hash_username(username, existing_salt) if existing_salt else hashed
            user_id = existing_user_id or user_id
            cur.execute(
                "UPDATE users SET organization = ?, last_access = ?, username_hash = ?, encrypted_username = ?, consent_given = ?, consent_timestamp = ? WHERE username = ?",
                (
                    organization,
                    now,
                    hashed,
                    encrypted,
                    1 if consent else 0,
                    now if consent else None,
                    username,
                ),
            )
        else:
            cur.execute(
                "INSERT INTO users (username, organization, first_access, last_access, user_id, username_hash, username_salt, encrypted_username, consent_given, consent_timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    username,
                    organization,
                    now,
                    now,
                    user_id,
                    hashed,
                    salt,
                    encrypted,
                    1 if consent else 0,
                    now if consent else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def create_user(username: str, password: str, name: str, organization: Optional[str]) -> bool:
    """Register a new user with password (bcrypt hashed)."""
    if not username or not password:
        return False
    
    # Check if user exists
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False  # User already exists
            
        now = datetime.utcnow().isoformat()
        
        # Hash password with bcrypt
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Legacy fields
        salt = secrets.token_hex(16)
        username_hash = _hash_username(username, salt)
        encrypted_username = _encrypt_username(username)
        user_id = secrets.token_hex(12)
        
        cur.execute(
            "INSERT INTO users (username, password_hash, name, organization, first_access, last_access, user_id, username_hash, username_salt, encrypted_username) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                username,
                password_hash,
                name,
                organization,
                now,
                now,
                user_id,
                username_hash,
                salt,
                encrypted_username
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def verify_login(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify username and password. Returns user dict if successful, None otherwise."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, password_hash, username_salt, name, organization, user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return None
            
        db_username, db_pwd_hash, salt, name, org, user_id = row
        
        # Verify password
        if not db_pwd_hash or not salt:
            return None # Legacy user without password
            
        check_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        if check_hash == db_pwd_hash:
            # Update last access
            now = datetime.utcnow().isoformat()
            conn.execute("UPDATE users SET last_access = ? WHERE username = ?", (now, username))
            conn.commit()
            
            return {
                "username": db_username,
                "name": name,
                "organization": org,
                "user_id": user_id
            }
        return None
    finally:
        conn.close()



def record_activity(username: Optional[str], page: str, action: str, details: Optional[Dict[str, Any]]) -> None:
    """Persist an activity log entry with user_id if available."""
    if not username:
        return
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        user_id = row[0] if row and row[0] else None
        conn.execute(
            "INSERT INTO activities (username, user_id, timestamp, page, action, details) VALUES (?,?,?,?,?,?)",
            (
                username,
                user_id,
                datetime.utcnow().isoformat(),
                page,
                action,
                json.dumps(details) if details else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_design_data(username: str, team_name: Optional[str], program_card: Optional[str], current_step: Optional[int], workbook_responses: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        user_id = user_row[0] if user_row else None
        cur.execute("SELECT username FROM design_data WHERE username = ?", (username,))
        workbook_json = json.dumps(workbook_responses or {})
        if cur.fetchone():
            cur.execute(
                "UPDATE design_data SET team_name = ?, program_card = ?, current_step = ?, workbook_json = ?, user_id = ? WHERE username = ?",
                (team_name, program_card, current_step, workbook_json, user_id, username),
            )
        else:
            cur.execute(
                "INSERT INTO design_data (username, user_id, team_name, program_card, current_step, workbook_json) VALUES (?,?,?,?,?,?)",
                (username, user_id, team_name, program_card, current_step, workbook_json),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_randomization(username: str, total_units: int, arms: List[Dict[str, Any]], timestamp: Optional[str]) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        user_id = user_row[0] if user_row else None
        cur.execute("SELECT username FROM randomization WHERE username = ?", (username,))
        arms_json = json.dumps(arms or [])
        ts = timestamp or datetime.utcnow().isoformat()
        if cur.fetchone():
            cur.execute(
                "UPDATE randomization SET total_units = ?, arms_json = ?, timestamp = ?, user_id = ? WHERE username = ?",
                (total_units, arms_json, ts, user_id, username),
            )
        else:
            cur.execute(
                "INSERT INTO randomization (username, user_id, total_units, arms_json, timestamp) VALUES (?,?,?,?,?)",
                (username, user_id, total_units, arms_json, ts),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_all_users() -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, organization, first_access, last_access, user_id, consent_given, username_hash, encrypted_username FROM users ORDER BY last_access DESC")
        rows = cur.fetchall()
        return [
            {
                "username": r[0],
                "organization": r[1],
                "first_access": r[2],
                "last_access": r[3],
                "user_id": r[4],
                "consent": bool(r[5]),
                "hashed": bool(r[6]),
                "encrypted": bool(r[7]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def fetch_user_session(username: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, organization, first_access, last_access, user_id, consent_given, consent_timestamp FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return {
            "username": row[0],
            "organization": row[1],
            "first_access": row[2],
            "last_access": row[3],
            "user_id": row[4],
            "consent": bool(row[5]),
            "consent_timestamp": row[6],
        } if row else {}
    finally:
        conn.close()


def fetch_user_activity(username: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, page, action, details FROM activities WHERE username = ? ORDER BY timestamp ASC",
            (username,),
        )
        rows = cur.fetchall()
        return [
            {
                "timestamp": r[0],
                "page": r[1],
                "action": r[2],
                "details": json.loads(r[3]) if r[3] else None,
            }
            for r in rows
        ]
    finally:
        conn.close()


def fetch_user_design(username: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT team_name, program_card, current_step, workbook_json FROM design_data WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "team_name": row[0],
            "program_card": row[1],
            "current_step": row[2],
            "workbook_responses": json.loads(row[3]) if row[3] else {},
        }
    finally:
        conn.close()


def fetch_user_randomization(username: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT total_units, arms_json, timestamp FROM randomization WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "total_units": row[0],
            "treatment_arms": json.loads(row[1]) if row[1] else [],
            "timestamp": row[2],
        }
    finally:
        conn.close()

# Maintenance utilities
def delete_user(username: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM activities WHERE username = ?", (username,))
        conn.execute("DELETE FROM design_data WHERE username = ?", (username,))
        conn.execute("DELETE FROM randomization WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()

def anonymize_user(username: str) -> Optional[str]:
    """Replace username with pseudonymous handle; returns new username."""
    new_username = f"anon_{secrets.token_hex(6)}"
    conn = _connect()
    try:
        cur = conn.cursor()
        # Fetch existing fields to preserve user_id
        cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return None
        user_id = row[0]
        salt = secrets.token_hex(16)
        hashed = _hash_username(new_username, salt)
        encrypted = _encrypt_username(new_username)
        cur.execute(
            "UPDATE users SET username = ?, username_hash = ?, username_salt = ?, encrypted_username = ? WHERE user_id = ?",
            (new_username, hashed, salt, encrypted, user_id),
        )
        # Update child tables
        for table in ["activities", "design_data", "randomization"]:
            conn.execute(f"UPDATE {table} SET username = ? WHERE username = ?", (new_username, username))
        conn.commit()
        return new_username
    finally:
        conn.close()

def prune_activities(before_iso_timestamp: str) -> int:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM activities WHERE timestamp < ?", (before_iso_timestamp,))
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

def vacuum_db() -> None:
    conn = _connect()
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


def fetch_users_for_auth() -> Dict[str, Dict[str, Any]]:
    """Fetch all users with credentials for streamlit-authenticator."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, name, password_hash FROM users WHERE password_hash IS NOT NULL")
        rows = cur.fetchall()
        users = {}
        for r in rows:
            users[r[0]] = {
                "name": r[1] or r[0],
                "password": r[2],
                "email": f"{r[0]}@example.com", # Placeholder as we don't store email yet
            }
        return users
    finally:
        conn.close()
