import sqlite3
from pathlib import Path
import sys

# Add current directory to path to allow imports
sys.path.append(str(Path.cwd()))

from rct_field_flow.persistence import init_db

DB_PATH = Path("rct_field_flow/persistent_data/rct_field_flow.db")

def check_schema():
    print("Running init_db()...")
    try:
        init_db()
        print("init_db() completed.")
    except Exception as e:
        print(f"init_db() failed: {e}")

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        print("--- USERS TABLE INFO ---")
        cur = conn.execute("PRAGMA table_info(users)")
        columns = cur.fetchall()
        for col in columns:
            print(col)
            
        print("\n--- CHECKING FOR 'name' COLUMN ---")
        col_names = [c[1] for c in columns]
        if "name" in col_names:
            print("Column 'name' EXISTS.")
        else:
            print("Column 'name' MISSING.")
            
    finally:
        conn.close()

if __name__ == "__main__":
    check_schema()
