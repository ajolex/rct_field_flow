import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from rct_field_flow.persistence import init_db, create_user

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    
    # Create admin user
    print("Creating admin user (aj-admin)...")
    success = create_user("aj-admin", "admin123", "Admin User", "Research Organization")
    if success:
        print("✅ Admin user created successfully!")
    else:
        print("⚠️ Admin user already exists or error occurred")
    
    # Create test user
    print("Creating test user (testuser)...")
    success = create_user("testuser", "password123", "Test User", "Student")
    if success:
        print("✅ Test user created successfully!")
    else:
        print("⚠️ Test user already exists or error occurred")
    
    print("\nYou can now log in with:")
    print("  Username: aj-admin, Password: admin123")
    print("  Username: testuser, Password: password123")
