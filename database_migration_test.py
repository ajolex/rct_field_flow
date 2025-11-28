"""
Test script to verify PostgreSQL migration and database connectivity
Run this locally to test SQLite, then deploy to test PostgreSQL on Streamlit Cloud
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from rct_field_flow.persistence import (
    init_db,
    create_user,
    record_activity,
    fetch_all_users,
    fetch_user_activity,
    _use_postgres,
    _connect
)

def test_database_migration():
    """Test database operations to ensure PostgreSQL compatibility"""
    
    print("=" * 60)
    print("RCT Field Flow - Database Migration Test")
    print("=" * 60)
    
    # Check which database we're using
    using_postgres = _use_postgres()
    print(f"\n✓ Database type: {'PostgreSQL' if using_postgres else 'SQLite'}")
    
    # Test connection
    try:
        conn = _connect()
        print("✓ Database connection successful")
        conn.close()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    # Initialize database
    try:
        init_db()
        print("✓ Database schema initialized")
    except Exception as e:
        print(f"✗ Schema initialization failed: {e}")
        return False
    
    # Test user creation
    test_username = f"test_user_migration"
    try:
        success = create_user(
            username=test_username,
            password="test_password_123",
            name="Test Migration User",
            organization="Test Organization"
        )
        if success:
            print(f"✓ User creation successful: {test_username}")
        else:
            print(f"ℹ️  User already exists: {test_username}")
    except Exception as e:
        print(f"✗ User creation failed: {e}")
        return False
    
    # Test activity recording
    try:
        record_activity(
            username=test_username,
            page="test",
            action="migration_test",
            details={"test": "PostgreSQL migration"}
        )
        print("✓ Activity recording successful")
    except Exception as e:
        print(f"✗ Activity recording failed: {e}")
        return False
    
    # Test data retrieval
    try:
        users = fetch_all_users()
        print(f"✓ User retrieval successful ({len(users)} users)")
        
        activities = fetch_user_activity(test_username)
        print(f"✓ Activity retrieval successful ({len(activities)} activities)")
    except Exception as e:
        print(f"✗ Data retrieval failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Database migration is ready.")
    print("=" * 60)
    
    if not using_postgres:
        print("\nℹ️  NOTE: Currently using SQLite (local development)")
        print("   To test PostgreSQL:")
        print("   1. Configure .streamlit/secrets.toml with Supabase credentials")
        print("   2. Deploy to Streamlit Cloud")
        print("   3. Check the app logs to verify PostgreSQL connection")
    
    return True

if __name__ == "__main__":
    success = test_database_migration()
    sys.exit(0 if success else 1)
