"""Analytics Dashboard for RCT Field Flow
Access and analyze user activity, engagement, and usage patterns.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

# Database path
DB_PATH = Path(__file__).parent / "rct_field_flow" / "persistent_data" / "rct_field_flow.db"


def get_connection():
    """Get database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_user_summary():
    """Get summary statistics for all users."""
    conn = get_connection()
    
    query = """
    SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN password_hash IS NOT NULL THEN 1 END) as registered_users,
        COUNT(CASE WHEN consent_given = 1 THEN 1 END) as users_with_consent,
        MIN(first_access) as first_user_date,
        MAX(last_access) as last_activity_date
    FROM users
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_all_users():
    """Get detailed list of all users."""
    conn = get_connection()
    
    query = """
    SELECT 
        username,
        name,
        organization,
        first_access,
        last_access,
        user_id,
        consent_given,
        CASE WHEN password_hash IS NOT NULL THEN 'Yes' ELSE 'No' END as has_password
    FROM users
    ORDER BY last_access DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_user_activity_summary():
    """Get activity summary by user."""
    conn = get_connection()
    
    query = """
    SELECT 
        u.username,
        u.name,
        u.organization,
        COUNT(a.id) as total_activities,
        COUNT(DISTINCT a.page) as unique_pages_visited,
        MIN(a.timestamp) as first_activity,
        MAX(a.timestamp) as last_activity
    FROM users u
    LEFT JOIN activities a ON u.username = a.username
    GROUP BY u.username, u.name, u.organization
    ORDER BY total_activities DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_activity_by_page():
    """Get activity counts by page."""
    conn = get_connection()
    
    query = """
    SELECT 
        page,
        COUNT(*) as activity_count,
        COUNT(DISTINCT username) as unique_users
    FROM activities
    GROUP BY page
    ORDER BY activity_count DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_activity_by_action():
    """Get activity counts by action type."""
    conn = get_connection()
    
    query = """
    SELECT 
        action,
        COUNT(*) as activity_count,
        COUNT(DISTINCT username) as unique_users
    FROM activities
    GROUP BY action
    ORDER BY activity_count DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_daily_active_users(days=30):
    """Get daily active users for the last N days."""
    conn = get_connection()
    
    # Calculate the date N days ago
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = f"""
    SELECT 
        DATE(timestamp) as date,
        COUNT(DISTINCT username) as active_users,
        COUNT(*) as total_activities
    FROM activities
    WHERE timestamp >= '{cutoff_date}'
    GROUP BY DATE(timestamp)
    ORDER BY date DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_user_details(username):
    """Get detailed information for a specific user."""
    conn = get_connection()
    
    # User info
    user_query = """
    SELECT 
        username,
        name,
        organization,
        first_access,
        last_access,
        user_id,
        consent_given,
        consent_timestamp
    FROM users
    WHERE username = ?
    """
    user_df = pd.read_sql_query(user_query, conn, params=(username,))
    
    # Activity log
    activity_query = """
    SELECT 
        timestamp,
        page,
        action,
        details
    FROM activities
    WHERE username = ?
    ORDER BY timestamp DESC
    """
    activity_df = pd.read_sql_query(activity_query, conn, params=(username,))
    
    # Design data
    design_query = """
    SELECT 
        team_name,
        program_card,
        current_step,
        workbook_json
    FROM design_data
    WHERE username = ?
    """
    design_df = pd.read_sql_query(design_query, conn, params=(username,))
    
    # Randomization data
    random_query = """
    SELECT 
        total_units,
        arms_json,
        timestamp
    FROM randomization
    WHERE username = ?
    """
    random_df = pd.read_sql_query(random_query, conn, params=(username,))
    
    conn.close()
    
    return {
        'user_info': user_df,
        'activities': activity_df,
        'design_data': design_df,
        'randomization': random_df
    }


def get_randomization_stats():
    """Get statistics about randomization usage."""
    conn = get_connection()
    
    query = """
    SELECT 
        u.username,
        u.organization,
        r.total_units,
        r.arms_json,
        r.timestamp as randomization_date
    FROM randomization r
    JOIN users u ON r.username = u.username
    ORDER BY r.timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse arms_json to get treatment arm info
    if not df.empty and 'arms_json' in df.columns:
        df['num_arms'] = df['arms_json'].apply(
            lambda x: len(json.loads(x)) if x else 0
        )
    
    return df


def get_design_workbook_stats():
    """Get statistics about design workbook usage."""
    conn = get_connection()
    
    query = """
    SELECT 
        u.username,
        u.organization,
        d.team_name,
        d.program_card,
        d.current_step,
        d.workbook_json
    FROM design_data d
    JOIN users u ON d.username = u.username
    ORDER BY d.current_step DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse workbook_json to count responses
    if not df.empty and 'workbook_json' in df.columns:
        df['num_responses'] = df['workbook_json'].apply(
            lambda x: len(json.loads(x)) if x else 0
        )
    
    return df


def export_analytics_report(output_dir="analytics_reports"):
    """Export comprehensive analytics report to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    reports = {
        f"user_summary_{timestamp}.csv": get_user_summary(),
        f"all_users_{timestamp}.csv": get_all_users(),
        f"user_activity_summary_{timestamp}.csv": get_user_activity_summary(),
        f"activity_by_page_{timestamp}.csv": get_activity_by_page(),
        f"activity_by_action_{timestamp}.csv": get_activity_by_action(),
        f"daily_active_users_{timestamp}.csv": get_daily_active_users(30),
        f"randomization_stats_{timestamp}.csv": get_randomization_stats(),
        f"design_workbook_stats_{timestamp}.csv": get_design_workbook_stats(),
    }
    
    for filename, df in reports.items():
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Exported: {filepath}")
    
    print(f"\n✓ All reports exported to {output_path.absolute()}")


def print_quick_stats():
    """Print quick statistics to console."""
    print("=" * 80)
    print("RCT FIELD FLOW - ANALYTICS DASHBOARD")
    print("=" * 80)
    
    # User summary
    user_summary = get_user_summary()
    print("\n📊 USER OVERVIEW")
    print("-" * 80)
    for col in user_summary.columns:
        print(f"{col}: {user_summary[col].values[0]}")
    
    # Top pages
    print("\n📄 TOP PAGES BY ACTIVITY")
    print("-" * 80)
    pages = get_activity_by_page()
    print(pages.head(10).to_string(index=False))
    
    # Top actions
    print("\n⚡ TOP ACTIONS")
    print("-" * 80)
    actions = get_activity_by_action()
    print(actions.head(10).to_string(index=False))
    
    # Recent activity
    print("\n📅 DAILY ACTIVE USERS (Last 7 Days)")
    print("-" * 80)
    dau = get_daily_active_users(7)
    print(dau.to_string(index=False))
    
    # Randomization usage
    print("\n🎲 RANDOMIZATION USAGE")
    print("-" * 80)
    rand_stats = get_randomization_stats()
    if not rand_stats.empty:
        print(f"Total randomizations: {len(rand_stats)}")
        print(f"Total units randomized: {rand_stats['total_units'].sum()}")
        print(f"Average units per randomization: {rand_stats['total_units'].mean():.0f}")
    else:
        print("No randomization data available")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "export":
            # Export full analytics report
            export_analytics_report()
        
        elif command == "user" and len(sys.argv) > 2:
            # Get details for a specific user
            username = sys.argv[2]
            details = get_user_details(username)
            print(f"\n📋 USER DETAILS: {username}")
            print("=" * 80)
            print("\nUser Info:")
            print(details['user_info'])
            print(f"\nTotal Activities: {len(details['activities'])}")
            print(f"\nRecent Activities:")
            print(details['activities'].head(20))
            
        elif command == "users":
            # List all users
            users = get_all_users()
            print("\n👥 ALL USERS")
            print("=" * 80)
            print(users.to_string(index=False))
        
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python analytics_dashboard.py          - Show quick statistics")
            print("  python analytics_dashboard.py export   - Export full analytics reports")
            print("  python analytics_dashboard.py users    - List all users")
            print("  python analytics_dashboard.py user <username> - Get details for a user")
    else:
        # Default: show quick stats
        print_quick_stats()
