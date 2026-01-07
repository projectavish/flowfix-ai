"""Schema inspection and migration utility"""
import sys
sys.path.insert(0, 'src')

from utils import get_engine
from sqlalchemy import text

engine = get_engine()
conn = engine.connect()

# List all tables
result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
tables = [row[0] for row in result.fetchall()]

print("="*80)
print("DATABASE SCHEMA INSPECTION")
print("="*80)
print(f"\nTables found: {len(tables)}")
print("-" * 80)

for table in tables:
    print(f"\n{table}:")
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    columns = result.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

print("\n" + "="*80)
