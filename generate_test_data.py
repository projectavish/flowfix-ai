import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 500 tasks with RECENT dates (last 100 days)
np.random.seed(42)
n = 500

# Create dates from today backwards (Feb 26, 2026)
end_date = datetime(2026, 2, 26)
dates = [end_date - timedelta(days=int(x)) for x in np.random.randint(0, 100, n)]

df = pd.DataFrame({
    'task_id': [f'T{i:04d}' for i in range(n)],
    'task_name': [f'Task {i}' for i in range(n)],
    'assignee': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'], n),
    'project': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], n),
    'status': np.random.choice(['In Progress', 'Completed', 'Blocked', 'Not Started'], n, p=[0.4, 0.3, 0.2, 0.1]),
    'priority': np.random.choice(['Critical', 'High', 'Medium', 'Low'], n, p=[0.1, 0.3, 0.4, 0.2]),
    'created_at': dates,
    'actual_duration': np.random.randint(1, 20, n),
    'bottleneck_type': np.random.choice(['Process', 'Resource', 'Technical', 'Communication', None], n, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
    'severity_score': np.random.randint(10, 95, n),
    'detected_date': [d if np.random.random() > 0.3 else None for d in dates]
})

df.to_csv('test_tasks.csv', index=False)
print(f"✅ Generated {len(df)} tasks")
print(f"Date range: {min(dates).date()} to {max(dates).date()}")
print("File saved: test_tasks.csv")