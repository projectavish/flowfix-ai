import sqlite3
import pandas as pd

DB = "data/workflow_data.db"
conn = sqlite3.connect(DB)

q = """
select
  t.task_id,
  t.task_name,
  t.assignee,
  t.status,
  t.priority,
  t.project,
  t.actual_duration,
  t.is_delayed,
  d.prediction_value as predicted_duration,
  p.prediction_value as predicted_delay_probability
from tasks t
left join ml_predictions d
  on t.task_id = d.task_id
 and d.model_type = 'duration'
left join ml_predictions p
  on t.task_id = p.task_id
 and p.model_type = 'delay_probability'
limit 10
"""

df = pd.read_sql_query(q, conn)
print(df)

conn.close()