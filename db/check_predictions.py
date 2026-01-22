import sqlite3

DB = "data/workflow_data.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

print("rows ml_predictions:",
      cur.execute("select count(*) from ml_predictions").fetchone()[0])

print("by model_type:",
      cur.execute("""
        select model_type, count(*)
        from ml_predictions
        group by model_type
      """).fetchall())

print("duration min/avg/max:",
      cur.execute("""
        select min(prediction_value),
               avg(prediction_value),
               max(prediction_value)
        from ml_predictions
        where model_type='duration'
      """).fetchone())

print("delay_prob min/avg/max:",
      cur.execute("""
        select min(prediction_value),
               avg(prediction_value),
               max(prediction_value)
        from ml_predictions
        where model_type='delay_probability'
      """).fetchone())

conn.close()
print("done")