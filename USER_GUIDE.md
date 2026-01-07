# FlowFix AI - Complete User Guide

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Basic Workflow
```bash
# 1. Ingest data
python src/ingestion.py

# 2. Detect bottlenecks
python src/bottleneck_detector.py

# 3. Get AI suggestions
python src/gpt_suggester.py

# 4. Generate predictions
python src/ml_predictor.py

# 5. View dashboard
streamlit run dashboard/app.py
```

---

## Module Guide

### 1. Data Ingestion (`ingestion.py`)
Loads and merges task data with intelligent duplicate handling.

```bash
# CLI Usage
python src/ingestion.py --source data/FlowFixAI_FinalTaskData_1000.csv
python src/ingestion.py --config config.yaml
python src/ingestion.py --validate-only
```

**Features:**
- Merge logic with configurable strategies (replace/keep_first/keep_last/append)
- Data validation and type checking
- YAML configuration support
- Comprehensive logging

---

### 2. Bottleneck Detector (`bottleneck_detector.py`)
Identifies workflow bottlenecks using clustering and severity scoring.

```bash
# CLI Usage
python src/bottleneck_detector.py report
python src/bottleneck_detector.py detect --assignee "John Doe"
python src/bottleneck_detector.py severity
python src/bottleneck_detector.py export
python src/bottleneck_detector.py auto-report
```

**Features:**
- Severity scoring algorithm (0-100 scale)
- ML-based duration estimation
- Auto-generated reports
- Export to CSV/JSON

**Severity Levels:**
- 0-25: Low (monitoring)
- 26-50: Medium (action needed)
- 51-75: High (urgent)
- 76-100: Critical (immediate attention)

---

### 3. GPT Suggester (`gpt_suggester.py`)
AI-powered suggestions with retry logic and quality scoring.

```bash
# CLI Usage
python src/gpt_suggester.py suggest <task_id>
python src/gpt_suggester.py batch --limit 10
python src/gpt_suggester.py analyze <task_id>
python src/gpt_suggester.py quality
python src/gpt_suggester.py alert
```

**Features:**
- Exponential backoff retry (2s→4s→8s)
- Prompt versioning (v2.0)
- Quality scoring (0-100)
- Sentiment analysis (positive/neutral/negative)
- Urgency detection (low/medium/high/critical)
- High-severity alerts

**Quality Metrics:**
- Length score (max 25 points)
- Recommendation count (max 25 points)
- Actionability (max 25 points)
- Specificity (max 25 points)

---

### 4. ML Predictor (`ml_predictor.py`)
Machine learning models for duration/delay prediction with SHAP explainability.

```bash
# CLI Usage
python src/ml_predictor.py train
python src/ml_predictor.py predict <task_id>
python src/ml_predictor.py batch
python src/ml_predictor.py evaluate
python src/ml_predictor.py explain <task_id>
python src/ml_predictor.py optimize
```

**Features:**
- 18 engineered features
- SHAP visualizations (summary plots, importance charts)
- Optimal threshold finding (Youden's J statistic)
- Model versioning (v1.1)
- Training run logging
- Feature set export for reproducibility

**Models:**
- Duration predictor: Random Forest Regressor
- Delay predictor: Random Forest Classifier

---

### 5. Reassignment Tracker (`reassignment_tracker.py`)
Tracks task reassignments with ML/bottleneck integration.

```bash
# CLI Usage
python src/reassignment_tracker.py report
python src/reassignment_tracker.py rebalance
python src/reassignment_tracker.py auto-reassign
python src/reassignment_tracker.py effectiveness
```

**Features:**
- Bottleneck-triggered reassignment
- ML-triggered auto-reassignment (P(delay)>70%)
- Weekly workload rebalancing
- Effectiveness tracking (before/after metrics)
- Multiple trigger types: manual/bottleneck/ml_prediction/gpt

---

### 6. Improvement Tracker (`improvement_tracker.py`)
Tracks improvement actions with scoring and KPI integration.

```bash
# CLI Usage
python src/improvement_tracker.py log <action> <description>
python src/improvement_tracker.py mark-applied <action_id>
python src/improvement_tracker.py compare <period1> <period2>
python src/improvement_tracker.py report
python src/improvement_tracker.py kpis
python src/improvement_tracker.py history --days 30
```

**Features:**
- Improvement scoring (0-100)
- Weighted algorithm: duration 40%, delay 30%, bottleneck 30%
- API routes for web integration
- Auto-push to dashboard
- Historical trend analysis

**Score Categories:**
- 80-100: Excellent
- 60-79: Significant
- 40-59: Moderate
- 0-39: Minor

---

### 7. PowerBI Export (`export_for_powerbi.py`)
Exports database to Excel for Power BI with correct schema.

```bash
# CLI Usage
python src/export_for_powerbi.py
python src/export_for_powerbi.py quick
python src/export_for_powerbi.py custom my_report
```

**Features:**
- 7 sheets: Tasks, Bottlenecks, GPT_Suggestions, Improvements, Reassignments, ML_Predictions, Summary_Metrics
- Fixed column names (root_causes, recommendations)
- Feedback fields integration
- Try/except per sheet with fallback queries
- Dynamic/static filenames

---

### 8. PowerBI Fixer (`fix_database_for_powerbi.py`)
Cleans and normalizes database for Power BI compatibility.

```bash
# CLI Usage
python src/fix_database_for_powerbi.py
python src/fix_database_for_powerbi.py --dry-run
python src/fix_database_for_powerbi.py --verbose
python src/fix_database_for_powerbi.py --nulls-only
python src/fix_database_for_powerbi.py --types-only
python src/fix_database_for_powerbi.py --casing-only
python src/fix_database_for_powerbi.py --orphans-only
```

**Features:**
- NULL value fixing (table-specific defaults)
- Data type standardization (dates→TEXT YYYY-MM-DD, durations→INTEGER)
- Casing normalization (Priority: High/Medium/Low, Status: In Progress/Completed)
- Orphan record removal (FK validation)
- Database vacuum
- Verification reports

---

### 9. Feedback Loop (`feedback_loop.py`)
Tracks suggestion feedback with impact measurement.

```bash
# CLI Usage
python src/feedback_loop.py mark <task_id> applied --notes "Worked great" --helpful True
python src/feedback_loop.py impact <task_id>
python src/feedback_loop.py summary
python src/feedback_loop.py summary --status applied
python src/feedback_loop.py applied --limit 20
python src/feedback_loop.py export --output exports/feedback.csv
python src/feedback_loop.py view --status applied --limit 10
python src/feedback_loop.py trends --days 30
python src/feedback_loop.py report
```

**Features:**
- Track actual impact (duration reduced, delay prevention)
- Impact scoring (0-100): duration 50pts, delay prevention 25pts, completion 25pts
- Input validation
- Feedback viewer with filtering
- CSV export
- Trend analysis
- ROI calculations

**Feedback Statuses:**
- applied: Suggestion implemented
- rejected: Suggestion not used
- pending: Under review
- under_review: Being evaluated

---

### 10. PDF Generator (`pdf_generator.py`)
Generates comprehensive PDF reports with charts.

```bash
# CLI Usage
python src/pdf_generator.py
python src/pdf_generator.py --output exports/report.pdf
python src/pdf_generator.py --no-charts
python src/pdf_generator.py --verbose
```

**Features:**
- Safe file path handling
- Smart text truncation (preserve sentence boundaries)
- Graceful fallback for missing data
- Impact summary with ROI calculations
- Matplotlib impact charts
- SHAP visualization integration
- 6-section comprehensive report

**Report Sections:**
1. Executive Summary (overview, key metrics, team size)
2. Impact Summary (ROI, time saved, AI performance)
3. Bottleneck Analysis (types, frequency, severity)
4. Team Performance (workload, delays, bottlenecks)
5. AI-Powered Recommendations (top quality suggestions)
6. Next Steps (immediate, short-term, long-term actions)

---

## Dashboard (`streamlit run dashboard/app.py`)

Interactive web dashboard with:
- Real-time metrics
- Bottleneck visualizations
- GPT suggestion viewer
- ML prediction charts
- Improvement tracking
- Filters by assignee, priority, project
- Date range selection

---

## Database Schema

**Tables:**
- `tasks` - Main task data
- `gpt_suggestions` - AI recommendations with feedback
- `bottleneck_history` - Detected bottlenecks
- `ml_predictions` - Model predictions
- `ml_training_log` - Training history
- `task_reassignments` - Reassignment tracking
- `improvement_log` - Improvement actions
- `feedback_log` - Impact tracking
- `dashboard_summary` - KPI aggregates
- `ingestion_log` - Data load history

---

## Configuration Files

### `.env` (required)
```
OPENAI_API_KEY=sk-...
```

### `config.yaml` (optional)
```yaml
data_source: data/FlowFixAI_FinalTaskData_1000.csv
merge_strategy: replace
database_path: flowfix.db
enable_validation: true
```

---

## Typical Workflow

### Initial Setup
```bash
# 1. Install and configure
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key" > .env

# 2. Ingest data
python src/ingestion.py

# 3. Detect bottlenecks
python src/bottleneck_detector.py report

# 4. Train ML models
python src/ml_predictor.py train
```

### Daily Operations
```bash
# Morning: Check dashboard
streamlit run dashboard/app.py

# Get AI suggestions for bottlenecks
python src/gpt_suggester.py batch --limit 5

# Predict delays
python src/ml_predictor.py batch

# Export for Power BI
python src/export_for_powerbi.py quick
```

### Weekly Review
```bash
# Generate PDF report
python src/pdf_generator.py

# Review feedback
python src/feedback_loop.py report

# Rebalance workload
python src/reassignment_tracker.py rebalance

# Track improvements
python src/improvement_tracker.py report
```

---

## Troubleshooting

### OpenAI API Errors
- Check `.env` file has valid API key
- GPT suggester has retry logic (3 attempts)
- Use `--verbose` flag for detailed logs

### Database Errors
- Run `fix_database_for_powerbi.py --verbose` to clean data
- Check `flowfix.db` exists in project root
- Verify table schema with SQLite browser

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version ≥3.8
- Update packages: `pip install --upgrade -r requirements.txt`

### SHAP Visualization Errors
- Train ML model first: `python src/ml_predictor.py train`
- Check `models/shap_plots/` directory exists
- Install matplotlib: `pip install matplotlib`

---

## Logging

All modules log to individual files:
- `ingestion.log`
- `bottleneck_detector.log`
- `gpt_suggester.log`
- `ml_predictor.log`
- `feedback_loop.log`
- `pdf_generator.log`

Use `--verbose` flag for console output.

---

## API Routes (Improvement Tracker)

```python
# Track improvement
POST /api/improvement/track
{
    "action": "Reassigned tasks",
    "description": "Redistributed workload",
    "category": "reassignment"
}

# Get stats
GET /api/improvement/stats

# Mark applied
PUT /api/improvement/<action_id>/apply
```

---

## Performance Tips

1. **Batch Processing**: Use batch commands for multiple tasks
2. **Caching**: Models cached in `models/` directory
3. **Incremental Updates**: Use `--validate-only` to check before ingesting
4. **Parallel Execution**: Run independent modules simultaneously
5. **Database Optimization**: Run `vacuum_database()` weekly

---

## Project Structure

```
Avish_flow/
├── data/
│   └── FlowFixAI_FinalTaskData_1000.csv
├── src/
│   ├── ingestion.py
│   ├── bottleneck_detector.py
│   ├── gpt_suggester.py
│   ├── ml_predictor.py
│   ├── reassignment_tracker.py
│   ├── improvement_tracker.py
│   ├── export_for_powerbi.py
│   ├── fix_database_for_powerbi.py
│   ├── feedback_loop.py
│   ├── pdf_generator.py
│   └── utils.py
├── models/
│   ├── duration_predictor_v1.1_*.pkl
│   ├── delay_predictor_v1.1_*.pkl
│   ├── shap_plots/
│   └── model_summary.txt
├── exports/
│   ├── powerbi_data_*.xlsx
│   ├── feedback_report.csv
│   └── flowfix_report_*.pdf
├── notebooks/
│   ├── eda.ipynb
│   └── ml_modeling.ipynb
├── dashboard/
│   └── app.py
├── flowfix.db (SQLite database)
├── requirements.txt
├── .env (create this)
└── README.md
```

---

## Support

For issues or questions:
1. Check logs in project root (*.log files)
2. Review error messages with `--verbose`
3. Verify database schema with PowerBI fixer
4. Refer to module-specific documentation in source files

---

**Last Updated:** January 8, 2026
**Version:** 2.0 (Production Release)
