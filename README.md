
### Note: This project is actively evolving. Documentation and features are being refined as part of ongoing development.


# FlowFix AI - Production Workflow Analysis System

AI-powered workflow analysis system that identifies bottlenecks, predicts delays, and provides actionable recommendations to improve team productivity.

## Features

### Core Capabilities
- **Intelligent Data Ingestion** - Merge logic, validation, YAML config support
- **Bottleneck Detection** - ML-based severity scoring (0-100), auto-reporting
- **AI Recommendations** - GPT-4 suggestions with retry logic and quality scoring
- **ML Predictions** - Duration/delay forecasting with SHAP explainability
- **Task Reassignment** - Smart workload balancing with effectiveness tracking
- **Improvement Tracking** - Action logging with impact measurement and scoring
- **Feedback Loop** - Track suggestion impact with before/after metrics
- **PowerBI Integration** - Excel exports with proper schema and data cleaning
- **PDF Reports** - Professional reports with charts and ROI calculations
- **Interactive Dashboard** - Real-time Streamlit dashboard with filters

### Production Features
- Comprehensive CLI interfaces for all modules
- Automatic schema migrations
- Error handling and graceful fallbacks
- Extensive logging (module-specific log files)
- Model versioning and training history
- Data validation and type checking
- Query optimization and caching

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Basic Workflow
```bash
# Ingest data
cd src
python ingestion.py

# Detect bottlenecks
python bottleneck_detector.py report

# Get AI suggestions
python gpt_suggester.py batch --limit 5

# Train ML models
python ml_predictor.py train

# Launch dashboard
cd ..
streamlit run dashboard/app.py
```

## Module Documentation

### Core Modules

**ingestion.py** - Data ingestion with merge strategies
```bash
python src/ingestion.py --source data/file.csv
python src/ingestion.py --config config.yaml --validate-only
```

**bottleneck_detector.py** - Bottleneck analysis with severity scoring
```bash
python src/bottleneck_detector.py report
python src/bottleneck_detector.py detect --assignee "John Doe"
python src/bottleneck_detector.py severity
```

**gpt_suggester.py** - AI recommendations with quality scoring
```bash
python src/gpt_suggester.py suggest <task_id>
python src/gpt_suggester.py batch --limit 10
python src/gpt_suggester.py quality
```

**ml_predictor.py** - ML predictions with SHAP explainability
```bash
python src/ml_predictor.py train
python src/ml_predictor.py predict <task_id>
python src/ml_predictor.py explain <task_id>
```

**reassignment_tracker.py** - Task reassignment with effectiveness tracking
```bash
python src/reassignment_tracker.py report
python src/reassignment_tracker.py rebalance
python src/reassignment_tracker.py auto-reassign
```

**improvement_tracker.py** - Improvement action tracking
```bash
python src/improvement_tracker.py log <action> <description>
python src/improvement_tracker.py report
python src/improvement_tracker.py kpis
```

**feedback_loop.py** - Feedback tracking with impact measurement
```bash
python src/feedback_loop.py mark <task_id> applied --helpful True
python src/feedback_loop.py summary
python src/feedback_loop.py impact <task_id>
python src/feedback_loop.py report
```

**export_for_powerbi.py** - PowerBI data export
```bash
python src/export_for_powerbi.py
python src/export_for_powerbi.py quick
```

**fix_database_for_powerbi.py** - Database cleanup
```bash
python src/fix_database_for_powerbi.py
python src/fix_database_for_powerbi.py --dry-run --verbose
```

**pdf_generator.py** - PDF report generation
```bash
python src/pdf_generator.py
python src/pdf_generator.py --output exports/report.pdf
```

## Project Structure

```
Avish_flow/
├── src/              # All production modules
├── notebooks/        # EDA and ML experiments
├── models/           # Trained models and SHAP plots
├── exports/          # PowerBI exports and PDF reports
├── dashboard/        # Streamlit dashboard
├── data/             # Dataset (1000 task records)
├── flowfix.db        # SQLite database
└── *.log             # Module-specific logs
```

## Database Schema

- **tasks** - Main task data
- **gpt_suggestions** - AI recommendations with feedback
- **bottleneck_history** - Detected bottlenecks
- **ml_predictions** - Model predictions
- **task_reassignments** - Reassignment tracking
- **improvement_log** - Improvement actions
- **feedback_log** - Impact tracking
- **dashboard_summary** - KPI aggregates

## Advanced Usage

See **USER_GUIDE.md** for:
- Detailed CLI command reference
- API route documentation
- Configuration options
- Troubleshooting guide
- Performance optimization tips
- Complete workflow examples

## Key Metrics

- **Severity Scoring**: 0-100 scale (Low/Medium/High/Critical)
- **Quality Scoring**: 0-100 for AI suggestions
- **Impact Scoring**: 0-100 for improvements
- **ROI Calculations**: Time saved, cost savings

## Requirements

- Python 3.8+
- OpenAI API key
- SQLite 3.x
- See requirements.txt for complete list

## Development

All modules include:
- Comprehensive logging
- Error handling with fallbacks
- Input validation
- CLI interfaces
- Type hints
- Docstrings

**Version**: 2.0 (Production Release)  
**Last Updated**: January 8, 2026
