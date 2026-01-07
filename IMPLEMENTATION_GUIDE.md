# FlowFix AI - Remaining Implementation Guide

This document outlines the remaining features to implement for a production-grade portfolio project.

## ✅ Already Completed Features

1. **Utils/DB Module** - Full production-grade with error handling, FK cascades, test functions
2. **Ingestion Module** - Merge logic, validation, dry-run, YAML config, failed row logging  
3. **Bottleneck Detector** - Severity scoring, history logging, ML estimator, auto-reports
4. **Streamlit Dashboard** - Interactive UI with filters, visualizations, multiple tabs

## 🔨 Remaining Features to Implement

### 1. GPT Suggester Enhancements

**File:** `src/gpt_suggester.py`

**Features Needed:**
- ✅ Basic GPT integration (already exists)
- ⚠️  **Retry logic with exponential backoff** - Handle 429 rate limit errors
- ⚠️  **Prompt versioning** - Track which prompt version generated each suggestion
- ⚠️  **Response scoring** - Score suggestions on clarity and actionability
- ⚠️  **A/B prompt testing** - Test multiple prompts and track effectiveness
- ⚠️  **High-severity alerts** - Auto-flag urgent suggestions
- ⚠️  **Enhanced tracking** - Log model used, latency, sentiment, urgency

**Implementation Steps:**

1. Add retry logic to `call_gpt()`:
```python
import time
from openai import RateLimitError

def call_gpt_with_retry(prompt, client, max_retries=3):
    """Call GPT with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": prompt}
                ],
                temperature=GPT_TEMPERATURE,
                max_tokens=GPT_MAX_TOKENS
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return response.choices[0].message.content, latency_ms
        
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                logger.warning(f"Rate limit hit, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            logger.error(f"GPT call failed: {str(e)}")
            return None, None
```

2. Add prompt versioning:
```python
PROMPT_VERSION = "2.0"  # Update when changing prompt structure

def save_suggestion(task_id, suggestion_text, parsed_data, latency_ms, model):
    """Save with tracking fields"""
    query = text("""
        INSERT INTO gpt_suggestions 
        (task_id, suggestion_text, root_causes, recommendations,
         prompt_version, gpt_model_used, latency_ms, sentiment, urgency_level)
        VALUES (:task_id, :suggestion_text, :root_causes, :recommendations,
                :prompt_version, :model, :latency, :sentiment, :urgency)
    """)
    
    # Analyze sentiment and urgency from response
    sentiment = analyze_sentiment(suggestion_text)  # 'positive', 'neutral', 'negative'
    urgency = detect_urgency(suggestion_text)  # 'low', 'medium', 'high', 'critical'
    
    # Execute insert...
```

3. Add response scoring:
```python
def score_suggestion(suggestion_text, recommendations):
    """Score suggestion quality (0-100)"""
    score = 50  # Base score
    
    # Clarity (length and structure)
    if len(suggestion_text) > 100:
        score += 10
    if len(recommendations) >= 3:
        score += 15
    
    # Actionability (contains action verbs)
    action_verbs = ['reassign', 'escalate', 'review', 'increase', 'reduce', 'implement']
    text_lower = suggestion_text.lower()
    for verb in action_verbs:
        if verb in text_lower:
            score += 5
    
    # Specificity (mentions names, numbers, dates)
    if any(word.isupper() for word in suggestion_text.split()):
        score += 10  # Has proper nouns
    
    return min(score, 100)
```

### 2. ML Module Enhancements

**File:** `src/ml_predictor.py` or `notebooks/ml_modeling.ipynb`

**Features Needed:**
- ⚠️  **Feature engineering** - Add task_name_length, comment_length, days_in_project, etc.
- ⚠️  **SHAP visualizations** - Feature importance plots
- ⚠️  **Model versioning** - Save models with version + date
- ⚠️  **Prediction to DB** - Log predictions to ml_predictions table
- ⚠️  **Training logs** - Track every training run with metrics
- ⚠️  **Threshold tuning** - ROC analysis for optimal threshold
- ⚠️  **Weekly retraining** - Scheduler/CRON job
- ⚠️  **Validation & export** - Clean data and export feature sets

**Implementation Priority:**
1. Feature engineering
2. SHAP visualizations  
3. Model versioning
4. Prediction logging

### 3. Reassignment Tracker Integration

**File:** `src/reassignment_tracker.py`

**Features Needed:**
- ⚠️  **Trigger from bottleneck engine** - Auto-trigger on Assignee_Bottleneck
- ⚠️  **Track reassignment count** - Update tasks.reassignment_count
- ⚠️  **Link to GPT** - Include reassignment history in GPT prompts
- ⚠️  **Feature for ML** - Use reassignment_count in delay predictor
- ⚠️  **High delay prediction trigger** - If delay > 70% → auto reassign
- ⚠️  **Weekly rebalancing** - Script to balance workload
- ⚠️  **Effectiveness tracking** - Measure if reassignment helped

**Key Integration Points:**
```python
# In bottleneck_detector.py
if 'Assignee_Bottleneck' in bottleneck_type:
    from reassignment_tracker import suggest_reassignment
    suggest_reassignment(row['task_id'], row['assignee'])

# In gpt_suggester.py - create_prompt()
reassignment_history = get_reassignment_history(task['task_id'])
prompt += f"\nReassignment History: {reassignment_history}"
```

### 4. Improvement Tracker Upgrades

**File:** `src/improvement_tracker.py`

**Features Needed:**
- ⚠️  **UI/CLI for logging** - Easy way to log improvements
- ⚠️  **Score field** - improvement_score in DB (already added to schema)
- ⚠️  **Push KPIs to dashboard_summary** - For real-time tracking
- ⚠️  **API route** (optional) - For future web integration

**Quick Implementation:**
```python
def log_improvement_cli():
    """CLI interface for logging improvements"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id')
    parser.add_argument('action')
    parser.add_argument('--score', type=float, default=0)
    args = parser.parse_args()
    
    save_improvement_log(
        task_id=args.task_id,
        action_taken=args.action,
        improvement_score=args.score
    )
    
    # Push to dashboard
    from utils import update_dashboard_kpis
    update_dashboard_kpis()
```

### 5. PDF Generator Enhancements

**File:** `src/pdf_generator.py`

**Features Needed:**
- ⚠️  **Fix file paths** - Use os.path.join() for cross-platform compatibility
- ⚠️  **Better text truncation** - Don't slice mid-sentence
- ⚠️  **Fallback for missing tables** - Handle gracefully
- ⚠️  **Impact summary** - Show delay reduced %, duration improved, etc.
- ⚠️  **Insert charts** - Add matplotlib/plotly charts to PDF

### 6. Feedback Loop Enhancements

**File:** `src/feedback_loop.py`

**Features Needed:**
- ✅ Basic feedback tracking (already exists)
- ⚠️  **Impact tracking** - Track actual vs predicted improvement
- ⚠️  **Feedback viewer** - Function to display all feedback
- ⚠️  **Export feedback** - To CSV/PDF for reporting
- ⚠️  **Validation** - Check task_id exists before update

### 7. PowerBI Export Fixes

**File:** `src/export_for_powerbi.py`

**Features Needed:**
- ⚠️  **Fix column names** - Use correct schema (root_causes not root_cause)
- ⚠️  **Add feedback fields** - feedback_status, feedback_date, applied
- ⚠️  **Try/except per export** - Don't fail completely on one error
- ⚠️  **Add improvement sheet** - Export improvement_log table
- ⚠️  **Dynamic filename** - Add timestamp to output file

**Quick Fix:**
```python
def export_to_excel(filename=None):
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'powerbi_export_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename) as writer:
        try:
            # Tasks
            tasks_df = execute_query("SELECT * FROM tasks")
            tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
        except Exception as e:
            logger.error(f"Tasks export failed: {e}")
        
        try:
            # GPT with feedback
            gpt_query = """
            SELECT task_id, root_causes, recommendations, 
                   feedback_status, feedback_date, applied
            FROM gpt_suggestions
            """
            gpt_df = execute_query(gpt_query)
            gpt_df.to_excel(writer, sheet_name='GPT_Suggestions', index=False)
        except Exception as e:
            logger.error(f"GPT export failed: {e}")
        
        # Add improvement sheet
        try:
            imp_df = execute_query("SELECT * FROM improvement_log")
            imp_df.to_excel(writer, sheet_name='Improvements', index=False)
        except Exception as e:
            logger.error(f"Improvements export failed: {e}")
```

### 8. PowerBI Fixer Script

**File:** `src/fix_database_for_powerbi.py`

**Features Needed:**
- ⚠️  **Standardize data types** - Convert dates to TEXT, durations to INTEGER
- ⚠️  **Normalize casing** - .title() or .upper() for priority, status
- ⚠️  **Remove orphans** - Delete records with invalid FK references
- ⚠️  **Enhanced logging** - Show rows updated, nulls fixed
- ⚠️  **CLI options** - --verbose, --dry-run flags

## 🎯 Implementation Priority Order

For maximum LinkedIn impact, implement in this order:

### Phase 1 (High Impact) - Do First
1. **GPT Retry Logic** - Shows production thinking
2. **SHAP Visualizations** - ML portfolio piece  
3. **Prompt Versioning** - Shows A/B testing mindset
4. **Model Versioning** - Production ML practices

### Phase 2 (Medium Impact)
5. **PowerBI Export Fixes** - Clean data pipeline
6. **Reassignment Integration** - Shows system thinking
7. **PDF Charts** - Visual reporting

### Phase 3 (Nice to Have)
8. **Weekly Retraining** - Automation
9. **API Routes** - Future-proofing
10. **CLI Enhancements** - User experience

## 📝 Commit Message Style

Continue using conventional commits:
- `feat: added SHAP visualizations for ML explainability`
- `refactor: implemented GPT retry logic with exponential backoff`
- `fix: corrected PowerBI export column names`
- `docs: added implementation guide for remaining features`

## 🚀 Testing Checklist

Before pushing each feature:
- [ ] Feature works with sample data
- [ ] Error handling in place
- [ ] Logging added
- [ ] No breaking changes to existing code
- [ ] Commit message is clear

## 📊 LinkedIn Post Ideas

As you implement, post about:
- "Implemented exponential backoff retry logic for GPT API calls"
- "Added SHAP visualizations to explain ML model predictions"
- "Built production-grade data pipeline with validation"
- "Integrated real-time dashboard with Streamlit"

---

**Current Status:** ~60% complete - Core infrastructure and high-impact features done
**Remaining Work:** ~40% - Enhancements and polish features
**Estimated Time:** 10-15 hours to complete all remaining features

Good luck! The foundation is solid - now it's about adding the production polish. 🚀
