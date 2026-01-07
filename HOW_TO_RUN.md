# 🚀 FlowFix AI - How to Run This Project

**A Simple Guide for Anyone to Understand and Run**

---

## 📖 What Does This Project Do?

**FlowFix AI** helps teams find problems in their workflow and suggests fixes using AI.

**Real-world example:**
- Your team has 1,000 tasks in a project
- Some tasks take too long (bottlenecks)
- Some team members are overloaded
- FlowFix AI automatically finds these problems and tells you how to fix them

**What you get:**
1. 📊 **Dashboard** - See your team's performance at a glance
2. 🤖 **AI Suggestions** - ChatGPT recommends solutions
3. 📈 **Predictions** - Machine learning predicts which tasks will be delayed

---

## 🏗️ How Does It Work?

Think of it like a factory assembly line:

```
Step 1: Read Data          →  Load tasks from CSV file
Step 2: Find Problems      →  Detect bottlenecks (369 found!)
Step 3: Train AI           →  Teach computer to predict delays
Step 4: Get Suggestions    →  Ask ChatGPT for solutions
Step 5: Visualize          →  Show everything in Power BI dashboard
```

---

## 📁 Project Files Explained

```
Avish_flow/
│
├── data/                              
│   ├── FlowFixAI_FinalTaskData_1000.csv  ← Your task data (1,000 tasks)
│   └── workflow_data.db                   ← Database (auto-created)
│
├── src/                               ← The "brain" of the project
│   ├── ingestion.py                   ← Step 1: Loads data into database
│   ├── bottleneck_detector.py         ← Step 2: Finds workflow problems
│   ├── ml_predictor.py                ← Step 3: Trains AI models
│   ├── gpt_suggester.py               ← Step 4: Gets ChatGPT recommendations
│   └── utils.py                       ← Helper functions (don't touch!)
│
├── notebooks/                         ← Analysis notebooks
│   ├── eda.ipynb                      ← Data exploration & charts
│   └── ml_modeling.ipynb              ← Model training details
│
├── dashboard/                         
│   └── flowfix_dashboard.pbix         ← Power BI dashboard (double-click to open)
│
├── models/                            
│   ├── duration_predictor.pkl         ← Trained AI model (auto-saved)
│   └── delay_classifier.pkl           ← Trained AI model (auto-saved)
│
├── exports/                           
│   └── gpt_suggestions.csv            ← AI recommendations (auto-saved)
│
├── requirements.txt                   ← List of needed software
├── .env.example                       ← API key template
└── README.md                          ← Project overview
```

---

## ⚙️ Installation & Setup

### **Step 1: Install Python** (if you don't have it)
1. Go to: https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. Install it (check "Add to PATH" during installation)

### **Step 2: Open Terminal**
1. Open folder: `C:\Users\anshu\Desktop\Projects\Avish_flow`
2. Right-click empty space → **Open in Terminal** (or PowerShell)

### **Step 3: Create Virtual Environment**
```bash
python -m venv .venv
```

### **Step 4: Activate Virtual Environment**
```bash
.venv\Scripts\activate
```
You'll see `(.venv)` at the start of your command line.

### **Step 5: Install Required Packages**
```bash
pip install -r requirements.txt
```
Wait 1-2 minutes for installation.

### **Step 6: Setup OpenAI API Key** (Optional - for GPT suggestions)
1. Copy `.env.example` and rename to `.env`
2. Open `.env` file
3. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. If you don't have an API key, skip GPT step (everything else still works!)

---

## 🚀 How to Run the Project

### **Method 1: Run Complete Pipeline (Automated)**

Just run these commands one by one:

```bash
# Navigate to src folder
cd src

# Step 1: Load data (creates database)
python ingestion.py

# Step 2: Find bottlenecks
python bottleneck_detector.py

# Step 3: Train AI models (optional)
python ml_predictor.py

# Step 4: Get GPT suggestions (needs API key)
python gpt_suggester.py
```

**That's it!** Your database is ready for Power BI.

---

### **Method 2: Run Jupyter Notebooks (For Analysis)**

```bash
# Install Jupyter (if not already)
pip install jupyter

# Start Jupyter
jupyter notebook
```

Browser opens automatically. Click on:
1. `notebooks/eda.ipynb` - See data analysis
2. `notebooks/ml_modeling.ipynb` - See model training

---

### **Method 3: Open Power BI Dashboard**

**No coding needed!**

1. Navigate to: `dashboard/`
2. Double-click: `flowfix_dashboard.pbix`
3. Power BI Desktop opens
4. Explore 5 pages:
   - Page 1: Executive Summary
   - Page 2: Bottleneck Analysis
   - Page 3: GPT Recommendations
   - Page 4: Team Performance
   - Page 5: Project Insights

---

## 🎯 What Happens When You Run Each Script?

### **1. ingestion.py** (2 seconds)
**What it does:** Reads CSV file, cleans data, saves to database

**Output you'll see:**
```
🚀 Starting data ingestion...
✅ Read 1000 records
✅ Cleaned 1000 records
✅ Loaded to database
```

**What's created:**
- `data/workflow_data.db` (SQLite database)
- 1,000 tasks ready for analysis

---

### **2. bottleneck_detector.py** (5 seconds)
**What it does:** Finds workflow problems

**Output you'll see:**
```
🔍 BOTTLENECK DETECTION ENGINE
✅ Found 81 delayed tasks
✅ Found 214 start delays
✅ Found 114 blocked tasks

Bottleneck Summary:
   Resource_Availability: 159
   Blocked: 114
   Assignee_Bottleneck: 64
   Stalled: 24
   Review_Bottleneck: 8
```

**What's updated:**
- Tasks marked with bottleneck types
- Summary saved to database

---

### **3. ml_predictor.py** (30-60 seconds)
**What it does:** Trains 2 AI models

**Output you'll see:**
```
🤖 Training Duration Predictor...
Test MAE: 2.38 days
✅ Model saved

🤖 Training Delay Classifier...
Test Accuracy: 75.4%
✅ Model saved
```

**What's created:**
- `models/duration_predictor.pkl`
- `models/delay_classifier.pkl`

---

### **4. gpt_suggester.py** (10-30 seconds)
**What it does:** Asks ChatGPT for recommendations

**Output you'll see:**
```
🤖 GPT SUGGESTION ENGINE
Processing 3 bottlenecked tasks...
✅ Generated 6 recommendations
✅ Saved to exports/gpt_suggestions.csv
```

**What's created:**
- `exports/gpt_suggestions.csv`
- Recommendations saved to database

---

## 🐛 Common Issues & Solutions

### **Issue 1: "python: command not found"**
**Solution:** Python not installed or not in PATH
- Reinstall Python and check "Add to PATH"

### **Issue 2: "No module named 'pandas'"**
**Solution:** Virtual environment not activated or packages not installed
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### **Issue 3: "FileNotFoundError: CSV file not found"**
**Solution:** Make sure you're in the right folder
```bash
cd C:\Users\anshu\Desktop\Projects\Avish_flow
cd src
python ingestion.py
```

### **Issue 4: "OpenAI API Error"**
**Solution:** API key missing or wrong model
- Add API key to `.env` file
- Or skip GPT step (other parts work fine!)

### **Issue 5: Power BI can't connect to database**
**Solution:** Use Python script connection method
- In Power BI: Get Data → Python script
- Paste connection code (see README.md)

---

## 📊 Understanding the Results

### **What Do the Numbers Mean?**

**Executive Summary Page:**
- **Total Tasks: 1000** - Total tasks in your dataset
- **Delayed Tasks: 350** - Tasks that took longer than expected
- **Bottlenecks: 369** - Tasks with specific problems
- **Avg Duration: 6.7 days** - Average time to complete a task

**Bottleneck Types:**
- **Resource_Availability (159)** - Tasks delayed because team member not available
- **Blocked (114)** - Tasks stuck waiting for something
- **Assignee_Bottleneck (64)** - Specific person overloaded
- **Stalled (24)** - Tasks inactive for too long
- **Review_Bottleneck (8)** - Tasks stuck in review

**ML Model Performance:**
- **Duration Predictor MAE: 2.38 days** - On average, predictions are off by 2.38 days (pretty good!)
- **Delay Classifier Accuracy: 75.4%** - Correctly predicts delays 3 out of 4 times

---

## 🎓 For Presentations & Interviews

### **30-Second Explanation:**
> "FlowFix AI analyzes 1,000 workflow tasks to find bottlenecks. It uses machine learning to predict delays and ChatGPT to recommend solutions. I built a complete pipeline in Python, trained 2 ML models, and created an interactive Power BI dashboard."

### **Key Points to Mention:**
1. **Data Processing**: Cleaned and normalized 1,000 tasks
2. **Analysis**: Detected 369 bottlenecks across 5 categories
3. **Machine Learning**: Trained RandomForest models with 75% accuracy
4. **AI Integration**: Used GPT-4 for intelligent recommendations
5. **Visualization**: Built 5-page Power BI dashboard

### **Skills Demonstrated:**
- Python (pandas, scikit-learn, SQLAlchemy)
- SQL (database design, complex queries)
- Machine Learning (regression, classification)
- AI/NLP (OpenAI GPT-4 API)
- Data Visualization (Power BI)
- Software Engineering (modular code, documentation)

---

## 🔄 Re-running with New Data

Want to analyze different tasks?

1. Replace `data/FlowFixAI_FinalTaskData_1000.csv` with your CSV
2. Make sure columns match:
   - Task ID, Task Name, Assignee, Status
   - Created Date, Start Date, End Date
   - Priority, Comments, Project
3. Run the pipeline again:
   ```bash
   cd src
   python ingestion.py
   python bottleneck_detector.py
   python ml_predictor.py
   python gpt_suggester.py
   ```
4. Open Power BI → Click **Refresh** button

---

## 📞 Need Help?

**Check these first:**
1. Read error message carefully
2. Make sure virtual environment is activated: `(.venv)`
3. Check you're in correct folder: `cd src`
4. Verify all files exist in `data/` folder

**Still stuck?**
- Check `README.md` for detailed documentation
- Look at `SETUP_GUIDE.md` for installation help
- Review code comments in Python files

---

## ✅ Quick Checklist

Before presenting this project:

- [ ] All scripts run without errors
- [ ] Database created: `data/workflow_data.db` exists
- [ ] Models trained: `models/` folder has .pkl files
- [ ] Dashboard opens: `dashboard/flowfix_dashboard.pbix` works
- [ ] Can explain what each script does
- [ ] Understand the bottleneck types
- [ ] Can navigate all 5 Power BI pages
- [ ] ✨ NEW: PDF reports generated in `exports/` folder
- [ ] ✨ NEW: Improvement tracking and feedback loop working

---

## ✨ NEW FEATURES (Advanced)

After running the main pipeline, you can now use these additional features:

### 1. **Generate PDF Reports** 📄
```powershell
..\.venv\Scripts\python.exe pdf_generator.py
```
**What it does**: Creates a professional PDF report with:
- Executive summary (total tasks, delays, bottlenecks)
- Bottleneck breakdown by type
- Team performance metrics
- AI recommendations
- Actionable next steps

**Output**: `exports/flowfix_report_[timestamp].pdf` (opens like any PDF)

---

### 2. **Track Improvements** 📈
```powershell
..\.venv\Scripts\python.exe improvement_tracker.py
```
**What it does**: Measures your workflow's health over time
- Captures baseline metrics (current state)
- Tracks improvement actions
- Compares before vs after
- Shows improvement percentage

**Use case**: "We reduced delays from 10% to 7% - that's 30% improvement!"

---

### 3. **Feedback Loop for AI Suggestions** 🔄
```powershell
..\.venv\Scripts\python.exe feedback_loop.py
```
**What it does**: Track which AI suggestions you actually used
- Mark suggestions as Applied/Pending/Rejected
- Add notes explaining your decision
- Measure impact of applied suggestions
- Generate feedback reports

**Use case**: "3 out of 6 suggestions were applied successfully"

---

### 4. **Task Reassignment Tracking** 👥
```powershell
..\.venv\Scripts\python.exe reassignment_tracker.py
```
**What it does**: Monitor when tasks change owners
- Records who gave/received tasks
- Logs reasons for reassignment (e.g., "workload balancing")
- Analyzes reassignment patterns
- Measures impact on completion rates

**Use case**: "Omar gave away 2 tasks because he was overloaded"

---

## 🎉 You're Done!

**Congratulations!** You now have a complete, professional-grade data analytics project with **100% of features** from the specification.

**What makes this project special:**
✅ Real-world problem solving
✅ End-to-end automation
✅ Multiple technologies integrated
✅ Production-ready code
✅ Professional documentation
✅ Portfolio-worthy results
✅ ✨ **Advanced features**: PDF reports, improvement tracking, feedback loop, reassignment tracking

---

**Built with ❤️ using Python, Machine Learning, GPT-4, and Power BI**

*Last updated: January 2026*
*Status: ✅ 100% Complete*
