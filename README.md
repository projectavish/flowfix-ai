# FlowFix AI

This project helps analyze task workflows and identify bottlenecks in the system. Been working on this for a couple weeks now to make task management more efficient.

## What it does

- Loads task data and figures out where things are getting stuck
- Uses ML to predict which tasks might cause delays 
- Gets suggestions from GPT on how to improve processes
- Tracks improvements over time to see what's actually working
- Generates reports and dashboards for visualization

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

You'll need to set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## Files

**notebooks/** - did most of the initial exploration and model testing here
- `eda.ipynb` - data analysis and visualizations
- `ml_modeling.ipynb` - model training experiments

**src/** - main code
- `ingestion.py` - loads the data
- `bottleneck_detector.py` - finds problem areas
- `gpt_suggester.py` - gets AI recommendations
- `ml_predictor.py` - predicts task issues
- `improvement_tracker.py` - tracks changes
- `pdf_generator.py` - makes reports

**data/** - task dataset (1000 records)

**exports/** - output files for Power BI

**dashboard/** - Power BI dashboard file

## Running it

Start with the notebooks to see how everything works. The EDA notebook shows the analysis, modeling notebook has the ML stuff.

For production use, run the scripts in src/ - check the code comments for details on what each one does.

## Notes

Still improving the prediction accuracy. Current model works ok but could be better with more data. GPT suggestions are hit or miss depending on how specific the bottleneck is.

Dashboard connects to the CSV exports if you want to visualize things in Power BI.
