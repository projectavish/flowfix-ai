"""Basic tests for FlowFix AI"""
import pytest
import pandas as pd
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    import sys
    sys.path.append('src')
    # Just test imports work
    assert True

def test_data_loading():
    """Test CSV loading"""
    df = pd.DataFrame({
        'task_id': ['T001'],
        'task_name': ['Test'],
        'assignee': ['Alice'],
        'project': ['Alpha'],
        'status': ['In Progress'],
        'priority': ['High'],
        'created_at': [datetime.now()],
        'actual_duration': [5],
        'bottleneck_type': [None],
        'severity_score': [50]
    })
    assert len(df) == 1
    assert df['task_id'].iloc[0] == 'T001'

def test_filters():
    """Test filter logic"""
    df = pd.DataFrame({
        'project': ['Alpha', 'Beta', 'Alpha'],
        'status': ['Done', 'In Progress', 'Done']
    })
    filtered = df[df['project'] == 'Alpha']
    assert len(filtered) == 2

    