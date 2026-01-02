"""
GPT-4 Suggestion Engine for FlowFix AI
Generates AI-powered recommendations for bottlenecked tasks
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from utils import get_engine, execute_query
import json

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o-mini')
GPT_TEMPERATURE = float(os.getenv('GPT_TEMPERATURE', 0.3))
GPT_MAX_TOKENS = int(os.getenv('GPT_MAX_TOKENS', 500))


def initialize_openai():
    """Initialize OpenAI client"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
        raise ValueError(
            "OpenAI API key not set. Please set OPENAI_API_KEY in .env file"
        )
    
    return OpenAI(api_key=OPENAI_API_KEY)


def get_bottlenecked_tasks():
    """Retrieve tasks with bottlenecks that need suggestions"""
    query = """
    SELECT 
        t.task_id,
        t.task_name,
        t.assignee,
        t.status,
        t.priority,
        t.project,
        t.start_date,
        t.end_date,
        t.actual_duration,
        t.bottleneck_type,
        t.comments
    FROM tasks t
    WHERE t.bottleneck_type IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM gpt_suggestions g 
            WHERE g.task_id = t.task_id
        )
    ORDER BY 
        CASE t.priority 
            WHEN 'High' THEN 1 
            WHEN 'Medium' THEN 2 
            ELSE 3 
        END,
        t.actual_duration DESC
    LIMIT 50
    """
    
    return execute_query(query)


def create_prompt(task):
    """Create GPT prompt for a specific task"""
    
    # Calculate context information
    duration_text = f"{task['actual_duration']} days" if task['actual_duration'] else "Unknown"
    comments_text = task['comments'] if task['comments'] and str(task['comments']) != 'nan' else "No comments available"
    
    prompt = f"""You are an expert project management consultant analyzing workflow bottlenecks.

Task Details:
- Task Name: {task['task_name']}
- Assignee: {task['assignee']}
- Project: {task['project']}
- Priority: {task['priority']}
- Status: {task['status']}
- Duration: {duration_text}
- Bottleneck Type: {task['bottleneck_type']}
- Comments: {comments_text}

Based on this information, please provide:

1. TWO most likely root causes for this bottleneck
2. THREE specific, actionable recommendations to resolve this issue and prevent similar delays in future sprints

Format your response as:
ROOT CAUSES:
1. [First root cause]
2. [Second root cause]

RECOMMENDATIONS:
1. [First recommendation]
2. [Second recommendation]
3. [Third recommendation]

Keep responses concise and actionable."""

    return prompt


def call_gpt(prompt, client):
    """Call GPT-4 API with retry logic"""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert project management consultant specializing in workflow optimization and bottleneck analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=GPT_TEMPERATURE,
            max_tokens=GPT_MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"❌ Error calling GPT API: {str(e)}")
        return None


def parse_gpt_response(response_text):
    """Parse GPT response into structured format"""
    try:
        # Split by sections
        parts = response_text.split('RECOMMENDATIONS:')
        
        if len(parts) == 2:
            root_causes_section = parts[0].replace('ROOT CAUSES:', '').strip()
            recommendations_section = parts[1].strip()
            
            # Extract causes (lines starting with numbers)
            root_causes = [
                line.strip() 
                for line in root_causes_section.split('\n') 
                if line.strip() and line.strip()[0].isdigit()
            ]
            
            # Extract recommendations
            recommendations = [
                line.strip() 
                for line in recommendations_section.split('\n') 
                if line.strip() and line.strip()[0].isdigit()
            ]
            
            return {
                'root_causes': root_causes,
                'recommendations': recommendations
            }
    except:
        pass
    
    # Fallback: return full text
    return {
        'root_causes': [],
        'recommendations': [],
        'full_text': response_text
    }


def save_suggestion(task_id, suggestion_text, parsed_data):
    """Save GPT suggestion to database with Power BI compatible defaults"""
    engine = get_engine()
    
    # Convert lists to JSON strings with defaults for empty lists
    root_causes = parsed_data.get('root_causes', [])
    recommendations = parsed_data.get('recommendations', [])
    
    # Ensure we have at least empty strings, not NULL
    root_causes_json = json.dumps(root_causes) if root_causes else '[]'
    recommendations_json = json.dumps(recommendations) if recommendations else '[]'
    suggestion_text = suggestion_text if suggestion_text else ''
    
    query = text("""
        INSERT INTO gpt_suggestions 
        (task_id, suggestion_text, root_causes, recommendations)
        VALUES (:task_id, :suggestion_text, :root_causes, :recommendations)
    """)
    
    with engine.connect() as conn:
        conn.execute(query, {
            'task_id': task_id,
            'suggestion_text': suggestion_text,
            'root_causes': root_causes_json,
            'recommendations': recommendations_json
        })
        conn.commit()


def generate_suggestions(limit=None):
    """Main function to generate GPT suggestions for bottlenecked tasks"""
    print("\n" + "="*60)
    print("🤖 GPT SUGGESTION ENGINE")
    print("="*60 + "\n")
    
    # Check API key
    try:
        client = initialize_openai()
    except ValueError as e:
        print(f"❌ {str(e)}")
        print("\nTo use GPT suggestions:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run this script again")
        return
    
    # Get bottlenecked tasks
    print("📊 Fetching bottlenecked tasks...")
    tasks = get_bottlenecked_tasks()
    
    if len(tasks) == 0:
        print("✅ No new bottlenecked tasks found requiring suggestions")
        return
    
    if limit:
        tasks = tasks.head(limit)
    
    print(f"✅ Found {len(tasks)} tasks needing suggestions\n")
    
    # Generate suggestions
    success_count = 0
    error_count = 0
    
    for idx, task in tasks.iterrows():
        print(f"Processing [{idx+1}/{len(tasks)}]: {task['task_id']} - {task['task_name'][:50]}...")
        
        # Create prompt
        prompt = create_prompt(task)
        
        # Call GPT
        suggestion_text = call_gpt(prompt, client)
        
        if suggestion_text:
            # Parse response
            parsed_data = parse_gpt_response(suggestion_text)
            
            # Save to database
            try:
                save_suggestion(task['task_id'], suggestion_text, parsed_data)
                success_count += 1
                print(f"   ✅ Suggestion generated and saved")
            except Exception as e:
                print(f"   ❌ Error saving: {str(e)}")
                error_count += 1
        else:
            error_count += 1
        
        print()
    
    print("\n" + "="*60)
    print("📊 SUGGESTION GENERATION COMPLETE")
    print("="*60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📝 Total: {len(tasks)}")
    print("\n")


def export_suggestions_to_csv():
    """Export GPT suggestions to CSV file"""
    query = """
    SELECT 
        g.id,
        g.task_id,
        t.task_name,
        t.assignee,
        t.project,
        t.bottleneck_type,
        g.suggestion_text,
        g.root_causes,
        g.recommendations,
        g.applied,
        g.created_at
    FROM gpt_suggestions g
    JOIN tasks t ON g.task_id = t.task_id
    ORDER BY g.created_at DESC
    """
    
    df = execute_query(query)
    
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    output_path = os.path.join(exports_dir, "gpt_suggestions.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Exported {len(df)} suggestions to {output_path}")
    return output_path


def get_suggestion_summary():
    """Get summary statistics of GPT suggestions"""
    query = """
    SELECT 
        COUNT(*) as total_suggestions,
        COUNT(CASE WHEN applied = 1 THEN 1 END) as applied_count,
        COUNT(DISTINCT task_id) as unique_tasks
    FROM gpt_suggestions
    """
    
    return execute_query(query)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Processing limit: {limit} tasks")
        except ValueError:
            print("Usage: python gpt_suggester.py [limit]")
            sys.exit(1)
    
    # Generate suggestions
    generate_suggestions(limit=limit)
    
    # Export to CSV
    print("\n📤 Exporting suggestions...")
    export_suggestions_to_csv()
    
    # Show summary
    summary = get_suggestion_summary()
    if len(summary) > 0:
        print(f"\n📊 Summary:")
        print(f"   Total Suggestions: {summary.iloc[0]['total_suggestions']}")
        print(f"   Applied: {summary.iloc[0]['applied_count']}")
        print(f"   Unique Tasks: {summary.iloc[0]['unique_tasks']}")
