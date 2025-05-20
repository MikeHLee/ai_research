import gradio as gr
import requests
import json
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_ENDPOINT = os.environ.get('API_ENDPOINT', 'http://localhost:8000/query')

def process_query(question, history):
    """
    Process a natural language query and return the results.
    
    Args:
        question: The natural language question
        history: Chat history
        
    Returns:
        tuple: (history, response)
    """
    try:
        # Send request to API
        response = requests.post(
            API_ENDPOINT,
            json={'question': question},
            headers={'Content-Type': 'application/json'}
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            
            # Format response
            if result.get('success', False):
                sql_query = result.get('sql_query', '')
                summary = result.get('summary', '')
                raw_results = result.get('raw_results', '')
                
                # Create formatted response
                formatted_response = f"""### Summary
{summary}

### SQL Query
```sql
{sql_query}
```

### Raw Results
```
{raw_results}
```"""
                
                return history + [[question, formatted_response]]
            else:
                error_message = result.get('summary', 'An error occurred while processing your query.')
                return history + [[question, f"### Error\n{error_message}"]]
        else:
            error_message = f"Error: {response.status_code} - {response.text}"
            return history + [[question, f"### Error\n{error_message}"]]
            
    except Exception as e:
        return history + [[question, f"### Error\nAn error occurred: {str(e)}"]]

# Create Gradio interface
with gr.Blocks(title="Natural Language to SQL Query Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Natural Language to SQL Query Engine
    
    Ask questions about your data in natural language, and get answers with the corresponding SQL query and raw results.
    
    **Example questions:**
    - What are the top 5 products by sales?
    - How many customers made purchases last month?
    - What is the average order value by region?
    """)
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=500,
        show_copy_button=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about your data...",
            show_label=False,
            container=False,
            scale=9
        )
        submit = gr.Button("Send", scale=1)
    
    clear = gr.Button("Clear")
    
    # Set up event handlers
    msg.submit(process_query, [msg, chatbot], [chatbot]).then(
        lambda: "", None, msg
    )
    
    submit.click(process_query, [msg, chatbot], [chatbot]).then(
        lambda: "", None, msg
    )
    
    clear.click(lambda: [], None, chatbot)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
