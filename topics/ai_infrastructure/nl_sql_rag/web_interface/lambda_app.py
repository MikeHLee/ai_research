import json
import os
import boto3
import requests
from botocore.exceptions import ClientError

# Configuration
API_ENDPOINT = os.environ.get('API_ENDPOINT')

def lambda_handler(event, context):
    """
    Lambda function handler for the web interface.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        dict: Response object with HTML content or API response
    """
    # Check if this is an API Gateway request
    if event.get('httpMethod'):
        # Handle API Gateway request
        path = event.get('path', '/')
        http_method = event.get('httpMethod')
        
        if path == '/' and http_method == 'GET':
            # Serve the HTML interface
            return serve_html_interface()
        elif path == '/query' and http_method == 'POST':
            # Process a query
            try:
                body = json.loads(event.get('body', '{}'))
                question = body.get('question')
                
                if not question:
                    return {
                        'statusCode': 400,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({'error': 'Missing required parameter: question'})
                    }
                
                # Forward the request to the query processor API
                response = requests.post(
                    API_ENDPOINT,
                    json={'question': question},
                    headers={'Content-Type': 'application/json'}
                )
                
                return {
                    'statusCode': response.status_code,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': response.text
                }
                
            except Exception as e:
                return {
                    'statusCode': 500,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': str(e)})
                }
        else:
            # Handle 404
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'text/plain'},
                'body': 'Not Found'
            }
    else:
        # Direct Lambda invocation
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Invalid request format'})
        }

def serve_html_interface():
    """
    Serve the HTML interface for the web application.
    
    Returns:
        dict: Response object with HTML content
    """
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Natural Language to SQL Query Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .chat-container {{ height: 60vh; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .user-message {{ background-color: #f1f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; margin-left: auto; }}
            .assistant-message {{ background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            code {{ color: #d63384; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Natural Language to SQL Query Engine</h1>
            <p class="lead">Ask questions about your data in natural language, and get answers with the corresponding SQL query and raw results.</p>
            
            <div class="row mb-3">
                <div class="col">
                    <h5>Example questions:</h5>
                    <ul>
                        <li>What are the top 5 products by sales?</li>
                        <li>How many customers made purchases last month?</li>
                        <li>What is the average order value by region?</li>
                    </ul>
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer"></div>
            
            <div class="row">
                <div class="col">
                    <div class="input-group mb-3">
                        <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about your data..." aria-label="Question">
                        <button class="btn btn-primary" type="button" id="sendButton">Send</button>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <button class="btn btn-secondary" id="clearButton">Clear Chat</button>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const chatContainer = document.getElementById('chatContainer');
                const questionInput = document.getElementById('questionInput');
                const sendButton = document.getElementById('sendButton');
                const clearButton = document.getElementById('clearButton');
                
                // Function to add a message to the chat
                function addMessage(content, isUser) {{
                    const messageDiv = document.createElement('div');
                    messageDiv.className = isUser ? 'user-message' : 'assistant-message';
                    
                    if (isUser) {{
                        messageDiv.textContent = content;
                    }} else {{
                        // Process markdown-like formatting
                        let formattedContent = content
                            .replace(/### ([^\n]+)/g, '<h5>$1</h5>')
                            .replace(/```sql\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                            .replace(/```\n([\s\S]*?)```/g, '<pre>$1</pre>');
                        
                        messageDiv.innerHTML = formattedContent;
                    }}
                    
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }}
                
                // Function to send a query
                async function sendQuery(question) {{
                    try {{
                        addMessage(question, true);
                        
                        // Show loading message
                        const loadingDiv = document.createElement('div');
                        loadingDiv.className = 'assistant-message';
                        loadingDiv.textContent = 'Processing your query...';
                        chatContainer.appendChild(loadingDiv);
                        
                        const response = await fetch('/query', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify({{
                                question: question
                            }})
                        }});
                        
                        // Remove loading message
                        chatContainer.removeChild(loadingDiv);
                        
                        if (response.ok) {{
                            const result = await response.json();
                            
                            if (result.success) {{
                                const formattedResponse = `### Summary\n${{result.summary}}\n\n### SQL Query\n```sql\n${{result.sql_query}}\n```\n\n### Raw Results\n```\n${{result.raw_results}}\n````;
                                addMessage(formattedResponse, false);
                            }} else {{
                                const errorMessage = `### Error\n${{result.summary || 'An error occurred while processing your query.'}}`;
                                addMessage(errorMessage, false);
                            }}
                        }} else {{
                            const errorText = await response.text();
                            const errorMessage = `### Error\nError: ${{response.status}} - ${{errorText}}`;
                            addMessage(errorMessage, false);
                        }}
                    }} catch (error) {{
                        addMessage(`### Error\nAn error occurred: ${{error.message}}`, false);
                    }}
                }}
                
                // Event listeners
                sendButton.addEventListener('click', function() {{
                    const question = questionInput.value.trim();
                    if (question) {{
                        sendQuery(question);
                        questionInput.value = '';
                    }}
                }});
                
                questionInput.addEventListener('keypress', function(event) {{
                    if (event.key === 'Enter') {{
                        const question = questionInput.value.trim();
                        if (question) {{
                            sendQuery(question);
                            questionInput.value = '';
                        }}
                    }}
                }});
                
                clearButton.addEventListener('click', function() {{
                    chatContainer.innerHTML = '';
                }});
            }});
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html
    }
