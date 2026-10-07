import json
import ollama
from pydantic import BaseModel
from typing import List, Optional

# Default model, optimized for Apple Silicon (M2 Pro, 16GB RAM)
DEFAULT_MODEL = "qwen2.5-coder:7b"

class Exercise(BaseModel):
    concept: str
    description: str
    starter_code: str
    test_assertions: List[str]

def pull_model_if_missing(model_name: str = DEFAULT_MODEL):
    """Pulls the model from Ollama if it isn't already downloaded."""
    try:
        ollama.show(model_name)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"Downloading model {model_name}. This may take a minute...")
            ollama.pull(model_name)
        else:
            raise

def generate_exercise(concept: str, level: int, model_name: str = DEFAULT_MODEL) -> Optional[Exercise]:
    """
    Prompts Ollama to generate a Python programming exercise based on a specific concept.
    Returns an Exercise Pydantic model.
    """
    prompt = f"""
You are an expert Python tutor for an ABSOLUTE BEGINNER. 
The student is currently at Level {level} out of 20. 
They are learning about: "{concept}".

CRITICAL RULES:
1. Make the task EXTREMELY SIMPLE. If they are learning 'print()', just ask them to print a single word. Do NOT ask for user input. Do NOT combine concepts.
2. Provide a short, 1-sentence description of the task. IF YOUR TEST EXPECTS A SPECIFIC STRING OR VARIABLE NAME, YOU MUST EXPLICITLY TELL THE STUDENT IN THE DESCRIPTION.
3. Provide starter code IF the task requires the user to modify an existing variable (e.g., if you ask them to use `x`, you must provide `x = 5` as starter code).
4. Provide 1 to 3 Python `assert` statements that will be run AFTER the user's code to test if they solved it correctly. You can test variables they created or their printed output.
5. TO TEST PRINTED OUTPUT: You MUST use the special string variable `__stdout__` (this is a regular Python string, not a stream). ONLY use standard string methods like .strip() or ==. Do NOT call .reset() or any stream methods. Example: `assert __stdout__.strip() == 'hello'`

Respond strictly in JSON format matching this schema:
{{
    "concept": "The concept being taught",
    "description": "The extremely simple prompt explaining the task",
    "starter_code": "Any initial code to give the user, or an empty string",
    "test_assertions": ["assert my_var == 5", "assert __stdout__.strip() == 'hello'"]
}}
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful Python programming tutor that only outputs valid JSON.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            format=Exercise.model_json_schema()
        )
        
        # Parse the JSON response into our Pydantic model
        exercise_data = json.loads(response['message']['content'])
        return Exercise(**exercise_data)
        
    except Exception as e:
        print(f"Error generating exercise: {e}")
        return None

def generate_hint(exercise: Exercise, user_code: str, error_message: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Prompts Ollama to generate a minimal hint based on the user's failed code.
    Returns a string containing the hint.
    """
    prompt = f"""
The student is trying to solve this Python exercise: "{exercise.description}"

Here is the code they wrote:
```python
{user_code}
```

When tested, it produced this error/failure:
{error_message}

Provide a MAXIMUM 1-sentence hint that points them in the right direction. 
DO NOT give them the answer code. 
DO NOT tell them exactly what to type.
Just explain what went wrong conceptually or point out where to look.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful but strict Python tutor. You provide extremely short, 1-sentence hints.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response['message']['content'].strip()
        
    except Exception as e:
        return "Hint generation failed. Look closely at the error message!"

def generate_lesson(concept: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Prompts Ollama to generate a simple lesson introducing a new concept.
    """
    prompt = f"""
You are a Python tutor for an absolute beginner.
Introduce the concept: "{concept}".

Keep the explanation EXTREMELY simple, conversational, and very short (3-4 sentences max).
Include a very small code example. Do not use complex jargon.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a friendly, encouraging Python tutor. Your explanations are tiny and simple.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response['message']['content'].strip()
        
    except Exception as e:
        return "New concept! Pay attention to the instructions."
