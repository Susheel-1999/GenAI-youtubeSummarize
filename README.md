# GenAI- Youtube Summarize
App that takes a YouTube video URL, extracts its transcript, and generates a consolidated summary using LangChain’s Map-Reduce strategy and Groq’s Llama model.

# Steps to run Streamlit app:
1. Create a groq API or use an existing one, visit: https://console.groq.com/keys
2. Create a new environment: <br>
```conda create -p genai python==3.10 -y```
3. Activate the environment: <br>
```conda activate genai```
4. Install the requirements: <br>
```pip install -r requirements.txt```
5. Run app.py: <br>
```python app.py```

