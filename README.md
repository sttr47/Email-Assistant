# Email Assistant

This app reads an email and tells you how urgent it is (Low, Medium or High).
For Medium and High emails, it also writes a suggested reply based on a knowledge base.
It works with English, Azerbaijani and Russian emails.

## Try it online (easiest way)

1. Click this button: (https://colab.research.google.com/github/sttr47/Email-Assistant/blob/main/demo.ipynb)
2. Click **Runtime → Run all**.
3. When asked, paste your Groq API key. You can get a free key at [console.groq.com](https://console.groq.com). If you leave it empty, the urgency result still works, but replies will not.
4. Wait a few minutes. The first run downloads the models.
5. At the end of the last cell, open the link that looks like `https://xxxx.gradio.live`.

Note: the link stops working when the Colab session ends.

## Run on your own computer

1. Download this repository.
2. Install the packages: `pip install -r requirements.txt` and `pip install torch`
3. Create a file named `.env` in the main folder:
   ```
   GROQ_API_KEY=your_key_here
   MODEL_NAME=openai/gpt-oss-20b
   ```
4. Start the app: `python app_ui.py`
5. Open the link shown in the terminal.

## How it works

1. A fine-tuned XLM-RoBERTa model decides the urgency of the email.
2. For Medium and High emails, the app searches `docs/knowledge_base.txt` (Qdrant + reranker).
3. A Groq language model writes the reply from the search results.
4. The interface is built with Gradio.

## Files

- `app_ui.py`: the web interface
- `main.py`: urgency model and reply logic
- `app/`: search and knowledge base code
- `docs/knowledge_base.txt`: the text used for replies
- `demo.ipynb`: the Colab notebook
