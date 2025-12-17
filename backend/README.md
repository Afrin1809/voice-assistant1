1. Ensure your questions file is at /mnt/data/questions.csv (uploaded)
   - CSV must have 'question' and 'answer' columns (case-insensitive).
2. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
3. Run:
   python app.py
4. Open browser at: http://127.0.0.1:5000/
5. Click Start Listening (allow microphone). Ask a question.
