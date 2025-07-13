🧠 AI Chatbot with PyTorch & Flask

A simple, full-stack AI chatbot application built using PyTorch for intent classification and Flask for serving a web-based frontend. It uses classic NLP techniques (tokenization, stemming, and bag-of-words) to understand user input and generate intent-based responses.

🚀 Features

1. ✅ Intent classification with 95%+ training accuracy
2. 🧠 NLP preprocessing using NLTK (tokenization + stemming)
3. 🧱 Custom ANN model built with PyTorch
4. 📦 SQLite database for storing chat logs
5. 🌐 Full-stack integration with Flask + HTML/CSS frontend
6. 🗂️ Easily extensible intent system via intents.json
7. 🤖 Basic sentiment-aware responses for empathy

🛠 Tech Stack
Backend: Python, PyTorch, Flask
Frontend: HTML, CSS
NLP: NLTK, NumPy
Database: SQLite (via sqlite3 in Python)
Data Format: JSON (for intent structure)

⚙️ Installation & Setup

1. Create a virtual environment 
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

2. Install dependencies
pip install -r requirements.txt

3. Download NLTK tokenizer data
import nltk
nltk.download('punkt')

4. Train the chatbot model
python train.py

5. Run the Flask app
python app.py

📈 Training Output
Input size: determined by vocab size from intents
Architecture: 3-layer ANN with ReLU activations and dropout
Loss: CrossEntropyLoss
Optimizer: Adam (learning rate = 0.001)
Final Training Accuracy: 95%+
Trained model saved to data.pth
