🧠 AI Chatbot with PyTorch & Flask

A simple, full-stack AI chatbot application built using PyTorch for intent classification and Flask for serving a web-based frontend. It uses classic NLP techniques (tokenization, stemming, and bag-of-words) to understand user input and generate intent-based responses.

🚀 Features
✅ Intent classification with >95% training accuracy
🧠 NLP preprocessing with NLTK (tokenization, stemming)
📦 Data persistence using SQLite for storing conversation logs
🧱 Bag-of-Words vectorization + custom ANN (PyTorch)
🌐 Full-stack integration with Flask and HTML/CSS frontend
🗂️ Easily extensible intent system via intents.json
🤖 Smart, empathetic response logic using basic sentiment awareness

🛠 Tech Stack
Backend: Python, PyTorch, Flask
Frontend: HTML, CSS
NLP: NLTK, NumPy
Database: SQLite (via sqlite3 in Python)
Data Format: JSON (for intent structure)

⚙️ Installation & Setup

2. Create a virtual environment 
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies
pip install -r requirements.txt

4. Download NLTK tokenizer data
import nltk
nltk.download('punkt')

5. Train the chatbot model
python train.py

6. Run the Flask app
python app.py

📈 Training Output
Input size: determined by vocab size from intents
Architecture: 3-layer ANN with ReLU activations and dropout
Loss: CrossEntropyLoss
Optimizer: Adam (learning rate = 0.001)
Final Training Accuracy: 95%+
Trained model saved to data.pth
