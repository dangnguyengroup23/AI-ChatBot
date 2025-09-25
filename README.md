## 🧠 AI Chatbot with PyTorch & Flask

This is a simple AI chatbot application that uses PyTorch and Flask. PyTorch helps the chatbot understand what you mean, and Flask helps it show you a friendly web page. It's like teaching a computer to understand and chat with you!

We use some common tricks from the world of computers and language (like tokenization, stemming, and bag-of-words) to help the chatbot understand what you're saying and give you the best answer.

🚀 Features

Here's what the chatbot can do:

1. ✅ **Intent Classification:** It's really good at figuring out what you *intend* to say (over 95% accuracy after training!).
2. 🧠 **NLP Preprocessing:** It uses NLTK to break down your sentences into smaller parts (tokenization) and simplify words (stemming).
3. 🧱 **Custom ANN Model:** We built a special brain (ANN - Artificial Neural Network) for the chatbot using PyTorch.
4. 📦 **Chat Log Storage:** It remembers your conversations using a simple SQLite database.
5. 🌐 **Full-Stack Integration:** It's a complete web application with a nice-looking frontend built with Flask, HTML, and CSS.
6. 🗂️ **Easy to Extend:** You can easily teach it new things by adding new intents to the `intents.json` file.
7. 🤖 **Sentiment Awareness:** It can understand your mood a little bit and respond with some empathy.

🛠 Tech Stack

Here are the tools we used to build it:

*   **Backend:** Python, PyTorch, Flask
*   **Frontend:** HTML, CSS
*   **NLP:** NLTK, NumPy
*   **Database:** SQLite (using `sqlite3` in Python)
*   **Data Format:** JSON (for organizing the chatbot's knowledge)

⚙️ Installation & Setup

Follow these steps to get the chatbot up and running:

1.  **Create a virtual environment:** This helps keep things organized.

    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```

2.  **Install the necessary libraries:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download some data for NLTK:**

    ```python
    import nltk
    nltk.download('punkt')
    ```

4.  **Train the chatbot's brain:** This teaches it how to understand language.

    ```bash
    python train.py
    ```

5.  **Run the web application:** This starts the chatbot so you can talk to it.

    ```bash
    python app.py
    ```

📈 Training Output

When you train the chatbot, you'll see something like this:

*   **Input size:** Depends on how many different words the chatbot knows.
*   **Architecture:** It's like a 3-layer brain with special switches (ReLU activations) and some randomness (dropout).
*   **Loss:** We use a special formula (CrossEntropyLoss) to measure how well the chatbot is learning.
*   **Optimizer:** We use a tool called Adam to help the chatbot learn faster (learning rate = 0.001).
*   **Final Training Accuracy:** After training, the chatbot should be able to understand your intents with 95%+ accuracy.
*   **Trained model saved to:** `data.pth` (This is where the chatbot's learned knowledge is stored.)
