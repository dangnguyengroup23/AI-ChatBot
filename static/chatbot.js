const chatForm = document.getElementById("chat-form");
const chatLog = document.getElementById("chat-log");
const userInput = document.getElementById("user-input");
const startBtn = document.getElementById("start-btn");
const chatContainer = document.getElementById("chat-container");
const startScreen = document.getElementById("start-screen");


async function getBotResponse(message) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message: message })
  });

  const data = await response.json();
  return data.reply;
}

function addMessage(text, sender = "bot") {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.innerText = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}


startBtn.addEventListener("click", () => {
  startScreen.style.display = "none";
  chatContainer.style.display = "flex";
  addMessage("👋 Hello! I'm Tit. How can I help you today?", "bot");
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value;
  addMessage(text, "user");

  const reply = await getBotResponse(text);
  addMessage(reply, "bot");

  userInput.value = "";
});
