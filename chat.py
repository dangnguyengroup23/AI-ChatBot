import random
import json
import torch
from model import ANN
from nltk_utils import bagOfWord,tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
with open('intents.json','r') as f:
    intents = json.load(f)
FILE = 'data.pth'
data = torch.load(FILE)
inputSize = data['input_size']
hiddenSize = data['hidden_size']
outputSize = data['output_size']
allWords = data['all_words']
tags = data['tags']
model_state = data['model_state']


model = ANN(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(model_state)
model.eval()

botName = 'Tit'
print("Let's chat! type 'quit' to exit" )
while True:
    s = input('You: ')
    if s=='quit':
        break

    s = tokenize(s)
    X = bagOfWord(s,allWords)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).float()
    output = model(X)
    _, predict = torch.max(output,dim=1)
    tag = tags[predict.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predict.item()]

    if prob.item()>0.75:
        for i in intents['intents']:
            if tag == i['tag']:
                print(f"{botName}: {random.choice(i['responses'])}")
    else:
        print(f"{botName}: I do not understand...")

