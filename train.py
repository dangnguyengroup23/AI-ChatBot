import json
from nltk_utils import tokenize,stem,bagOfWord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ANN

from torch.utils.data import DataLoader,Dataset



with open('intents.json','r') as f:
    intents = json.load(f)


allWords= []

tags = []

xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w,tag))

ignoreWords=['?','!','.',',']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
allWords = sorted(set(allWords))

tags = sorted(set(tags))

X_train = []
y_train = []

for (patternSentence, tag) in xy:
    bag = bagOfWord(patternSentence,allWords)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)

class ChatDataSet(Dataset):
    def __init__(self):
        super().__init__()

        self.n_samples = len(X_train)
        self.x_data  = X_train
        self.y_data  = y_train
    def __getitem__(self,index):
        return torch.tensor(self.x_data[index],dtype=torch.float),torch.tensor(self.y_data[index],dtype=torch.long)
    def __len__(self):
        return self.n_samples
    

batch_size = 8
hiddenSize = 32
outputSize = len(tags)
inputSize = len(X_train[0])
learningRate = 0.001
epochs = 1000

dataset = ChatDataSet()
print(dataset)
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = ANN(inputSize, hiddenSize, outputSize).to(device)

lossFunct = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
model.train()
for i in range(epochs):
    total=0
    correct=0
    for words,labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        yHat = model(words)
        loss = lossFunct(yHat,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(yHat, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    if(i+1) % 100==0:
        print(f'epoch {i+1}/{epochs},loss = {loss.item():.4f},accuracy = {acc:.2f}%')
print(f'final Accuracy: {acc:.2f}%')

data = {
    "model_state" : model.state_dict(),
    "input_size":inputSize,
    "output_size": outputSize,
    "hidden_size": hiddenSize,
    "all_words" : allWords,
    "tags":tags
}
FILE =  "data.pth"

torch.save(data,FILE)

print(f'training complete,file saved to {FILE}') 