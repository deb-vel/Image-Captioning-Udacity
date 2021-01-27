import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()
    
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        captions = self.embed(captions)
        
        features = features.unsqueeze(1)
        concat_inputs = torch.cat((features, captions),1)
        hidden, _ = self.lstm(concat_inputs)
        output = self.fc1(hidden)
        return output
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sent_pred = []
        
        for i in range(max_len):
            out, states = self.lstm(inputs, states) #pass through lstm
            out = self.fc1(out) #pass through linear layer
          
            _, word = out.max(2) #get the maximum
            
            word_item = word.item()
            sent_pred.append(word_item) #build sentence
            
            inputs = self.embed(word) #update input for next iteration
            
        return sent_pred