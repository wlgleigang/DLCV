import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn=nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,drop_prob=0.2):
        super(DecoderRNN, self).__init__()
        self.caption_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers,dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        caption_embeds = self.caption_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1),caption_embeds),1)
        out, hidden = self.lstm(inputs)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    def init_weights(self):
        self.fc.bias.data.fill_(0.01)
        torch.nn.init.xavier_normal_(self.fc.weight)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            _, predicted = out.max(1) 
            tokens.append(predicted.item())
            inputs = self.caption_embeddings(predicted) 
            inputs = inputs.unsqueeze(1)
        return tokens