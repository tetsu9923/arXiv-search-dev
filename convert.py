import random
import torch
import numpy as np
import torch.nn as nn


#class Function(nn.Module):
#    def __init__(self):
#        super(Function, self).__init__()
#        self.fc = nn.Linear(768, 128)
#    def forward(self, x):
#        return self.fc(x)


#seed = 0
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)

embeddings = np.load('./data/embeddings.npy')

#div = 2
#func = Function().to('cpu')

new_embeddings = embeddings * (255/2)

#new_title_embeddings = []
#new_abst_embeddings = []

#for title, abst in zip(title_embeddings, abst_embeddings):
    #new_title = [sum(title[i*div:(i+1)*div])/div for i in range(len(title)//div)]
    #new_abst = [sum(abst[i*div:(i+1)*div])/div for i in range(len(abst)//div)]
    #new_title_embeddings.append(new_title)
    #new_abst_embeddings.append(new_abst)

    #new_title = func(torch.tensor(title))
    #new_abst = func(torch.tensor(abst))
    #new_title_embeddings.append(new_title.tolist())
    #new_abst_embeddings.append(new_abst.tolist())

#new_title_embeddings = np.array(new_title_embeddings)
#new_abst_embeddings = np.array(new_abst_embeddings)

print(new_embeddings.shape)

np.save('./data/embeddings_int8.npy', new_embeddings.astype(np.int8))