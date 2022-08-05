import argparse
import pickle

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def Dim_reduction(sentences, tokenizer, model, device, mode):
    vecs = []
    with torch.no_grad():
        for sentence in tqdm(sentences):
            inputs = tokenizer(sentence, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            #inputs['input_ids'] = inputs['input_ids'].to(device)
            #inputs['attention_mask'] = inputs['attention_mask'].to(device)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)

    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :128]

    np.save("./data/{}_kernel.npy".format(mode), kernel)
    np.save("./data/{}_bias.npy".format(mode), bias)

    embeddings = []
    embeddings = np.vstack(vecs)
    embeddings = transform_and_normalize(
        embeddings, 
        kernel=kernel,
        bias=bias
    )
    return embeddings

def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    with open('./data/raw_title.pkl', 'rb') as f:
        title_list = pickle.load(f)
    with open('./data/raw_abst.pkl', 'rb') as f:
        abst_list = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)

    with torch.no_grad():
        title_embeddings = Dim_reduction(title_list, tokenizer, model, device, "title")
    print(title_embeddings.shape)

    with torch.no_grad():
        abst_embeddings = Dim_reduction(abst_list, tokenizer, model, device, "abst")
    print(abst_embeddings.shape)

    np.save('./data/title_embeddings_whiten.npy', title_embeddings)
    np.save('./data/abst_embeddings_whiten.npy', abst_embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    main(args)