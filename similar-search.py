import argparse
import pickle
import random

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main(args):
    top_n = args.top_n
    query_title = args.title
    query_abst = args.abst
    use_title = args.title != ''
    use_abst = args.abst != ''
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    model = model.to('cpu')

    with open('./data/raw_title.pkl', 'rb') as f:
        title_list = pickle.load(f)
    with open('./data/raw_abst.pkl', 'rb') as f:
        abst_list = pickle.load(f)
    with open('./data/raw_link.pkl', 'rb') as f:
        link_list = pickle.load(f)

    with torch.no_grad():
        query_input = tokenizer(query_title, return_tensors='pt').to('cpu')
        query1 = model(**query_input).pooler_output[0].cpu().numpy()
        query_input = tokenizer(query_abst, return_tensors='pt').to('cpu')
        query2 = model(**query_input).pooler_output[0].cpu().numpy()

        #kernel = np.load("./data/title_kernel.npy")
        #bias = np.load("./data/title_bias.npy")
        #inputs = tokenizer(query_title, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cpu')
        #hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states
        #output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        #query1 = transform_and_normalize(output_hidden_state.cpu().numpy()[0], kernel, bias).squeeze()

        #kernel = np.load("./data/abst_kernel.npy")
        #bias = np.load("./data/abst_bias.npy")
        #inputs = tokenizer(query_abst, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cpu')
        #hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states
        #output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        #query2 = transform_and_normalize(output_hidden_state.cpu().numpy()[0], kernel, bias).squeeze()

    title_embeddings = np.load('./data/title_embeddings_int8.npy')
    abst_embeddings = np.load('./data/abst_embeddings_int8.npy')

    title_embeddings = title_embeddings * (2/255)
    abst_embeddings = abst_embeddings * (2/255)

    print(query1)
    print(title_embeddings[0])

    if use_title and use_abst:
        query = np.concatenate([query1, query2])
        embeddings = np.concatenate([title_embeddings, abst_embeddings], axis=1)
    elif use_title:
        query = query1
        embeddings = title_embeddings
    elif use_abst:
        query = query2
        embeddings = abst_embeddings
    else:
        raise ValueError

    sim_list = []
    for vector in embeddings:
        sim_list.append(cos_similarity(query, vector))
        
    sim_list = np.array(sim_list)
    sim_idx = np.argsort(sim_list)[::-1]
    for i in range(top_n):
        print('Similarlity: {}'.format(sim_list[sim_idx[i]]))
        print('Title: {}'.format(title_list[sim_idx[i]]))
        print('Link: {}'.format(link_list[sim_idx[i]]))
        print('Abstract: \n{}'.format(abst_list[sim_idx[i]]))

    with open('./data/results.txt', mode='w') as f:
        for i in range(top_n):
            f.write('Similarlity: {}\n'.format(sim_list[sim_idx[i]]))
            f.write('Title: {}\n'.format(title_list[sim_idx[i]]))
            f.write('Link: {}\n'.format(link_list[sim_idx[i]]))
            f.write('Abstract: \n{}\n\n'.format(abst_list[sim_idx[i]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--abst', type=str, default='')
    args = parser.parse_args()
    main(args)