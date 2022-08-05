import argparse
import pickle

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def batch(list1, list2, batch_size=1):
    l = len(list1)
    assert l == len(list2)
    for ndx in range(0, l, batch_size):
        yield list1[ndx:min(ndx + batch_size, l)], list2[ndx:min(ndx + batch_size, l)]


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    with open('./data/raw_title.pkl', 'rb') as f:
        title_list = pickle.load(f)
    with open('./data/raw_abst.pkl', 'rb') as f:
        abst_list = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)

    for i, (title, abst) in enumerate(batch(title_list, abst_list, batch_size=batch_size)):
        with torch.no_grad():
            _input = tokenizer(title, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            output = model(**_input).pooler_output
            output = output.cpu()
    
            if i == 0:
                title_embeddings = output
            else:
                title_embeddings = torch.cat((title_embeddings, output), dim=0)

            del _input
            del output
            torch.cuda.empty_cache()
        print(batch_size*(i+1))

    for i, (title, abst) in enumerate(batch(title_list, abst_list, batch_size=batch_size)):
        with torch.no_grad():
            _input = tokenizer(abst, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            output = model(**_input).pooler_output
            output = output.cpu()
    
            if i == 0:
                abst_embeddings = output
            else:
                abst_embeddings = torch.cat((abst_embeddings, output), dim=0)

            del _input
            del output
            torch.cuda.empty_cache()
        print(batch_size*(i+1))

    np.save('./data/title_embeddings.npy', title_embeddings.cpu().detach().numpy())
    np.save('./data/abst_embeddings.npy', abst_embeddings.cpu().detach().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    main(args)