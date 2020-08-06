import os
import torch
import numpy as np
import pickle
import csv
from src.utils import convert_tensor_to_texts

def sample_from_vae(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    sample_num = config['sample_num']
    sample_save_path = os.path.join(base_path, 'sample%d.tsv' % sample_num)
    save_encoding = config['save_encoding']

    with open(vocab_path, 'rb') as handle:
        vocab = pickle.load(handle)

    model = torch.load(save_path)

    batch_size = config['batch_size']
    batch_sizes = [batch_size] * (sample_num // batch_size) + [sample_num % batch_size]

    sentences = ['sentence']

    if save_encoding:
        encoding = []

    for batch_size in batch_sizes:
        if save_encoding:
            output, output_encoding = model.sample(num=batch_size, output_encoding=True)
            encoding.append(output_encoding)
        else:
            output = model.sample(num=batch_size)
        sentences.extend(convert_tensor_to_texts(output, vocab))

    sentences = [[sentence] for sentence in sentences]
    if save_encoding:
        encoding = np.concatenate(encoding, axis=0)
        encoding_save_path = '.'.join(sample_save_path.split('.')[0:-1]) + '.npy'
        np.save(encoding_save_path, encoding)

    with open(sample_save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sentences)