import os
import torch
from torch import nn, optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import logging
import pickle
import math
from src.model.text_vae import TextVAE
from src.constants import PAD_INDEX, UNK, PAD, SOS, EOS
from src.train.eval import eval_text_vae
from src.gaussian_kldiv import GaussianKLDiv

def train_vae(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)

    base_path = config['base_path']
    save_path = os.path.join(base_path, 'vae.pkl')
    vocab_path = os.path.join(base_path, 'vocab.pkl')

    logger.info('build dataset')

    TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
    fields = [('sentence', TEXT)]
    train_data = TabularDataset(path=os.path.join(base_path, 'train.tsv'),
                                format='tsv', skip_header=True, fields=fields)
    dev_data = TabularDataset(path=os.path.join(base_path, 'dev.tsv'),
                              format='tsv', skip_header=True, fields=fields)

    logger.info('build vocabulary')
    TEXT.build_vocab(train_data, specials=[UNK, PAD, SOS, EOS])
    vocab = TEXT.vocab
    vocab_size = len(vocab.itos)
    logger.info('vocab_size: %d' % vocab_size)
    logger.info('save vocabulary')
    with open(vocab_path, 'wb') as handle:
        pickle.dump(vocab, handle)

    logger.info('build data iterator')
    device = torch.device('cuda:0')
    train_iter = Iterator(train_data, batch_size=config['batch_size'], shuffle=True, device=device)
    dev_iter = Iterator(dev_data, batch_size=config['batch_size'], shuffle=False, device=device)

    logger.info('build model')
    model = TextVAE(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        enc_dec_tying=config['enc_dec_tying'],
        dec_gen_tying=config['dec_gen_tying']
    )

    logger.info('transfer model to GPU')
    model = model.to(device)

    logger.info('set up criterion and optimizer')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    kldiv = GaussianKLDiv()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    logger.info('start train')

    min_dev_loss = 1e9
    corr_dev_wer = 1

    for epoch in range(config['epoches']):

        total_tokens = 0
        correct_tokens = 0
        total_loss = 0

        for i, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit, mean, std = model(src, trg_input)
            trg_output = trg_output.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, trg_output) + kldiv(mean, std) * config['lambd']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()

            mask = (trg_output != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

            if i % config['eval_freq'] == 0:
                train_loss = total_loss / total_tokens
                train_wer = 1 - correct_tokens / total_tokens
                total_loss = 0
                correct_tokens = 0
                total_tokens = 0
                dev_loss, dev_wer = eval_text_vae(model, dev_iter, criterion)
                logger.info('[epoch %2d step %4d]\ttrain_ppl: %.4f train_wer: %.4f dev_ppl: %.4f dev_wer: %.4f' %
                            (epoch, i, math.exp(train_loss), train_wer, math.exp(dev_loss), dev_wer))
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    corr_dev_wer = dev_wer
                    torch.save(model, save_path)

    logger.info('dev_ppl: %.4f\tdev_wer: %.4f' % (math.exp(min_dev_loss), corr_dev_wer))
    logger.info('finish')