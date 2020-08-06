from src.constants import PAD_INDEX, EOS_INDEX

def sentence_clip(sentence):
    mask = (sentence != PAD_INDEX)
    sentence_lens = mask.long().sum(dim=1, keepdim=False)
    max_len = sentence_lens.max().item()
    return sentence[:, :max_len]

def convert_tensor_to_texts(tensor, vocab):
    f = lambda line: ' '.join([vocab.itos[index] for index in line])
    indices = tensor.tolist()
    texts = []
    for line in indices:
        if EOS_INDEX in line:
            eos_position = line.index(EOS_INDEX)
            line = line[0: eos_position]
        texts.append(f(line))
    return texts