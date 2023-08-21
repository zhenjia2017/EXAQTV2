import yaml
import torch
import os
import json
import pickle
import nltk
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from operator import itemgetter, attrgetter
import warnings


def get_config(config_path):
    print ("\nconfig_path:", config_path)
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def use_cuda(var):
    if torch.cuda.is_available():
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #print ("\n\ntorch.cuda.device_count()")
        #print (torch.cuda.device_count())
        return var.cuda()
    else:
        return var

def save_model(the_model, path):
    if os.path.exists(path):
        path = path + '_copy'
    print("saving model to ...", path)
    torch.save(the_model, path)

def load_model(path):
    if not os.path.exists(path):
        assert False, 'cannot find model: ' + path
    print("loading model from ...", path)
    return torch.load(path)

def load_date(filename):
    dates = []
    with open(filename) as f_in:
        for line in f_in:
            date_range = dict()
            date = line.strip().replace("T00: 00:00Z","")
            if date.startswith("-"):
                year = date.split("-")[1]
                month = date.split("-")[2]
                day = date.split("-")[3]
            else:
                year = date.split("-")[0]
                month = date.split("-")[1]
                day = date.split("-")[2]
            date_range['year'] = int(year)
            date_range['month'] = int(month)
            date_range['day'] = int(day)
            dates.append(date_range)

        specs = (('year', False), ('month', False), ('day', False))
        for key, reverse in reversed(specs):
            dates.sort(key=itemgetter(key), reverse=reverse)

    return dates

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            #word = line.strip().decode('UTF-8')
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def read_padded(my_lstm, document_emb, document_mask):
    """
    this function take an embedded array, pack it, read, and pad it.
    in order to use Packed_Sequence, we should sort by length, and then reverse to the original order
    :document_emb: num_document, max_document_word, hidden_size
    :document_mask: num_document, max_document_word
    :my_lstm: lstm
    """
    num_document, max_document_word, _ = document_emb.size()
    hidden_size = my_lstm.hidden_size
    document_lengths = torch.sum(document_mask, dim=1).type('torch.IntTensor') # num_document
    document_lengths, perm_idx = document_lengths.sort(0, descending=True)
    document_emb = document_emb[use_cuda(perm_idx)]
    inverse_perm_idx = [0] * len(perm_idx)
    for i, idx in enumerate(perm_idx):
        inverse_perm_idx[idx.data[0]] = i
    inverse_perm_idx = torch.LongTensor(inverse_perm_idx)

    document_lengths_np = document_lengths.data.cpu().numpy()
    document_lengths_np[document_lengths_np == 0] = 1 # skip warning: length could be 0

    num_layer = 2 if my_lstm.bidirectional else 1
    hidden = (use_cuda(Variable(torch.zeros(num_layer, num_document, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, num_document, hidden_size))))

    document_emb = pack_padded_sequence(document_emb, document_lengths_np, batch_first=True) # padded array
    document_emb, hidden = my_lstm(document_emb, hidden) # [batch_size * max_relevant_doc, max_document_word, entity_dim * 2] [2, batch_size * max_relevant_doc, entity_dim]
    document_emb, _ = pad_packed_sequence(document_emb, batch_first=True)

    document_emb = document_emb[use_cuda(inverse_perm_idx)]
    hidden = (hidden[0][:, use_cuda(inverse_perm_idx), :], hidden[1][:, use_cuda(inverse_perm_idx), :])
    batch_max_document_word = document_emb.size()[1]
    if batch_max_document_word < max_document_word:
        all_zeros = use_cuda(Variable(torch.zeros((num_document, max_document_word, hidden_size * num_layer))))
        all_zeros[:, : batch_max_document_word, :] = document_emb
        document_emb = all_zeros

    assert (num_document, max_document_word, hidden_size * num_layer) == document_emb.size()

    return document_emb, hidden

def load_documents(document_file):
    print('loading document from', document_file)
    documents = dict()
    with open(document_file) as f_in:
        for line in tqdm(f_in):
            passage = json.loads(line)
            # tokenize document
            document_token = nltk.word_tokenize(passage['document']['text'])
            #if 'title' in passage:
            #    title_token = nltk.word_tokenize(passage['title']['text'])
            #    passage['tokens'] = document_token + ['|'] + title_token
            #else:
            passage['tokens'] = document_token
            documents[int(passage['documentId'])] = passage
    return documents

def index_document_entities(documents, word2id, entity2id, max_document_word):
    print('indexing documents ...')
    document_entity_indices = dict()
    document_texts = dict()
    document_texts[-1] = np.full(max_document_word, len(word2id), dtype=int)
    for next_id, document in tqdm(documents.items()):
        global_entity_ids = []
        word_ids = []
        word_weights = []
        if 'title' in document:
            for entity in document['title']['entities']:
                entity_len = entity['end'] - entity['start']
                global_entity_ids += [entity2id[entity['text']]] * entity_len
                word_ids += range(entity['start'], entity['end'])
                word_weights += [1.0 / entity_len] * entity_len
            title_len = len(nltk.word_tokenize(document['title']['text']))
        else:
            title_len = 0
        for entity in document['document']['entities']:
            # word_ids are off by (title_len + 1) because document is concatenated after title, and with an extra '|'
            if entity['start'] + title_len + 1 >= max_document_word:
                continue
            entity_len = min(max_document_word, entity['end'] + title_len + 1) - (entity['start'] + title_len + 1)
            global_entity_ids += [entity2id[entity['text']]] * entity_len
            word_ids += range(entity['start'] + title_len + 1, entity['start'] + title_len + 1 + entity_len)
            if entity_len != 0:
                word_weights += [1.0 / entity_len] * entity_len

        assert len(word_weights) == len(word_ids)
        document_entity_indices[next_id] = (global_entity_ids, word_ids, word_weights)

        one_doc_text = np.full(max_document_word, len(word2id), dtype=int)
        for t, token in enumerate(document['tokens']):
            if t < max_document_word:
                if token in word2id:
                    one_doc_text[t] = word2id[token]
                else:
                    one_doc_text[t] = word2id['__unk__']

        document_texts[next_id] = one_doc_text

    return document_entity_indices, document_texts

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)

def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        f_pred.write(json.dumps({'dist': output_dist, 'answers':answers, 'seeds': data_loader.data[data_id]['entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

class LeftMMFixed(torch.autograd.Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """
    # def __init__(self):
    #     super(LeftMMFixed, self).__init__()
    #     self.sparse_weights = None

    @staticmethod
    def forward(ctx, sparse_weights, x):
        # if self.sparse_weights is None:
        #     self.sparse_weights = sparse_weights
        ctx.save_for_backward(sparse_weights, x)
        output = torch.mm(sparse_weights, x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_weights, x = ctx.saved_variables
        #sparse_weights = self.sparse_weights
        return None, torch.mm(sparse_weights.t(), grad_output)

def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    # print ("\n\nI.size()",I.size())
    # print ("\n\nB",B)
    # print("\n\nM", M)
    # print("\n\nN", N)
    #entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
    lookup = Y[I[0, :], I[2, :], :]
    #print("\n\nlookup", lookup)
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.cuda.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
    #S = use_cuda(Variable(torch.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))

    prod_op = LeftMMFixed.apply
    prod = prod_op(S, lookup)
    warnings.filterwarnings("ignore")
    # print("S.size", S.size())
    # print ("prod.size", prod.size())
    return prod.view(B, M, K)


# if __name__  == "__main__":
#     load_documents('datasets/wikimovie/full_doc/documents.json')
