#import sys
import torch
from tqdm import tqdm
#import numpy as np
import json
import os

from exaqt.graftnet.graftnet_baseline import GRAFTNET
from exaqt.answer_predict.script_listscore import compare_pr
from exaqt.graftnet.data_loader_baseline import DataLoader
from exaqt.graftnet.util import use_cuda, sparse_bmm, get_config, load_dict, cal_accuracy, load_date
import yaml


def data_load(cfg, train_subg, dev_subg, test_subg, entity2id, word2id, relation2id):
    train_data = DataLoader(train_subg, word2id, relation2id, entity2id, cfg['max_query_word'])

    valid_data = DataLoader(dev_subg, word2id, relation2id, entity2id, cfg['max_query_word'])

    test_data = DataLoader(test_subg, word2id, relation2id, entity2id, cfg['max_query_word'])

    return train_data, valid_data, test_data

def train(cfg, train_data, valid_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file):
    print("training ...")
    # prepare data
    # entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    # word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    # relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    # create model & set parameters
    my_model = get_model(cfg, entity_emb_file, vocab_emb_file, relation_emb_file, train_data.num_kb_relation, len(entity2id), len(word2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])
    save_model_file = os.path.join(cfg['model_folder'], cfg['save_model_file'])
    best_dev_acc = 0.0
    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential = cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_acc, train_max_acc = [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                loss, pred, _ = my_model(batch)
                #break
                pred = pred.data.cpu().numpy()
                acc, max_acc = cal_accuracy(pred, batch[-1])
                #train_loss.append(loss.data[0])
                train_loss.append(loss.data)
                train_acc.append(acc)
                train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()

            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_acc', sum(train_acc) / len(train_acc))

            print("validating ...")
            eval_acc = inference(my_model, valid_data, entity2id, cfg)
            if eval_acc > best_dev_acc and cfg['to_save_model']:
               print("saving model to", save_model_file)
               torch.save(my_model.state_dict(), save_model_file)
               best_dev_acc = eval_acc

        except KeyboardInterrupt:
            break

    # Test set evaluation
    #print("evaluating on test")
    #print('loading model from ...', cfg['model_folder'] + cfg['save_model_file'])
    my_model.load_state_dict(torch.load(save_model_file))
    dev_acc = inference_test(my_model, valid_data, entity2id, cfg, log_info=True)
    return dev_acc

def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        f_pred.write(json.dumps({'dist': output_dist, 'id':data_loader.data[data_id]['id'], 'answers':answers, 'seeds': data_loader.data[data_id]['entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 20
    pred_file = os.path.join(cfg['model_folder'], cfg['pred_file'])
    if log_info:
        f_pred = open(pred_file, 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        acc, max_acc = cal_accuracy(pred, batch[-1])
        if log_info:
            output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
        #eval_loss.append(loss.data[0])
        eval_loss.append(loss.data)
        eval_acc.append(acc)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

def inference_test(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 1
    pred_file = os.path.join(cfg['model_folder'], cfg['pred_file'])
    if log_info:
        f_pred = open(pred_file, 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        acc, max_acc = cal_accuracy(pred, batch[-1])
        if log_info:
            output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
        #eval_loss.append(loss.data[0])
        eval_loss.append(loss.data)
        eval_acc.append(acc)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

def test(cfg, test_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file):
    print("testing ...")

    #test_data = DataLoader(cfg['data_folder'] + cfg['train_data'], word2id, relation2id,
    #                       entity2id, cfg['max_query_word'])

    # Test set evaluation
    print("evaluating on test")
    save_model_file = os.path.join(cfg['model_folder'], cfg['save_model_file'])
    print('loading model from ...', save_model_file)
    my_model = get_model(cfg, entity_emb_file, vocab_emb_file, relation_emb_file, test_data.num_kb_relation, len(entity2id), len(word2id))
    my_model.load_state_dict(torch.load(save_model_file))

    test_acc = inference_test(my_model, test_data, entity2id, cfg, log_info=True)

    return test_acc

def dev(cfg, dev_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file):
    print("testing ...")

    # Test set evaluation
    print("evaluating on dev")
    save_model_file = os.path.join(cfg['model_folder'], cfg['save_model_file'])
    print('loading model from ...', save_model_file)

    my_model = get_model(cfg, entity_emb_file, vocab_emb_file, relation_emb_file, dev_data.num_kb_relation, len(entity2id), len(word2id))
    my_model.load_state_dict(torch.load(save_model_file))

    dev_acc = inference_test(my_model, dev_data, entity2id, cfg, log_info=True)

    return dev_acc

def get_model(cfg, pretrained_entity_embedding_file, pretrained_word_embedding_file, pretrained_relation_emb_file, num_kb_relation, num_entities, num_vocab):
    my_model = use_cuda(GRAFTNET(pretrained_entity_embedding_file, pretrained_word_embedding_file, pretrained_relation_emb_file, cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'], cfg['word_dim'],
                 cfg['pagerank_lambda'], cfg['fact_scale'], cfg['lstm_dropout'], cfg['linear_dropout']))

    return my_model

def generate_config_file(path, config_file, train, dev, test, ADD_ENT_MAX, epoch, batch):

    config_dic = {}
    config_dic['name'] = 'graftnet_baseline'
    config_dic['data_folder'] = path + '/data/'
    config_dic['model_folder'] =  path + '/model/'
    config_dic['train_data'] =  train
    config_dic['dev_data'] =  dev
    config_dic['test_data'] =  test

    config_dic['word2id'] =  'vocab_v5c_multiseed_twohopM' + ADD_ENT_MAX +'.txt'
    config_dic['relation2id'] =  'relations_v5c_multiseed_twohopM' + ADD_ENT_MAX + '.txt'
    config_dic['entity2id'] =  'entities_v5c_multiseed_twohopM'  + ADD_ENT_MAX+ '.txt'

    config_dic['word_emb_file'] =  'vocab_v5c_multiseed_twohopM'  + ADD_ENT_MAX + '_emb_wiki2vec100d.npy'
    config_dic['entity_emb_file'] = "entities_v5c_multiseed_twohopM" + ADD_ENT_MAX + '_emb_wiki2vec100d.npy'
    config_dic['relation_emb_file'] =  'relations_v5c_multiseed_twohopM'  + ADD_ENT_MAX + '_emb_wiki2vec100d.npy'

    config_dic['to_save_model'] =  True
    config_dic['save_model_file'] =  'best_model_multiseed_twohop_100dM500_wiki2vec_v5c'
    config_dic['pred_file'] =  'best_model_multiseed_twohop_100dM500_wiki2vec_v5c'
    config_dic['load_model_file'] =  'best_model_multiseed_twohop_100dM500_wiki2vec_v5c'


    # graph options
    config_dic['fact_dropout'] =  0.1

    config_dic['num_layer'] =  3
    config_dic['max_query_word'] =  40

    config_dic['entity_dim'] =  100
    config_dic['word_dim'] =  100

    config_dic['pagerank_lambda'] =  0.8
    config_dic['corner_lambda'] =  1
    config_dic['fact_scale'] =  3

    # optimization
    config_dic['num_epoch'] =  epoch
    config_dic['batch_size'] =  batch
    config_dic['gradient_clip'] =  1
    #config_dic['gradient_clip'] = 0.5
    config_dic['learning_rate'] =  0.001
    config_dic['lstm_dropout'] =  0.3
    config_dic['linear_dropout'] = 0.2
    #config_dic['linear_dropout'] = 0.1
    config_dic['is_debug'] =  False

    with open(config_file, 'w') as file:
        yaml.dump(config_dic, file, default_flow_style=False)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epoch', type=int, default=100)
    argparser.add_argument('-b', '--batch', type=int, default=25)
    args = argparser.parse_args()
    epoch = int(args.epoch)
    batch = int(args.batch)
    path = '/GW/qa2/work/exact/GRAFTNET'

    test_subg = 'test_v5c_multiseed_twohop_subgraph.json'
    dev_subg = 'dev_v5c_multiseed_twohop_subgraph.json'
    train_subg = 'train_v5c_multiseed_twohop_subgraph.json'
    config_file = path + '/config/' + 'configAnswer_v5c_multiseed_twohop' + '_' + str(epoch) + '_' + str(batch) + '.yml'
    ADD_ENT_MAX = '500'
    generate_config_file(path, config_file, train_subg, dev_subg, test_subg, ADD_ENT_MAX, epoch, batch)
    CFG = get_config(config_file)

    valid_acc = train(CFG)

    result_path = '/GW/qa2/work/exact/GRAFTNET/result/'
    re_fp = result_path + config_file.split('/')[len(config_file.split('/')) - 1].replace('.yml', 'test.txt')
    # result on dev set
    fp = open(re_fp, 'w', encoding='utf-8')
    #dev(CFG)
    pred_kb_file = CFG['model_folder'] + CFG['pred_file']
    eps_kb = 0.2
    compare_pr(pred_kb_file, eps_kb, fp)
    fp.close()
    # result on test set

    # re_fp = result_path + config_file.split('/')[len(config_file.split('/')) - 1].replace('.yml', 'test.txt')
    # fp = open(re_fp, 'w', encoding='utf-8')
    # test_acc = test(CFG)
    # pred_kb_file = CFG['model_folder'] + CFG['pred_file']
    # compare_pr(pred_kb_file, eps_kb, fp)
    #
    # fp.close()



