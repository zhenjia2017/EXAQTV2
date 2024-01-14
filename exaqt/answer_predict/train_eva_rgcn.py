import torch
from tqdm import tqdm
import json
import os
from exaqt.answer_predict.model import Exact
from exaqt.library.utils import get_logger, get_result_logger
from exaqt.answer_predict.data_loader import DataLoader
from exaqt.answer_predict.util import use_cuda, load_dict, cal_accuracy, load_map

class GCN:
    def __init__(self, config):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.name = self.config["name"]
        self.nerd = self.config["nerd"]
        self.input_dir = self.config["path_to_intermediate_results"]
        self.benchmark = self.config["benchmark"]
        self.nerd = self.config["nerd"]
        self.data_folder = os.path.join(self.input_dir, self.benchmark, self.nerd, 'answer_predict')
        self.result_path = os.path.join(self.data_folder, 'result')
        self.model_path = os.path.join(self.data_folder, 'model')
        self.entity2id = load_dict(os.path.join(self.data_folder, 'entities.txt'))
        self.word2id = load_dict(os.path.join(self.data_folder, 'words.txt'))
        self.relation2id = load_dict(os.path.join(self.data_folder, 'relations.txt'))
        self.trelation2id = load_dict(os.path.join(self.data_folder, 'trelations.txt'))
        self.date2id = load_dict(os.path.join(self.data_folder, 'dates.txt'))
        self.tempf2id = load_map(os.path.join(self.data_folder, 'tempfacts2id.pkl'))
        self.category2id = load_dict(os.path.join(self.data_folder, 'categories.txt'))
        self.signal2id = load_dict(os.path.join(self.data_folder, 'signals.txt'))

        self.len_entity2id, self.len_word2id, self.len_category2id, self.len_signal2id, self.len_date2id, self.len_tempf2id = len(self.entity2id), len(self.word2id), len(self.category2id), len(self.signal2id), len(self.date2id), len(
            self.tempf2id)

    def train(self):
        print("training ...")
        # prepare data
        train_data = DataLoader(os.path.join(self.data_folder, f"train_subgraph.json"), self.word2id, self.relation2id, self.trelation2id, self.entity2id, self.date2id, self.tempf2id, self.category2id, self.signal2id, self.config['max_query_word'], self.config['max_temp_fact'])
        valid_data = DataLoader(os.path.join(self.data_folder, f"dev_subgraph.json"), self.word2id, self.relation2id, self.trelation2id, self.entity2id, self.date2id, self.tempf2id, self.category2id, self.signal2id, self.config['max_query_word'], self.config['max_temp_fact'])

        # create model & set parameters
        my_model = self.get_model(train_data.num_kb_relation)
        trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.config['learning_rate'])

        best_dev_acc = 0.0

        for epoch in range(self.config['num_epoch']):
            try:
                print('epoch', epoch)
                train_data.reset_batches(is_sequential = self.config['is_debug'])
                # Train
                my_model.train()
                train_loss, train_acc, train_max_acc = [], [], []
                for iteration in tqdm(range(train_data.num_data // self.config['batch_size'])):
                    batch = train_data.get_batch(iteration, self.config['batch_size'], self.config['fact_dropout'])
                    loss, pred, pred_dist = my_model(batch)
                    #break
                    pred = pred.data.cpu().numpy()
                    acc, max_acc = cal_accuracy(pred, batch[-1])
                    train_loss.append(loss.data)
                    train_acc.append(acc)
                    train_max_acc.append(max_acc)
                    # back propogate
                    my_model.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), self.config['gradient_clip'])
                    optimizer.step()
                print('avg_training_loss', sum(train_loss) / len(train_loss))
                print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
                print('avg_training_acc', sum(train_acc) / len(train_acc))
                print("validating ...")
                eval_acc = self.inference_best_acc(my_model, valid_data)
                if eval_acc > best_dev_acc and self.config['to_save_model']:
                    print("saving model to", os.path.join(self.model_path, self.config['save_model_file']))
                    torch.save(my_model.state_dict(), os.path.join(self.model_path, self.config['save_model_file']))
                    best_dev_acc = eval_acc

            except KeyboardInterrupt:
                break

        return

    def output_pred_dist(self, pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
        for i, p_dist in enumerate(pred_dist):
            data_id = start_id + i
            l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
            output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
            answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
            f_pred.write(json.dumps({'dist': output_dist, 'id': data_loader.data[data_id]['id'], 'answers':answers, 'seeds': data_loader.data[data_id]['seed_entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

    def inference_best_acc(self, my_model, valid_data, log_info=False):
        # Evaluation
        my_model.eval()
        eval_loss, eval_acc, eval_max_acc = [], [], []
        id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        valid_data.reset_batches(is_sequential = True)
        test_batch_size = 32
        if log_info:
            f_pred = open(os.path.join(self.model_path, self.config['pred_file']), 'w')
        for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
            loss, pred, pred_dist = my_model(batch)
            pred = pred.data.cpu().numpy()
            acc, max_acc = cal_accuracy(pred, batch[-1])
            if log_info:
                self.output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
            eval_loss.append(loss.data)
            eval_acc.append(acc)
            eval_max_acc.append(max_acc)

        print('avg_loss', sum(eval_loss) / len(eval_loss))
        print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
        print('avg_acc', sum(eval_acc) / len(eval_acc))

        return sum(eval_acc) / len(eval_acc)

    def inference(self, my_model, valid_data, log_info=False):
        # Evaluation
        my_model.eval()
        eval_loss, eval_acc, eval_max_acc = [], [], []
        id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        valid_data.reset_batches(is_sequential = True)
        test_batch_size = 1
        if log_info:
            f_pred = open(os.path.join(self.model_path, self.config['pred_file']), 'w')
        for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
            loss, pred, pred_dist = my_model(batch)
            pred = pred.data.cpu().numpy()
            acc, max_acc = cal_accuracy(pred, batch[-1])
            if log_info:
                self.output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
            eval_loss.append(loss.data)
            eval_acc.append(acc)
            eval_max_acc.append(max_acc)

        print('avg_loss', sum(eval_loss) / len(eval_loss))
        print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
        print('avg_acc', sum(eval_acc) / len(eval_acc))

        return sum(eval_acc) / len(eval_acc)

    def test(self):
        print("testing ...")

        test_data = DataLoader(os.path.join(self.data_folder, f"test_subgraph.json"), self.word2id, self.relation2id, self.trelation2id, self.entity2id,
                           self.date2id, self.tempf2id, self.category2id, self.signal2id, self.config['max_query_word'], self.config['max_temp_fact'])

        # Test set evaluation
        print("evaluating on test")
        print('loading model from ...', os.path.join(self.model_path, self.config['save_model_file']))

        my_model = self.get_model(test_data.num_kb_relation)
        my_model.load_state_dict(torch.load(os.path.join(self.model_path, self.config['save_model_file'])))

        test_acc = self.inference(my_model, test_data, log_info=True)

        return test_acc

    def dev(self):
        print("dev testing ...")
        dev_data = DataLoader(os.path.join(self.data_folder, f"dev_subgraph.json"), self.word2id, self.relation2id, self.trelation2id, self.entity2id,
                           self.date2id, self.tempf2id, self.category2id, self.signal2id, self.config['max_query_word'], self.config['max_temp_fact'])

        # Test set evaluation
        print("evaluating on test")
        print('loading model from ...', os.path.join(self.model_path, self.config['save_model_file']))

        my_model = self.get_model(dev_data.num_kb_relation)
        my_model.load_state_dict(torch.load(os.path.join(self.model_path, self.config['save_model_file'])))

        dev_acc = self.inference(my_model, dev_data, log_info=True)

        return dev_acc

    def get_model(self, num_kb_relation):
        pretrained_word_emb_file = os.path.join(self.data_folder, self.config['word_emb'])
        pretrained_entity_emb_file = os.path.join(self.data_folder, self.config['entity_emb'])
        pretrained_relation_emb_file = os.path.join(self.data_folder, self.config['relation_emb'])
        pretrained_date_tem_file = os.path.join(self.data_folder, self.config['date_emb'])
        pretrained_tempfact_te_emb_file = os.path.join(self.data_folder, self.config['tempfact_te_emb'])
        pretrained_tempfact_emb_file = os.path.join(self.data_folder, self.config['tempfact_emb'])


        num_entities, num_vocab, num_categories, num_signals, num_dates, num_tempf = self.len_entity2id, self.len_word2id, self.len_category2id, self.len_signal2id, self.len_date2id, self.len_tempf2id
        type_dim = num_categories #multi-hot encoding dimension
        sig_dim = num_signals #multi-hot encoding dimension

        my_model = use_cuda(Exact(pretrained_word_emb_file, pretrained_relation_emb_file, pretrained_entity_emb_file, pretrained_date_tem_file, pretrained_tempfact_te_emb_file, pretrained_tempfact_emb_file, self.config['num_layer'], num_kb_relation, num_entities, num_vocab, num_tempf,  self.config['entity_dim'], self.config['word_dim'], self.config['tem_dim'], self.config['fact_dim'],
                 type_dim, sig_dim, self.config['pagerank_lambda'], self.config['fact_scale'], self.config['lstm_dropout'], self.config['linear_dropout'], self.config['TCE'], self.config['TSE'], self.config['TEE'], self.config['TE'], self.config['ATR']))

        return my_model




