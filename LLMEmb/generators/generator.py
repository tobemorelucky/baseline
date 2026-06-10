# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset, Seq2SeqDataset, SeqEvalDatasetFixedNeg
from utils.utils import unzip_data, concat_data, concat_data_with_user


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open('./data/%s/handled/%s.txt' % (self.dataset, self.inter_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        
        self.user_num = usernum
        self.item_num = itemnum

        for user in tqdm(User):
            nfeedback = len(User[user]) - self.args.aug_seq_len
            #nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test


    
    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader


    def make_evalloader(self, test=False):

        if self.args.fixed_eval_neg:

            # ---- fixed negative evaluation ----
            split_name = "test" if test else "valid"
            print("[FIXED_EVAL_NEG] enabled=1 split={}".format(split_name), flush=True)

            handled_dir = "./data/{}/handled".format(self.dataset)

            if test:
                neg_path = os.path.join(handled_dir, self.args.test_neg_file)
                pos_path = os.path.join(handled_dir, self.args.test_pos_file)
            else:
                neg_path = os.path.join(handled_dir, self.args.valid_neg_file)
                pos_path = os.path.join(handled_dir, self.args.valid_pos_file)

            self.logger.info(
                "[fixed_eval_neg] Loading fixed negatives for {}: neg={}, pos={}".format(
                    split_name, neg_path, pos_path
                )
            )

            fixed_neg = pickle.load(open(neg_path, "rb"))
            fixed_pos = pickle.load(open(pos_path, "rb"))

            neg_len = len(list(fixed_neg.values())[0]) if fixed_neg else 0

            self.logger.info(
                "[fixed_eval_neg] split={}, users={}, neg_len={}".format(
                    split_name, len(fixed_neg), neg_len
                )
            )

            # Build eval data: history only, NOT including the target item
            # - valid: history = train (not train+valid, because valid item is the target)
            # - test:  history = train + valid
            eval_data = []
            eval_users = []

            if test:
                for user in self.train:
                    history = self.train[user] + self.valid[user]
                    eval_data.append(history)
                    eval_users.append(user)
            else:
                for user in self.train:
                    history = self.train[user]
                    eval_data.append(history)
                    eval_users.append(user)

            # MANDATORY stdout: print first-user diagnostics
            if eval_users:
                first_uid = eval_users[0]
                first_pos = fixed_pos.get(first_uid)
                first_neg = fixed_neg.get(first_uid, [])
                print(
                    "[FIXED_EVAL_NEG] user_count={} neg_len={} first_user={} pos={} neg_head={}".format(
                        len(eval_users), neg_len, first_uid, first_pos, first_neg[:5]
                    ),
                    flush=True,
                )
                self.logger.info(
                    "[fixed_eval_neg] first user id={}, history_len={}, pos={}, first5_neg={}".format(
                        first_uid, len(eval_data[0]), first_pos, first_neg[:5]
                    )
                )

            self.eval_dataset = SeqEvalDatasetFixedNeg(
                eval_data, eval_users,
                self.item_num, self.args.max_len,
                fixed_neg, fixed_pos,
            )

        else:
            # ---- original random-negative evaluation ----
            print("[FIXED_EVAL_NEG] enabled=0 split={}".format("test" if test else "valid"), flush=True)
            if test:
                eval_dataset = concat_data([self.train, self.valid, self.test])
            else:
                eval_dataset = concat_data([self.train, self.valid])

            self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)

        eval_dataloader = DataLoader(self.eval_dataset,
                                     sampler=SequentialSampler(self.eval_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)

        return eval_dataloader

    
    def get_user_item_num(self):

        return self.user_num, self.item_num
    

    def get_item_pop(self):
        """get item popularity according to item index. return a np-array"""
        all_data = concat_data([self.train, self.valid, self.test])
        pop = np.zeros(self.item_num+1) # item index starts from 0
        
        for items in all_data:
            pop[items] += 1

        return pop
    

    def get_user_len(self):
        """get sequence length according to user index. return a np-array"""
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)
    
    

class Seq2SeqGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = Seq2SeqDataset(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    

    

    