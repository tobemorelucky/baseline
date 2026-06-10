# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec


def _load_item_embedding_table(path, item_num, add_extra_zero=True):
    """Safely load item embedding table, handling whether padding row 0 is already present.

    Args:
        path: pkl file path
        item_num: number of actual items (excluding padding)
        add_extra_zero: if True, append an extra zero row at the end (for mask token / extra item)

    Returns:
        float32 numpy array of shape (item_num+1, dim) or (item_num+2, dim) if add_extra_zero
    """
    emb = pickle.load(open(path, "rb"))
    emb = np.array(emb, dtype=np.float32)

    if emb.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding array, got shape {emb.shape}. path={path}"
        )

    print(f"[_load_item_embedding_table] LLM emb path: {path}")
    print(f"  original shape: {emb.shape}")
    print(f"  item_num: {item_num}")

    if emb.shape[0] == item_num + 1:
        print(f"  -> shape[0] == item_num+1, already has padding row 0, keeping as-is")
    elif emb.shape[0] == item_num:
        print(f"  -> shape[0] == item_num, prepending padding row 0")
        emb = np.insert(emb, 0, values=np.zeros((1, emb.shape[1]), dtype=np.float32), axis=0)
    else:
        raise ValueError(
            f"Unexpected embedding shape: {emb.shape}. "
            f"Expected ({item_num}, D) or ({item_num + 1}, D). "
            f"path={path}"
        )

    if add_extra_zero:
        emb = np.concatenate([emb, np.zeros((1, emb.shape[1]), dtype=np.float32)], axis=0)

    print(f"  final shape: {emb.shape}")
    return emb


class SASRecPLUS(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = _load_item_embedding_table(args.llm_emb_path, item_num, add_extra_zero=True)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.filter_init_modules = ["item_emb"]
        self._init_weights()


    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats



class Bert4RecPLUS(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = _load_item_embedding_table(args.llm_emb_path, item_num, add_extra_zero=True)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))
        # self.pad_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        item_seq_emb[log_seqs==self.mask_token] = self.mask_embedding
        # item_seq_emb[log_seqs==0] = self.pad_embedding

        return item_seq_emb
    

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    


class GRU4RecPLUS(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = _load_item_embedding_table(args.llm_emb_path, item_num, add_extra_zero=True)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.filter_init_modules = ["item_emb"]
        self._init_weights()


    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        return item_seq_emb


    def log2feats(self, log_seqs):
        '''Get the representation of given sequence'''
        seqs = self.item_emb(log_seqs)
        seqs = self.adapter(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats




