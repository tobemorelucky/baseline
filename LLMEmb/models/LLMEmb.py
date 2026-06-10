# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.Adapter import SASRecPLUS, Bert4RecPLUS, GRU4RecPLUS
from models.utils import Contrastive_Loss2



def _load_srs_embedding_table(path, item_num):
    """Safely load SRS item embedding table, handling whether padding row 0 is already present.

    Args:
        path: pkl file path
        item_num: number of actual items (excluding padding)

    Returns:
        float32 numpy array of shape (item_num+1, dim)
    """
    emb = pickle.load(open(path, "rb"))
    emb = np.array(emb, dtype=np.float32)

    if emb.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding array, got shape {emb.shape}. path={path}"
        )

    print(f"[_load_srs_embedding_table] SRS emb path: {path}")
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

    print(f"  final shape: {emb.shape}")
    return emb


class LLMEmbSASRec(SASRecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        
        srs_item_emb = _load_srs_embedding_table("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), item_num)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()

    
    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss
    


class LLMEmbBert4Rec(Bert4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        srs_item_emb = _load_srs_embedding_table("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), item_num)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()

    
    def forward(self, seq, pos, neg, positions, **kwargs):

        loss =  super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss
    


class LLMEmbGRU4Rec(GRU4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        srs_item_emb = _load_srs_embedding_table("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), item_num)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()


    def forward(self, seq, pos, neg, positions, **kwargs):

        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss




