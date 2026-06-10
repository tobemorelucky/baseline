#!/usr/bin/env python3
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def summarize_obj(name: str, obj: Any):
    print(f"[REF] {name}: type={type(obj)}, len={len(obj) if hasattr(obj, '__len__') else 'NA'}")
    if isinstance(obj, dict):
        keys = list(obj.keys())[:3]
        for k in keys:
            print(f"      sample {repr(k)} -> {repr(obj[k])}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            print(f"      sample[{i}] = {repr(v)}")
    else:
        print(f"      repr={repr(obj)[:300]}")


def parse_inter(inter_path: Path) -> Dict[int, List[int]]:
    """
    Robust parser:
    - accepts whitespace/tab/comma separated lines
    - ignores header or invalid lines
    - uses first two integer-like columns as user_id, item_id
    - preserves line order as sequence order
    """
    user_seq: Dict[int, List[int]] = {}

    with open(inter_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # support comma/tab/space
            if "," in line and "\t" not in line:
                parts = line.split(",")
            else:
                parts = line.replace("\t", " ").split()

            ints = []
            for p in parts:
                try:
                    # support "12", "12.0"
                    ints.append(int(float(p)))
                except Exception:
                    pass

            if len(ints) < 2:
                # probably header
                continue

            u, i = ints[0], ints[1]
            if u <= 0 or i <= 0:
                # LLMEmb/PoMRec are usually 1-based ids; skip suspicious 0 id lines
                # If your data really uses 0-based ids, remove this check.
                continue

            user_seq.setdefault(u, []).append(i)

    return user_seq


def build_eval_dicts(user_seq: Dict[int, List[int]], num_neg: int, seed: int):
    """
    Leave-one-out:
    - valid positive = second last item
    - test positive = last item
    - negatives sampled from all item ids excluding all user positives
    """
    all_items = set()
    for seq in user_seq.values():
        all_items.update(seq)

    max_item = max(all_items)
    item_pool = list(range(1, max_item + 1))

    valid_pos = {}
    test_pos = {}
    valid_neg = {}
    test_neg = {}

    rng = random.Random(seed)

    skipped = 0
    for u in sorted(user_seq.keys()):
        seq = user_seq[u]
        if len(seq) < 3:
            skipped += 1
            continue

        user_items = set(seq)
        valid_item = seq[-2]
        test_item = seq[-1]

        candidates = [x for x in item_pool if x not in user_items]
        if len(candidates) < num_neg:
            raise RuntimeError(
                f"user={u} has only {len(candidates)} negative candidates, "
                f"less than num_neg={num_neg}. max_item={max_item}"
            )

        valid_pos[u] = valid_item
        test_pos[u] = test_item
        valid_neg[u] = rng.sample(candidates, num_neg)
        test_neg[u] = rng.sample(candidates, num_neg)

    stats = {
        "num_users_total": len(user_seq),
        "num_users_eval": len(valid_pos),
        "num_users_skipped_len_lt_3": skipped,
        "max_item": max_item,
        "num_items_seen": len(all_items),
        "num_neg": num_neg,
        "seed": seed,
    }

    return valid_pos, test_pos, valid_neg, test_neg, stats


def convert_like_reference(ref: Any, obj_dict: Dict[int, Any]):
    """
    Convert generated dict into the same broad structure as Beauty reference.
    Usually Beauty is dict, then this returns dict directly.

    If Beauty reference is a list:
    - pair-list format: [(u, value), ...]
    - indexed-list format: [None/0, value_for_user1, ...]
    """
    if isinstance(ref, dict):
        # match key type roughly
        if len(ref) == 0:
            return obj_dict
        ref_key = next(iter(ref.keys()))
        if isinstance(ref_key, str):
            return {str(k): v for k, v in obj_dict.items()}
        return obj_dict

    if isinstance(ref, list):
        if len(ref) > 0 and isinstance(ref[0], (tuple, list)) and len(ref[0]) == 2:
            return [(u, obj_dict[u]) for u in sorted(obj_dict.keys())]

        # indexed by user id, with padding at index 0
        max_u = max(obj_dict.keys()) if obj_dict else 0
        out = [None] * (max_u + 1)
        for u, v in obj_dict.items():
            out[u] = v
        return out

    # fallback: keep dict
    return obj_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="ml-1m or toys")
    parser.add_argument("--inter_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--beauty_dir", default="data/beauty/handled")
    parser.add_argument("--num_neg", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--check_only", type=int, default=0)
    args = parser.parse_args()

    inter_path = Path(args.inter_path)
    out_dir = Path(args.out_dir)
    beauty_dir = Path(args.beauty_dir)

    ref_valid_pos = load_pkl(beauty_dir / "valid_pos.pkl")
    ref_test_pos = load_pkl(beauty_dir / "test_pos.pkl")
    ref_valid_neg = load_pkl(beauty_dir / "valid_neg.pkl")
    ref_test_neg = load_pkl(beauty_dir / "test_neg.pkl")

    print("=" * 80)
    print("[INFO] Beauty reference format:")
    summarize_obj("valid_pos.pkl", ref_valid_pos)
    summarize_obj("test_pos.pkl", ref_test_pos)
    summarize_obj("valid_neg.pkl", ref_valid_neg)
    summarize_obj("test_neg.pkl", ref_test_neg)

    print("=" * 80)
    print(f"[INFO] Parsing inter: {inter_path}")
    user_seq = parse_inter(inter_path)
    print(f"[INFO] users={len(user_seq)}")
    lens = [len(v) for v in user_seq.values()]
    print(f"[INFO] min_len={min(lens)}, max_len={max(lens)}, avg_len={sum(lens)/len(lens):.4f}")

    valid_pos, test_pos, valid_neg, test_neg, stats = build_eval_dicts(
        user_seq=user_seq,
        num_neg=args.num_neg,
        seed=args.seed,
    )

    print("=" * 80)
    print("[INFO] Generated stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    out_valid_pos = convert_like_reference(ref_valid_pos, valid_pos)
    out_test_pos = convert_like_reference(ref_test_pos, test_pos)
    out_valid_neg = convert_like_reference(ref_valid_neg, valid_neg)
    out_test_neg = convert_like_reference(ref_test_neg, test_neg)

    print("=" * 80)
    print("[INFO] Output preview:")
    summarize_obj("valid_pos.pkl", out_valid_pos)
    summarize_obj("test_pos.pkl", out_test_pos)
    summarize_obj("valid_neg.pkl", out_valid_neg)
    summarize_obj("test_neg.pkl", out_test_neg)

    if args.check_only:
        print("[CHECK_ONLY] not writing files")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pkl(out_valid_pos, out_dir / "valid_pos.pkl")
    save_pkl(out_test_pos, out_dir / "test_pos.pkl")
    save_pkl(out_valid_neg, out_dir / "valid_neg.pkl")
    save_pkl(out_test_neg, out_dir / "test_neg.pkl")

    with open(out_dir / "fixed_neg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"[DONE] saved fixed eval files to: {out_dir}")


if __name__ == "__main__":
    main()
