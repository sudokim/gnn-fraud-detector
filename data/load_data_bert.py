import gzip
import json
import pickle
from pathlib import Path
import os

__all__ = ["load_amazon_data"]

from tqdm import tqdm


def load_meta_data(dir):
    meta_dict = {}
    meta_path = Path(dir) / "dataset" / "meta_Musical_Instruments.json.gz"
    print("Loading meta data...")
    with gzip.open(meta_path, "rb") as f:
        for line in f:
            line = eval(line)
            if "title" not in line.keys():
                line["title"] = ""
            meta_dict[line["asin"]] = line
    return meta_dict


def load_amazon_data(data):
    print("Loading AMAZON data for bert...")
    if Path(os.getcwd() + f"/dataset/{data}/train.json").exists():
        print("Data already exists")
        train_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/train.json"), "r"))
        val_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/val.json"), "r"))
        test_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/test.json"), "r"))
        return train_review, val_review, test_review

    dir = Path(__file__).parent.parent

    user_text = pickle.load(open(Path(dir) / "dataset" / data / "user_text.pkl",
                                 "rb"))  # {node_id: (reviewerID, [(asin, "리뷰텍스트"), ...]), ...}
    user_label = pickle.load(open(Path(dir) / "dataset" / "labels.pkl", "rb"))
    for user_node_id, user_label in zip(user_text.keys(), user_label):
        user_text[user_node_id] = list(user_text[user_node_id])
        user_text[user_node_id].append(int(user_label))
    print(f"Number of users: {len(user_text)}")

    # split train, valid, test and filter out users with no labels
    user_split = pickle.load(open(Path(dir) / "dataset" / "split_masks.pkl",
                                  "rb"))  # {"train": [True, False, True, True, ...], "val": [...], "test": [...]}
    train_split, val_split, test_split = user_split["train"], user_split["val"], user_split["test"]
    train_review, val_review, test_review = [], [], []
    for (user_node_id, user_info), train, val, test in zip(user_text.items(), train_split, val_split, test_split):
        if user_info[-1] != -100:
            # (user_node_id, asin, review, label)
            if train:
                train_review.extend(
                    [(user_node_id, purchase[0], purchase[1], user_info[-1]) for purchase in user_info[1]])
            elif val:
                val_review.extend(
                    [(user_node_id, purchase[0], purchase[1], user_info[-1]) for purchase in user_info[1]])
            elif test:
                test_review.extend(
                    [(user_node_id, purchase[0], purchase[1], user_info[-1]) for purchase in user_info[1]])
            else:
                raise ValueError("Invalid split mask")

    print(f"Number of train users: {len(train_review)}")
    print(f"Number of val users: {len(val_review)}")
    print(f"Number of test users: {len(test_review)}")

    # replace asin with item title
    meta_data = load_meta_data(dir)
    train_review = [(user_node_id, meta_data[asin]["title"], review, label) for user_node_id, asin, review, label in
                    train_review]
    val_review = [(user_node_id, meta_data[asin]["title"], review, label) for user_node_id, asin, review, label in
                  val_review]
    test_review = [(user_node_id, meta_data[asin]["title"], review, label) for user_node_id, asin, review, label in
                   test_review]

    # save
    json.dump(train_review, open(Path(dir) / "dataset" / "train.json", "w"))
    json.dump(val_review, open(Path(dir) / "dataset" / "val.json", "w"))
    json.dump(test_review, open(Path(dir) / "dataset" / "test.json", "w"))

    print("")
    return train_review, val_review, test_review


def load_steam_data(data):
    print("Loading STEAM data for bert...")
    if Path(os.getcwd() + f"/dataset/{data}/train.json").exists():
        print("Data already exists")
        train_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/train.json"), "r", encoding="utf-8"))
        val_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/val.json"), "r", encoding="utf-8"))
        test_review = json.load(open(Path(os.getcwd() + f"/dataset/{data}/test.json"), "r", encoding="utf-8"))
        return train_review, val_review, test_review
    else:
        print("Request data to Junyoung")
        raise NotImplementedError
