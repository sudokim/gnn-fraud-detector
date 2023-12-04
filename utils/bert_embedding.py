import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


__all__ = ["get_bert_embedding"]


@torch.no_grad()
def get_bert_embedding(
    user_text: dict[int, tuple[str, list[tuple[str, str]]]], bert_embedding_path: str
) -> torch.Tensor:
    """
    Generate BERT embedding for each user.

    Args:
        user_text (dict[int, tuple[str, list[tuple[str, str]]]]): The dataset in the format of
            {user_id: (reviewerID, [(asin, reviewText)])}
        bert_embedding_path (str): Path to the BERT embedding file. If the file exists, the embedding will be loaded. 
            Otherwise, the embedding will be computed and saved.

    Returns:
        torch.Tensor: BERT embedding of each user.
    """
    bert_embedding_path = Path(bert_embedding_path)

    if bert_embedding_path.exists():
        print("Loading Pre-Computed BERT Embedding...")
        embedding = pickle.load(bert_embedding_path.open("rb"))
        assert len(embedding) == len(user_text)

        # Convert into torch.Tensor
        embedding_tensor = torch.empty((len(embedding), embedding[0].shape[0]))
        for i, emb in embedding.items():
            embedding_tensor[i] = emb

        return embedding_tensor

    print("Loading BERT Model...")
    model = BertModel.from_pretrained("bert-base-cased").to("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    print("Computing BERT Embedding...")
    embedding: dict[int, torch.Tensor] = {}

    for user_id, (_, reviews) in tqdm(user_text.items(), desc="Computing BERT Embedding"):
        review_texts = [review_text for _, review_text in reviews]
        user_embedding = []
        for review_text in review_texts:
            review_text_tokenized = tokenizer(review_text, padding=True, truncation=True, return_tensors="pt")
            output = model.forward(**review_text_tokenized.to("cuda")).last_hidden_state[:, 0, :].mean(dim=0).cpu()

            user_embedding.append(output)

        embedding[user_id] = torch.stack(user_embedding).mean(dim=0)

    pickle.dump(embedding, bert_embedding_path.open("wb"))

    embedding_tensor = torch.empty((len(embedding), embedding[0].shape[0]))
    for i, emb in embedding.items():
        embedding_tensor[i] = emb

    return embedding_tensor
