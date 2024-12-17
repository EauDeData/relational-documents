import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function for processing a batch of samples.

    Args:
        batch (list): A list of dictionaries containing the following keys:
            - 'image': Tensor representing the image.
            - 'wordcrops': List of tensors representing word crops.
            - 'query': Tensor of tokenized query sentence.

    Returns:
        Tuple of:
            - Stacked tensor of images.
            - Stacked tensor of word crops sequences (no padding).
            - Padded tensor of tokenized sentences.
    """
    images = torch.stack([sample['image'] for sample in batch])

    # Since wordcrops are lists of tensors and have the same length across samples,
    # we can stack them directly.
    word_crops = torch.stack([torch.stack(sample['wordcrops']) for sample in batch])

    # Pad the query sentences to the same length and stack them.
    queries = [sample['query'] for sample in batch]
    max_leng_query = max(len(q) for q in queries)
    queries_padded = torch.tensor([q + [0] * (max_leng_query - len(q)) for q in queries], dtype=torch.int64)

    return images, word_crops.unsqueeze(2), queries_padded
