import torch


def collate_fn(batch: list[dict]) -> dict:
    """
    Collates a batch of samples from a dataset.

    This function is designed to handle batches where each sample is a dictionary.
    It recursively collates values associated with
    the same keys across all samples in the batch.
    For tensor values, it concatenates them along the first dimension.
    For dictionary values, it applies the same collation logic recursively.
    For other types of values, it simply aggregates them into a list.

    @param batch: A list of samples, where each sample is a dictionary.
    @return: A dictionary with the same keys as the samples, where each value is either
             a concatenated tensor, a recursively collated dictionary,
             or a list of values.
    """
    if isinstance(batch, list) and isinstance(batch[0], dict):
        batched_dict = {}
        for key in batch[0].keys():
            list_of_values = [item[key] for item in batch]
            batched_dict[key] = collate_fn(list_of_values)
        return batched_dict
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        return torch.cat(batch, dim=0)
    elif isinstance(batch, list):
        return batch
    else:
        return batch


def move_to_device(
    batch: torch.Tensor | dict, device: torch.device
) -> torch.Tensor | dict:
    """
    Moves a batch of data to a specified device (e.g., CPU or GPU).

    Args:
        batch (torch.Tensor or dict): The batch of data to move. Can be a single tensor
        or a dictionary of tensors.
        device (torch.device): The target device to which the batch should be moved.

    Returns:
        torch.Tensor or dict: The batch of data moved to the specified device.
                             If the input is a dictionary, the returned value will be
                             a dictionary with the same keys and values moved to the
                             specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)


def convert_dict_to_tensor(data_dict: dict) -> torch.Tensor:
    """
    Recursively converts lists within a dictionary to PyTorch tensors with dtype=torch.int64.

    @param data_dict: A dictionary potentially containing nested dictionaries and lists.
    @return: A new dictionary with the same structure as `data_dict`, but with lists converted to PyTorch tensors.
    """
    if not isinstance(data_dict, dict):
        if isinstance(data_dict, str):
            return data_dict
        return torch.tensor(data_dict, dtype=torch.int64)
    else:
        new_dict = {}
        for key in data_dict:
            new_dict[key] = convert_dict_to_tensor(data_dict[key])
    return new_dict
