from typing import Dict, Tuple

def create_label_dicts(entity_set: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {'O': 0}
    for idx, label in enumerate(entity_set, start=1):
        label2id[f'B-{label}'] = idx
        label2id[f'I-{label}'] = idx + len(entity_set)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label
