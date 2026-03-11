import re
from collections import Counter
from typing import Dict, List, Optional

from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_ABBR, MAP_ID_TO_NAME


_MAP_ALIASES: Dict[int, List[str]] = {
    0: ["las vegas", "vegas", "strip", "lav", "拉斯维加斯"],
    1: ["pittsburgh", "hazelwood", "pgh", "匹兹堡"],
    2: ["singapore", "one north", "one-north", "sgp", "新加坡"],
    3: ["boston", "bos", "波士顿"],
}


def _find_mentioned_classes(prompt: str, num_classes: int) -> List[int]:
    prompt_lc = prompt.lower()
    mentioned_classes: List[int] = []
    for class_label in range(num_classes):
        aliases = _MAP_ALIASES.get(class_label, []) + [MAP_ID_TO_NAME[class_label], MAP_ID_TO_ABBR[class_label].lower()]
        for alias in aliases:
            if alias in prompt_lc:
                mentioned_classes.append(class_label)
                break

    if mentioned_classes:
        return mentioned_classes

    # fallback: parse integer class ids from text, e.g. "class 0 and class 2"
    parsed_ids = [int(num_str) for num_str in re.findall(r"\b(\d+)\b", prompt_lc)]
    return [class_id for class_id in parsed_ids if 0 <= class_id < num_classes]


def _extract_class_weights(prompt: str, mentioned_classes: List[int], num_classes: int) -> Dict[int, int]:
    prompt_lc = prompt.lower()
    weights: Counter[int] = Counter()

    for class_label in range(num_classes):
        if class_label not in mentioned_classes:
            continue

        aliases = _MAP_ALIASES.get(class_label, []) + [MAP_ID_TO_NAME[class_label], MAP_ID_TO_ABBR[class_label].lower()]
        for alias in aliases:
            # Supports expressions like "8 boston" or "boston 8"
            before_match = re.search(rf"\b(\d+)\s+{re.escape(alias)}\b", prompt_lc)
            after_match = re.search(rf"\b{re.escape(alias)}\s+(\d+)\b", prompt_lc)
            if before_match:
                weights[class_label] += int(before_match.group(1))
            if after_match:
                weights[class_label] += int(after_match.group(1))

    return dict(weights)


def resolve_class_labels_from_prompt(
    prompt: Optional[str],
    num_classes: int,
    inference_batch_size: int,
    default_class_labels: List[int],
) -> List[int]:
    """Resolve class labels from a natural-language prompt."""

    if not prompt:
        return default_class_labels

    mentioned_classes = _find_mentioned_classes(prompt=prompt, num_classes=num_classes)
    if not mentioned_classes:
        return default_class_labels

    prompt_lc = prompt.lower()
    if len(mentioned_classes) == 1 and any(token in prompt_lc for token in ["only", "all", "just", "仅", "只", "全部"]):
        return [mentioned_classes[0]] * inference_batch_size

    class_weights = _extract_class_weights(prompt=prompt, mentioned_classes=mentioned_classes, num_classes=num_classes)
    if class_weights:
        total_weight = sum(class_weights.values())
        weighted_labels: List[int] = []
        for class_label, weight in class_weights.items():
            count = max(int(round(inference_batch_size * (weight / total_weight))), 1)
            weighted_labels.extend([class_label] * count)
        if len(weighted_labels) < inference_batch_size:
            weighted_labels.extend(mentioned_classes * inference_batch_size)
        return weighted_labels[:inference_batch_size]

    if any(token in prompt_lc for token in ["mostly", "mainly", "主要", "大部分"]) and len(mentioned_classes) >= 1:
        head_count = max(int(inference_batch_size * 0.7), 1)
        tail = mentioned_classes[1:] if len(mentioned_classes) > 1 else mentioned_classes
        tail_labels = (tail * inference_batch_size)[: inference_batch_size - head_count]
        return [mentioned_classes[0]] * head_count + tail_labels

    return (mentioned_classes * inference_batch_size)[:inference_batch_size]
