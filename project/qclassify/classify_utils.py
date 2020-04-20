import os
import logging
import math

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
from sklearn.metrics import matthews_corrcoef, f1_score

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class DataProcessor(object):
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.label_list = self.get_labels()

    def get_labels(self):
        return ["0", "1"]

    def get_train_examples(self, data_dir, train_file):
        examples = []
        file_dir = os.path.join(data_dir, train_file)
        with open(file_dir, "r", encoding=self.encoding) as f:
            for i, line in enumerate(f):
                items = [item.strip() for item in line.split("\t") if item != ""]
                if len(items) != 2:
                    continue
                example = InputExample(guid=i, text_a=items[0], text_b=None, label=items[1])
                examples.append(example)
            print(len(examples))
        return examples

    def get_test_examples(self, data_dir, test_file):
        examples = []
        file_dir = os.path.join(data_dir, test_file)
        with open(file_dir, "r", encoding=self.encoding) as f:
            for i, line in enumerate(f):
                items = [item.strip() for item in line.split("\t") if item != ""]
                label = "0"
                if len(items) == 2:
                    if items[-1] not in self.label_list:
                        continue
                    else:
                        label = items[-1]
                example = InputExample(guid=i, text_a=items[0], text_b=None, label=label)
                examples.append(example)
        return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length,
        label_list,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    logging_step = int(len(examples) / 10) if int(len(examples) / 10) else 1
    for (ex_index, example) in enumerate(examples):
        if ex_index % logging_step == 0:
            logger.info("Have processed {} examples to features, {}".format(ex_index, float(ex_index / len(examples))))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        label = label_map[example.label] if example.label else None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    # if not scores:
    #     return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_prob(preds):
    probs = []
    for pred in preds:
        probs.append(_compute_softmax(pred))
    return probs


def write_func(args, output_file_dir, examples, preds, probs):
    fout = open(output_file_dir, "w", encoding="utf-8")
    if args.unlabel:
        fout.write("\t".join(["query", "pred_label", "pos_score"]) + "\n")
        for i, pred in enumerate(preds):
            fout.write("\t".join([examples[i].text_a, str(pred), str(probs[i][1])])+"\n")
    else:
        fout.write("\t".join(["query", "true_label", "pred_label", "pos_score"]) + "\n")
        for i, pred in enumerate(preds):
            fout.write("\t".join([examples[i].text_a, examples[i].label, str(pred), str(probs[i][1])])+"\n")
    fout.close()
