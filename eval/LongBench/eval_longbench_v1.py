import os
import json
import argparse
import numpy as np

from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

# Mapping from dataset names to corresponding metric functions
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    """
    Parse command-line arguments for scorer.

    Args:
        args (list[str], optional): List of args to parse. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments with attributes 'path' and 'pred'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help='Root directory containing prediction files')
    parser.add_argument('--pred', type=str, default="pred",
                        help='Output prediction identifier')
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    """
    Compute length-based grouped scores for evaluation datasets with length categories.

    Args:
        dataset (str): Dataset name.
        predictions (list[str]): Model outputs.
        answers (list[list[str]]): Ground truth answers per instance.
        lengths (list[int]): Input lengths for each instance.
        all_classes (list): List of classes for classification or retrieval metrics.

    Returns:
        dict: Mean scores for each length bucket: '0-4k', '4-8k', '8k+'.
    """
    # Initialize buckets for different length ranges
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for pred, truths, length in zip(predictions, answers, lengths):
        # Truncate predictions on newline for certain datasets
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]
        # Compute best score against all ground truths
        score = max(
            dataset2metric[dataset](pred, gt, all_classes=all_classes)
            for gt in truths
        )
        # Append to correct bucket
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    # Convert to percentages and round
    return {k: round(100 * np.mean(v), 2) for k, v in scores.items()}

def scorer(dataset, predictions, answers, all_classes):
    """
    Compute overall average score for a dataset.

    Args:
        dataset (str): Dataset name.
        predictions (list[str]): Model outputs.
        answers (list[list[str]]): Ground truth answers per instance.
        all_classes (list): List of classes for classification or retrieval metrics.

    Returns:
        float: Percent score averaged over all instances.
    """
    total = 0.0
    for pred, truths in zip(predictions, answers):
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]
        total += max(
            dataset2metric[dataset](pred, gt, all_classes=all_classes)
            for gt in truths
        )
    if not predictions:
        return 0.0
    return round(100 * total / len(predictions), 4)

def process_file(file_path, scores):
    """
    Load a JSON or JSONL result file, compute metrics, and save to a .txt file.

    Args:
        file_path (str): Path to the result file.
        scores (dict): Dictionary to record dataset scores.
    """
    # Only support .json and .jsonl formats
    if not (file_path.endswith('.json') or file_path.endswith('.jsonl')):
        print("Only .json and .jsonl files are supported.")
        return

    predictions, answers, lengths = [], [], []
    # Use the filename (without extension) as default dataset key
    dataset = os.path.basename(file_path).split('.')[0]

    # Read in data entries
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                entry = json.loads(line)
                predictions.append(entry['pred'])
                answers.append(entry['answers'])
                all_classes = entry.get('all_classes', [])
                if 'length' in entry:
                    lengths.append(entry['length'])
        else:
            data_list = json.load(f)
            for entry in data_list:
                predictions.append(entry['pred'])
                answers.append(entry['answers'])
                all_classes = entry.get('all_classes', [])
                if 'length' in entry:
                    lengths.append(entry['length'])

    # Detect extended dataset suffix '_e'
    extended = False
    if dataset.endswith('_e'):
        extended = True
        dataset = dataset[:-2]

    # Resolve dataset name if not directly in mapping
    if dataset not in dataset2metric:
        for part in file_path.split(os.sep):
            if part in dataset2metric:
                dataset = part
                break

    # Compute the appropriate score
    if extended:
        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        dataset += '_e'
    else:
        score = scorer(dataset, predictions, answers, all_classes)

    # Prepare output text path (same base name, .txt extension)
    base = os.path.splitext(file_path)[0]
    out_txt = f"{base}.txt"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    # Write score info to text file
    with open(out_txt, 'w') as f:
        f.write(str({'dataset': dataset, 'score': score}))
    print(f"Data saved to {out_txt}")
    scores[dataset] = score

def process_directory(path, scores, model_name, args):
    """
    Recursively traverse a directory, processing all supported result files.

    Args:
        path (str): Root directory to search.
        scores (dict): Dictionary to accumulate scores.
        model_name (str): Name of the model (unused here).
        args (argparse.Namespace): Parsed CLI arguments.
    """
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            process_directory(full, scores, model_name, args)
        elif os.path.isfile(full):
            try:
                process_file(full, scores)
            except Exception as e:
                print(f"Error processing {full}: {e}")

if __name__ == '__main__':
    args = parse_args()
    scores = {}
    root_path = args.path
    model_name = os.path.basename(root_path)

    # Process all prediction files under the root path
    process_directory(root_path, scores, model_name, args)

    # Save aggregated scores to JSON
    output_path = f"output/{args.pred}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
