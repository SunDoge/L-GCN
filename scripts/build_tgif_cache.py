import argparse
import os
import pickle
from typing import Dict, Optional

import pandas as pd
import torch
from torchtext import vocab
from tqdm import tqdm

from utils import load_csv_from_dataset
from utils.dictionary import Dictionary, CharDictionary
from utils.const import IGNORE_INDEX
from typing import Union

MULTIPLE_CHOICE_TASKS = ['action', 'transition']
ALL_TASKS = ['action', 'transition', 'count', 'frameqa']

# class Args(TypedArgs):
#     all_tasks = ['action', 'transition', 'count', 'frameqa']

#     def __init__(self):
#         parser = argparse.ArgumentParser()

#         self.tasks = parser.add_argument(
#             '-t',
#             '--tasks',
#             nargs=argparse.ONE_OR_MORE,
#             choices=Args.all_tasks,
#             default=Args.all_tasks,
#         )

#         self.output_path: Union[str, argparse.Action] = parser.add_argument(
#             '-o',
#             '--output-path',
#             default='cache'
#         )

#         self.parse_args_from(parser)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--task', nargs=argparse.ONE_OR_MORE,
    choices=ALL_TASKS,
    default=ALL_TASKS,
    help='which subtask to preprocess'
)
parser.add_argument(
    '-o', '--output-path',
    default='cache/tgif'
)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')


def build_answer_dict(args: Args) -> Dict[str, int]:
    answer_dict_path = os.path.join(
        args.output_path, 'frameqa_answer_dict.pkl')

    # If cache exists, use cache
    if os.path.exists(answer_dict_path):
        print(f'{answer_dict_path} exists, load cache')
        with open(answer_dict_path, 'rb') as f:
            return pickle.load(f)

    print(f'{answer_dict_path} not exists, build cache')

    # Must be train split
    df = load_csv_from_dataset('frameqa', 'Train')
    all_answers = df['answer']

    answers = list(set(all_answers))

    answer_dict = {answer: i for i, answer in enumerate(answers)}

    with open(answer_dict_path, 'wb') as f:
        pickle.dump(answer_dict, f)

    return answer_dict


def build_pretrained_embedding(args: Args, task: str, dictionary: Dictionary, vector: vocab.Vocab):
    embedding_path = os.path.join(args.output_path, f'{task}_embedding.pt')

    if os.path.exists(embedding_path):
        print(f'{embedding_path} exists, return')
        return

    print(f'{embedding_path} not exists, build')
    word2idx = dictionary.word2idx

    weights = torch.zeros(len(word2idx), 300)

    for word, index in word2idx.items():
        weight = vector[word]
        weights[index] = weight

    torch.save(weights, embedding_path)


def process_open_ended(args: Args, task: str, dictionary: Dictionary, char_dictionary: CharDictionary,
                       answer_dict: Dict[str, int] = None):
    def process(split: str):
        print(f'processing {task} {split}')
        data = []
        df = load_csv_from_dataset(task, split)

        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            answer = row['answer']
            gif_name = row['gif_name']

            question_ids = dictionary.tokenize(question)
            question_chars = char_dictionary.tokenize(question)

            if task == 'frameqa':
                answer_id = IGNORE_INDEX
                if answer in answer_dict:
                    answer_id = answer_dict[answer]
            else:
                # https://github.com/YunseokJANG/tgif-qa/blob/master/code/gifqa/data_util/tgif.py#L561
                answer_id = max(float(answer), 1.0)
                # answer_id = float(answer)

            data.append({
                'question': question,
                'answer': answer,
                'question_ids': question_ids,
                'question_chars': question_chars,
                'answer_id': answer_id,
                'gif_name': gif_name
            })

        filename = os.path.join(
            args.output_path, f'{split}_{task}_question.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    process('Train')
    process('Test')


def build_dictionary(args: Args, task: str) -> Dictionary:
    dictionary_path = os.path.join(args.output_path, f'{task}_dictionary.pkl')

    if os.path.exists(dictionary_path):
        print(f'{dictionary_path} exists, load cache')
        return Dictionary.load_from_file(dictionary_path)

    dictionary = Dictionary()

    def build(split: str):
        df = load_csv_from_dataset(task, split)

        for question in df['question']:
            dictionary.tokenize(
                question, add_word=True, extra_dict=glove.stoi if split == 'Test' else None)

        if task in MULTIPLE_CHOICE_TASKS:
            for answer_key in ['a1', 'a2', 'a3', 'a4', 'a5']:
                for answer in df[answer_key]:
                    dictionary.tokenize(
                        answer, add_word=True, extra_dict=glove.stoi if split == 'Test' else None)

    build('Train')
    build('Test')

    dictionary.dump_to_file(dictionary_path)
    return dictionary


def build_char_dictionary(args: Args, task: str) -> Dictionary:
    dictionary_path = os.path.join(
        args.output_path, f'{task}_char_dictionary.pkl')

    if os.path.exists(dictionary_path):
        print(f'{dictionary_path} exists, load cache')
        return CharDictionary.load_from_file(dictionary_path)

    dictionary = CharDictionary()

    def build(split: str):
        df = load_csv_from_dataset(task, split)

        for question in df['question']:
            dictionary.tokenize(
                question, add_word=True, extra_dict=glove.stoi if split == 'Test' else None)

        if task in MULTIPLE_CHOICE_TASKS:
            for answer_key in ['a1', 'a2', 'a3', 'a4', 'a5']:
                for answer in df[answer_key]:
                    dictionary.tokenize(
                        answer, add_word=True, extra_dict=glove.stoi if split == 'Test' else None)

    build('Train')
    build('Test')

    dictionary.dump_to_file(dictionary_path)
    return dictionary


def process_multiple_choice(args: Args, task: str, dictionary: Dictionary, char_dictionary: CharDictionary):
    def process(split: str):
        print(f'processing {task} {split}')
        data = []
        df = load_csv_from_dataset(task, split)

        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            answer_keys = ['a1', 'a2', 'a3', 'a4', 'a5']
            answers = [row[key] for key in answer_keys]
            gif_name = row['gif_name']
            answer_id = int(row['answer'])

            question_ids = dictionary.tokenize(question)
            question_chars = char_dictionary.tokenize(question)
            answer_ids = [dictionary.tokenize(answer)
                          for answer in answers]
            answer_chars = [char_dictionary.tokenize(
                answer) for answer in answers]

            data.append({
                'question': question,
                'answers': answers,
                'question_ids': question_ids,
                'question_chars': question_chars,
                'answer_ids': answer_ids,
                'answer_chars': answer_chars,
                'answer_id': answer_id,
                'gif_name': gif_name
            })

        filename = os.path.join(
            args.output_path, f'{split}_{task}_question.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    process('Train')
    process('Test')


if __name__ == "__main__":
    # args = Args()
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    glove = vocab.GloVe()

    if 'frameqa' in args.tasks:
        answer_dict = build_answer_dict(args)
        dictionary = build_dictionary(args, 'frameqa')
        char_dictionary = build_char_dictionary(args, 'frameqa')
        build_pretrained_embedding(args, 'frameqa', dictionary, glove)
        process_open_ended(args, 'frameqa', dictionary,
                           char_dictionary, answer_dict=answer_dict)

    if 'count' in args.tasks:
        dictionary = build_dictionary(args, 'count')
        char_dictionary = build_char_dictionary(args, 'count')
        build_pretrained_embedding(args, 'count', dictionary, glove)
        process_open_ended(args, 'count', dictionary, char_dictionary)

    if 'action' in args.tasks:
        dictionary = build_dictionary(args, 'action')
        char_dictionary = build_char_dictionary(args, 'action')
        build_pretrained_embedding(args, 'action', dictionary, glove)
        process_multiple_choice(args, 'action', dictionary, char_dictionary)

    if 'transition' in args.tasks:
        dictionary = build_dictionary(args, 'transition')
        char_dictionary = build_char_dictionary(args, 'transition')
        build_pretrained_embedding(args, 'transition', dictionary, glove)
        process_multiple_choice(
            args, 'transition', dictionary, char_dictionary)
