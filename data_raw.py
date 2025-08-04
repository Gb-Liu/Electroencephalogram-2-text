import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from scipy.signal import butter, lfilter
from scipy.signal import freqz


def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)

    return torch.Tensor(y).float()


def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    if std == 0:
        std == 1e-6
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def get_input_sample(sent_obj, tokenizer, eeg_type=['FFD', 'TRT', 'GD'],
                     bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=56, add_CLS_token=False, subj='unspecified', raw_eeg=False):
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        expected_bands = len(eeg_type) * len(bands)  # 3×8=24

        valid_data_found = False
        for et in eeg_type:
            if et not in word_obj['word_level_EEG']:
                print(f"Warning: EEG type '{et}' not found for word '{word_obj['content']}'")
                continue
            for band in bands:
                key = et + band
                data = word_obj['word_level_EEG'][et][key]
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    frequency_features.append(data)
                    valid_data_found = True

        eeg_matrix = np.array(frequency_features).T

        return_tensor = torch.from_numpy(eeg_matrix)
        normalized_tensor = torch.zeros_like(return_tensor)

        for col in range(return_tensor.shape[1]):
            normalized_tensor[:, col] = normalize_1d(return_tensor[:, col])
        return normalized_tensor

    if sent_obj is None:
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True,
                                 return_tensors='pt', return_attention_mask=True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    # get input embeddings
    word_embeddings = []
    word_raw_embeddings = []
    word_contents = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        # word_embeddings.append(torch.ones(104 * (bands)))
        word_embeddings.append(torch.ones(105, 24))

    skipped_words = 0

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)

        if word_level_eeg_tensor is None:
            # print(f"Skipping word '{word['content']}' in sentence: {sent_obj['content']}")
            skipped_words += 1
            word_embeddings.append(torch.ones(105, 24))
            continue
        if word_level_eeg_tensor is None:
            word_embeddings.append(torch.ones(105, 24))
            continue
            # return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            return None

        word_contents.append(word['content'])
        word_embeddings.append(word_level_eeg_tensor)

    if len(word_embeddings) == 0:
        # print(f"All words skipped in sentence: {sent_obj['content']}")
        return None

    if len(word_embeddings) < 1:
        return None

    # pad to max_len
    while len(word_embeddings) < max_len:
        # word_embeddings.append(torch.zeros(105 * len(bands)))
        word_embeddings.append(torch.zeros(105, 24))
        # if raw_eeg:
        #     word_raw_embeddings.append(torch.zeros(1, 104))

    word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length', max_length=max_len,
                                        truncation=True, return_tensors='pt', return_attention_mask=True)

    input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
    input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]  # bart

    input_sample['input_embeddings'] = torch.stack(word_embeddings)  # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word']) + 1] = torch.ones(
            len(sent_obj['word']) + 1)  # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))  # 1 is not masked

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word']) + 1] = torch.zeros(
            len(sent_obj['word']) + 1)  # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(
            len(sent_obj['word']))  # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    # subject
    input_sample['subject'] = subj

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type=['FFD', 'TRT', 'GD'],
                 bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                 setting='unique_sent', is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer

        # 定义任务名称映射表
        self.task_name_mapping = {
            0: 'task1-SR', 
            1: 'task2-NR',
            2: 'taskNRv2' 
        }

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        print(f'[INFO] Starting to process {len(input_dataset_dicts)} tasks for {phase} set:')
        # for i in range(len(input_dataset_dicts)):
        #    task_name = self.task_name_mapping.get(i, f'task{i + 1}')
        #    print(f'  Task {i + 1}: {task_name}')

        for i, input_dataset_dict in enumerate(input_dataset_dicts):
            current_task = self.task_name_mapping.get(i, f'task{i + 1}')

            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                if i == 0:
                    print(f'[SUBJECTS] Available subjects: {subjects}')
            else:
                subjects = [subject]

            total_num_sentence = len(input_dataset_dict[subjects[0]])
            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            print(f'[PROCESSING] Task {i + 1} ({current_task}):'
                  f'  Total sentences: {total_num_sentence},'
                  f'  Train range: 0-{train_divider},'
                  f'  Dev range: {train_divider}-{dev_divider},'
                  f'  Test range: {dev_divider}-{total_num_sentence}')

            prev_count = len(self.inputs)

            if setting == 'unique_sent':
                if phase == 'train':
                    # print(f'  - Loading train samples (0-{train_divider})...')
                    for key in subjects:
                        for j in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][j], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key,
                                                            )
                            if input_sample is not None:
                                input_sample['task'] = current_task  
                                self.inputs.append(input_sample)

                elif phase == 'dev':
                    # print(f'  - Loading dev samples ({train_divider}-{dev_divider})........')
                    for key in subjects:
                        for j in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][j], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key,
                                                            )
                            if input_sample is not None:
                                input_sample['task'] = current_task
                                self.inputs.append(input_sample)

                elif phase == 'test':
                    # print(f'  - Loading test samples ({dev_divider}-{total_num_sentence})...')
                    for key in subjects:
                        for j in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][j], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key,
                                                            )
                            if input_sample is not None:
                                input_sample['task'] = current_task
                                self.inputs.append(input_sample)

            added_count = len(self.inputs) - prev_count
            print(f'  [RESULT] Added {current_task} ({phase} set):'
                  f'  Samples added in this task: {added_count},'
                  f'  Total samples after this task: {len(self.inputs)},'
                  f'  Subjects included: {len(set(sample["subject"] for sample in self.inputs if sample["task"] == current_task))}')

        print(f'[SUMMARY] Completed processing all tasks for {phase} set:  '
              f' Total tasks processed: {len(input_dataset_dicts)} ,'
              f' Final dataset size: {len(self.inputs)} ,'
              f' Unique subjects: {len(set(sample["subject"] for sample in self.inputs))}')
        print(f' Samples distribution:')
        # for task_name in self.task_name_mapping.values():
        #    count = len([x for x in self.inputs if x.get("task") == task_name])
        #   if count > 0:
        #       print(f'    - {task_name}: {count} samples')
        print()

        # print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size())
        # print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            # input_sample['sentiment_label'],
            # input_sample['sent_level_EEG'],
            # input_sample['input_raw_embeddings'],
            input_sample['word_contents'],
            input_sample['word_contents_attn'],
            input_sample['subject']
        )


'''sanity test'''
if __name__ == '__main__':

    check_dataset = 'ZuCo'  # 'stanford_sentiment'

    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []

        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset_wRaw.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        dataset_path_task2_v2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset_wRaw.pickle'
        with open(dataset_path_task2_v2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

        print()
        for key in whole_dataset_dicts[0]:
            print(f'task2_v2, sentence num in {key}:', len(whole_dataset_dicts[0][key]))
        print()

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dataset_setting = 'unique_sent'
        subject_choice = 'ALL'
        print(f'![Debug]using {subject_choice}')
        eeg_type_choice = ['FFD', 'TRT', 'GD']
        print(f'[INFO]eeg type {eeg_type_choice}')
        bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
        print(f'[INFO]using bands {bands_choice}')
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                                 eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice,
                               bands=bands_choice, setting=dataset_setting, )
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                                eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)

        print('trainset size:', len(train_set))
        print('devset size:', len(dev_set))
        print('testset size:', len(test_set))
