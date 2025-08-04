import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from transformers import BartTokenizer
from scipy.signal import butter, lfilter, freqz


def butter_bandpass_filter(signal, lowcut, highcut, fs=500, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return torch.Tensor(y).float()


def normalize_1d(input_tensor):
    # 归一化一维张量
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    if std == 0:
        std = 1e-6  # 避免除零错误
    return (input_tensor - mean) / std


def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                     max_len=56, add_CLS_token=False, subj='unspecified'):
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        # 仅处理 'GD' 类型，每个 band 对应 (105,) 数据
        frequency_features = []
        # 遍历 8 个频带（bands）
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type + band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105 * len(bands):
            return None
        # 转换为tensor并归一化
        return_tensor = torch.from_numpy(word_eeg_embedding).float()
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    feature_dim = 840
    input_sample = {}

    target_string = sent_obj['content']

    # 分词并填充到max_len
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len,
                                 truncation=True, return_tensors='pt', return_attention_mask=True)

    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # 替换异常文本（保持原逻辑）
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]

    # 处理单词级EEG嵌入
    word_embeddings = []
    word_contents = []

    # 添加CLS token（如果需要）
    if add_CLS_token:
        word_embeddings.append(torch.ones(104 * len(bands)))  # 840

    # 遍历句子中的每个单词
    for word in sent_obj['word']:
        # 获取当前单词的GD类型EEG特征
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type='GD', bands=bands)
        if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
            return None
        word_embeddings.append(word_level_eeg_tensor)
        word_contents.append(word['content'])

    # 验证有效单词数量
    if len(word_embeddings) < 1:
        return None

    # 填充到max_len（不足则补零向量）
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(840))  # 维度840

    # 处理单词内容的分词（用于辅助信息）
    word_contents_tokenized = tokenizer(' '.join(word_contents), padding='max_length',
                                        max_length=max_len, truncation=True, return_tensors='pt',
                                        return_attention_mask=True)

    input_sample['word_contents'] = word_contents_tokenized['input_ids'][0]
    input_sample['word_contents_attn'] = word_contents_tokenized['attention_mask'][0]

    # 堆叠所有单词嵌入 → 形状为 (max_len, 840)
    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    # 注意力掩码（区分有效单词和填充）
    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0表示掩码

    if add_CLS_token:
        # 有效长度 = 单词数 + 1（CLS），但不超过max_len
        valid_len = min(len(sent_obj['word']) + 1, max_len)
        input_sample['input_attn_mask'][:valid_len] = torch.ones(valid_len)
    else:
        # 有效长度 = 单词数，不超过max_len
        valid_len = min(len(sent_obj['word']), max_len)
        input_sample['input_attn_mask'][:valid_len] = torch.ones(valid_len)

    # 反向掩码（部分模型需要1表示掩码）
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 初始化为1（掩码）

    if add_CLS_token:
        valid_len = min(len(sent_obj['word']) + 1, max_len)
        input_sample['input_attn_mask_invert'][:valid_len] = torch.zeros(valid_len)  # 有效部分设为0（不掩码）
    else:
        valid_len = min(len(sent_obj['word']), max_len)
        input_sample['input_attn_mask_invert'][:valid_len] = torch.zeros(valid_len)
    # 句子长度和被试信息
    input_sample['seq_len'] = len(sent_obj['word'])
    input_sample['subject'] = subj

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD',  # 固定eeg_type为'GD'
                 bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                 setting='unique_sent', is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer
        # self.task_name_mapping = {0: 'task1-SR', 1: 'task2-NR', 2: 'taskNRv2'}

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        print(f'[INFO] Processing {len(input_dataset_dicts)} tasks for {phase} set (using only GD EEG type)')

        for input_dataset_dict in input_dataset_dicts:
            # 确定被试列表（全部或指定单个）
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print(f'[SUBJECTS] Available: {subjects}')
            else:
                subjects = [subject]

            # 划分训练/验证/测试集（按句子索引）
            total_sents = len(input_dataset_dict[subjects[0]])
            train_div = int(0.8 * total_sents)
            dev_div = train_div + int(0.1 * total_sents)

            # 按阶段加载数据
            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_div):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_div, dev_div):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'all':
                    print('[INFO]initializing all dataset...')
                    for key in subjects:
                        for i in range(int(1 * total_sents)):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_div, total_sents):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_sents):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_sents):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_sents):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type,
                                                            bands=bands, add_CLS_token=is_add_CLS_token, subj=key)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

        print(
            f'[SUMMARY] {phase} set size: {len(self.inputs)}, unique subjects: {len(set(s["subject"] for s in self.inputs))}')
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = self.inputs[idx]
        return (
            sample['input_embeddings'],
            sample['seq_len'],
            sample['input_attn_mask'],
            sample['input_attn_mask_invert'],
            sample['target_ids'],
            sample['target_mask'],
            sample['word_contents'],
            sample['word_contents_attn'],
            sample['subject']
        )


# 测试代码
if __name__ == '__main__':
    check_dataset = 'ZuCo'
    if check_dataset == 'ZuCo':
        whole_dataset_dicts = []
        # 加载数据集（根据实际路径调整）
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle'
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset_wRaw.pickle'
        dataset_path_task2_v2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset_wRaw.pickle'
        with open(dataset_path_task1, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))
        with open(dataset_path_task2, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))
        with open(dataset_path_task2_v2, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))

        # 打印数据集基本信息
        print("Task 1 subjects and sentence counts:")
        for subj in whole_dataset_dicts[0]:
            print(f"  {subj}: {len(whole_dataset_dicts[0][subj])} sentences")

        # 初始化tokenizer和数据集
        tokenizer = BartTokenizer.from_pretrained('./pretrained-models/bart-large')
        dataset_setting = 'unique_sent'
        eeg_type_choice = 'GD'  # 仅使用GD类型
        bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
        print(f"Using EEG type: {eeg_type_choice}, bands: {bands_choice}")

        # 加载各阶段数据集
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject='ALL',
                                 eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject='ALL',
                               eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
        test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject='ALL',
                                eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)

        print(f"Train size: {len(train_set)}, Dev size: {len(dev_set)}, Test size: {len(test_set)}")