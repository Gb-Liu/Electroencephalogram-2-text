import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob

from transformers import BartTokenizer, BartForConditionalGeneration
from data_no_conformer import ZuCo_dataset
from model_noconv import BrainTranslator
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config

from torch.nn.utils.rnn import pad_sequence
from openai import OpenAI  

# 设置DeepSeek API
deepseek_client = OpenAI(
    api_key="",  
    base_url="https://api.deepseek.com/v1"
)


def deepseek_refinement(corrupted_text):
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "作为文本重构器，你的任务是以最小改动恢复损坏句子到原始形式。请根据需要调整空格和标点符号。不要添加额外信息。如果无法重构文本，请回复[False]。"},
                {"role": "user", "content": f"重构以下文本: [{corrupted_text}]"}
            ],
            temperature=0.2,
            max_tokens=256
        )
        output = response.choices[0].message.content.strip()
        output = output.replace('[', '').replace(']', '')

        if len(output) < 10 and 'False' in output:
            return corrupted_text
        return output
    except Exception as e:
        print(f"DeepSeek API错误: {e}")
        return corrupted_text


def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='./results_raw/temp.txt', stepone=False):
    use_deepseek = False 

    print("Saving to: ", output_all_results_path)
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    refine_tokens_list = []
    refine_string_list = []

    with open(output_all_results_path, 'w', encoding='utf-8') as f:

        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, word_contents, word_contents_attn, subject_batch in dataloaders['test']:
            # load in batch
            input_embeddings_batch = input_embeddings.float().to(device)
            input_masks_batch = torch.stack(input_masks, 0).to(device)
            input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
            target_ids_batch = torch.stack(target_ids, 0).to(device)
            word_contents_batch = torch.stack(word_contents, 0).to(device)
            word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)
            subject_batch = np.array(subject_batch)

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)

            f.write(f'target string: {target_string}\n')
            target_tokens_string = "["
            for el in target_tokens:
                target_tokens_string = target_tokens_string + str(el) + " "
            target_tokens_string += "]"
            #f.write(f'target tokens: {target_tokens_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # forward
            seq2seqLMoutput = model(
                input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                target_ids_batch, seq_len, word_contents_batch, word_contents_attn_batch,
                stepone, subject_batch, device=device)

            """calculate loss"""
            loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long())

            # get predicted tokens
            logits = seq2seqLMoutput
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            f.write(f'predicted string: {predicted_string}\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)

            if use_deepseek:
                predicted_string_refined = deepseek_refinement(predicted_string).replace('\n', '')
                f.write(f'refined string: {predicted_string_refined}\n')

                refined_input_ids = tokenizer.encode(
                    predicted_string_refined,
                    return_tensors='pt',
                    add_special_tokens=False
                )[0].tolist()

                refine_tokens = tokenizer.convert_ids_to_tokens(
                    refined_input_ids,
                    skip_special_tokens=True
                )

                refine_tokens_list.append(refine_tokens)
                refine_string_list.append(predicted_string_refined)

            pred_tokens_string = "["
            for el in pred_tokens:
                pred_tokens_string = pred_tokens_string + str(el) + " "
            pred_tokens_string += "]"
            #f.write(f'predicted tokens (truncated): {pred_tokens_string}\n')
            f.write(f'################################################\n\n\n')

            sample_count += 1
            # statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss

    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    print("Predicted outputs")
    """ calculate corpus bleu score """
    weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print(rouge_scores)
    print()
    """ calculate bertscore"""
    from bert_score import score
    P, R, F1 = score(pred_string_list, target_string_list, lang='en', device="cuda:0", model_type="bert-large-uncased")
    print(f"bert_score P / R / F1: {np.mean(np.array(P))},{np.mean(np.array(R))},{np.mean(np.array(F1))}")
    # print(f"bert_score R: {np.mean(np.array(R))}")
    # print(f"bert_score F1: {np.mean(np.array(F1))}")
    print("*************************************")

    if use_deepseek:
        print()
        print("Refined outputs with DeepSeek")  
        """ calculate corpus bleu score """
        weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu(target_tokens_list, refine_tokens_list, weights=weight)
            print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print()
        """ calculate rouge score """
        rouge = Rouge()
        rouge_scores = rouge.get_scores(refine_string_list, target_string_list, avg=True)
        print(rouge_scores)
        print()
        """ calculate bertscore"""
        from bert_score import score
        P, R, F1 = score(refine_string_list, target_string_list, lang='en', device="cuda:0",
                         model_type="bert-large-uncased")
        print(f"bert_score P: {np.mean(np.array(P))}")
        print(f"bert_score R: {np.mean(np.array(R))}")
        print(f"bert_score F1: {np.mean(np.array(F1))}")
        print("*************************************")
        print("*************************************")

# Allows to pad and get real size of eeg vectors
def pad_and_sort_batch(data_loader_batch):
    input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, word_contents, word_contents_attn, subject = tuple(
        zip(*data_loader_batch))

    input_embeddings_padded = pad_sequence(
        input_embeddings,
        batch_first=True,
        padding_value=0
    )
    return input_embeddings_padded, seq_len, input_masks, input_mask_invert, target_ids, target_mask, word_contents, word_contents_attn, subject



if __name__ == '__main__':
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))

    batch_size = 1

    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')

    dataset_setting = 'unique_sent'
    task_name = training_config['task_name']
    model_name = training_config['model_name']

    output_all_results_path = f'./results_raw/{task_name}-{model_name}-all_1_decoding_results.txt'
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice,
                            bands=bands_choice, setting=dataset_setting)

    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO]test_set size: ', len(test_set))

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=pad_and_sort_batch)

    dataloaders = {'test': test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=840, decoder_embedding_size=1024,
                                additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    ''' eval '''

    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path=output_all_results_path)

