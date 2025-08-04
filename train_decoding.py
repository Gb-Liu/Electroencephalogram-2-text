import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pickle
import json
import pandas as pd
from glob import glob
import time
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertConfig, \
    BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoConfig

from data_raw import ZuCo_dataset
from model2 import BrainTranslator
from config import get_config

from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score

import warnings
warnings.filterwarnings('ignore')
import os
from transformers import logging

logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)

from torch.utils.tensorboard import SummaryWriter


LOG_DIR = "runs_h"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "train_full"))
dev_writer = SummaryWriter(os.path.join(LOG_DIR, "dev_full"))

SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
            'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 'YTL', 'YSL', 'YRP',
            'YAG', 'YDR', 'YAK']


def train_model(dataloaders, device, model, criterion, optimizer, scheduler, tokenizer, dataset_sizes, num_epochs=25,
                checkpoint_path_best='./checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b8_25_25_5e-05_5e-05_unique_sent.pt',
                checkpoint_path_last='./checkpoints/decoding_raw/last/temp_decoding.pt', stepone=False):

    since = time.time()
    best_loss = 100000000000
    train_losses = []
    val_losses = []
    test_losses = []

    index_plot = 0
    index_plot_dev = 0

    num_kernels = model.conv_module.num_kernels
    num_layers = model.num_layers

    # 用于保存每一轮的评估指标
    results = {
        'epoch': [],
        'train_loss': [], 'dev_loss': [], 'test_loss': [],
        'rouge_1_r': [], 'rouge_1_p': [], 'rouge_1_f': [],
        'rouge_2_r': [], 'rouge_2_p': [], 'rouge_2_f': [],
        'rouge_l_r': [], 'rouge_l_p': [], 'rouge_l_f': [],
        'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
        'bertscore_P': [], 'bertscore_R': [], 'bertscore_F1': []  }

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}' )
        print('-' * 100)

        train_epoch_loss = None
        dev_epoch_loss = None
        test_epoch_loss = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if phase == 'test':
                target_tokens_list = []
                target_string_list = []
                pred_tokens_list = []
                pred_string_list = []

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch", dynamic_ncols=True, mininterval=5.0) as tepoch:
                for batch_idx, (
                        input_embeddings, seq_len, input_masks, input_mask_invert,
                        target_ids, target_mask,
                        word_contents, word_contents_attn, subject_batch
                        ) in enumerate(tepoch):

                    input_embeddings_batch = input_embeddings.float().to(device)
                    input_masks_batch = torch.stack(input_masks, 0).to(device)
                    input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device)
                    target_ids_batch = torch.stack(target_ids, 0).to(device)
                    word_contents_batch = torch.stack(word_contents, 0).to(device)
                    word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device)

                    subject_batch = np.array(subject_batch)

                    target_string_list_bertscore = []

                    if phase == 'test' and stepone == False:
                        target_tokens = tokenizer.convert_ids_to_tokens(
                            target_ids_batch[0].tolist(), skip_special_tokens=True)
                        target_string = tokenizer.decode(
                            target_ids_batch[0], skip_special_tokens=True)
                        # add to list for later calculate metrics
                        target_tokens_list.append([target_tokens])
                        target_string_list.append(target_string)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                        target_ids_batch, seq_len, word_contents_batch, word_contents_attn_batch,
                        stepone, subject_batch , device=device)

                    """replace padding ids in target_ids with -100"""
                    target_ids_batch[target_ids_batch ==
                                     tokenizer.pad_token_id] = -100

                    """calculate loss"""
                    if stepone == True:
                        loss = seq2seqLMoutput
                    else:
                        loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long())

                    if phase == 'test' and stepone == False:
                        logits = seq2seqLMoutput
                        probs = logits[0].softmax(dim=1)
                        values, predictions = probs.topk(1)
                        predictions = torch.squeeze(predictions)
                        predicted_string = tokenizer.decode(predictions).split(
                            '</s></s>')[0].replace('<s>', '')

                        # convert to int list
                        predictions = predictions.tolist()
                        truncated_prediction = []
                        for t in predictions:
                            if t != tokenizer.eos_token_id:
                                truncated_prediction.append(t)
                            else:
                                break
                        pred_tokens = tokenizer.convert_ids_to_tokens(
                            truncated_prediction, skip_special_tokens=True)
                        pred_tokens_list.append(pred_tokens)
                        pred_string_list.append(predicted_string)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss

                    # 获取当前学习率（从优化器获取，兼容所有调度器）
                    current_lr = optimizer.param_groups[0]['lr']
                    tepoch.set_postfix(loss=loss.item(), lr=current_lr)

                    if phase == 'train':
                        val_writer.add_scalar("train_full", loss.item(), index_plot)
                        index_plot += 1
                    if phase == 'dev':
                        dev_writer.add_scalar("dev_full", loss.item(), index_plot_dev)
                        index_plot_dev += 1

                    # 仅对CyclicLR在训练阶段更新（ReduceLROnPlateau在验证后更新）
                    if phase == 'train' and not isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_losses.append(epoch_loss)
                torch.save(model.state_dict(), checkpoint_path_last)
            elif phase == 'dev':
                dev_epoch_loss = epoch_loss
                val_losses.append(epoch_loss)
                # 对ReduceLROnPlateau，在验证阶段传入验证损失更新
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(dev_epoch_loss)
            elif phase == 'test':
                test_epoch_loss = epoch_loss
                test_losses.append(epoch_loss)

            if phase == 'train':
                train_writer.add_scalar("train", epoch_loss, epoch)
            elif phase == 'dev':
                train_writer.add_scalar("val", epoch_loss, epoch)

            # 每个epoch结束后统一记录对比指标
            if train_epoch_loss is not None and dev_epoch_loss is not None:
                train_writer.add_scalars('loss train/val', {
                    'train': train_epoch_loss,
                    'val': dev_epoch_loss,
                }, epoch)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')

            if phase == 'test' and stepone == False:
                print("Evaluation on test")

                # Filter out empty strings
                filtered_pred_string_list = [s for s in pred_string_list if s.strip()]
                filtered_target_string_list = [s for s in target_string_list if s.strip()]

                # Calculate ROUGE scores safely
                try:
                    rouge = Rouge()
                    if filtered_pred_string_list and filtered_target_string_list:
                        rouge_scores = rouge.get_scores(filtered_pred_string_list, filtered_target_string_list,
                                                        avg=True, ignore_empty=True)
                    else:
                        rouge_scores = {
                            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                        }
                    print('ROUGE scores:', rouge_scores)
                except Exception as e:
                    print(f"ROUGE calculation failed: {e}")
                    rouge_scores = {
                        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    }

                # Calculate BLEU scores safely
                try:
                    if target_tokens_list and pred_tokens_list:
                        bleu_1 = corpus_bleu(target_tokens_list, pred_tokens_list, weights=(1.0, 0, 0, 0))
                        bleu_2 = corpus_bleu(target_tokens_list, pred_tokens_list, weights=(0.5, 0.5, 0, 0))
                        bleu_3 = corpus_bleu(target_tokens_list, pred_tokens_list, weights=(0.33, 0.33, 0.33, 0))
                        bleu_4 = corpus_bleu(target_tokens_list, pred_tokens_list, weights=(0.25, 0.25, 0.25, 0.25))
                    else:
                        bleu_1, bleu_2, bleu_3, bleu_4 = 0.0, 0.0, 0.0, 0.0
                    print(f'BLEU-1: {bleu_1}, BLEU-2: {bleu_2}, BLEU-3: {bleu_3}, BLEU-4: {bleu_4}')
                except Exception as e:
                    print(f"BLEU calculation failed: {e}")
                    bleu_1, bleu_2, bleu_3, bleu_4 = 0.0, 0.0, 0.0, 0.0

                # Calculate BERTScore safely
                # 确保过滤后的候选文本和参考文本数量一致
                filtered_pairs = [(p, r) for p, r in zip(pred_string_list, target_string_list)
                                  if p.strip() and r.strip()]
                filtered_pred_string_list = [p for p, r in filtered_pairs]
                filtered_target_string_list = [r for p, r in filtered_pairs]

                # 打印过滤后的样本数量
                print(f"过滤后: {len(filtered_pred_string_list)} 个有效样本")

                # 计算BERTScore safely
                try:
                    if filtered_pred_string_list and filtered_target_string_list:
                        # 检查数量是否匹配
                        if len(filtered_pred_string_list) != len(filtered_target_string_list):
                            print(
                                f"警告: 过滤后候选文本({len(filtered_pred_string_list)})和参考文本({len(filtered_target_string_list)})数量不一致")
                            min_len = min(len(filtered_pred_string_list), len(filtered_target_string_list))
                            filtered_pred_string_list = filtered_pred_string_list[:min_len]
                            filtered_target_string_list = filtered_target_string_list[:min_len]
                            print(f"已截断为相同长度: {min_len}")

                        P, R, F1 = score(filtered_pred_string_list, filtered_target_string_list,
                                         lang='en', device=device, model_type="bert-large-uncased")
                        print(
                            f"BERTScore P: {np.mean(np.array(P)):.4f}, R: {np.mean(np.array(R)):.4f}, F1: {np.mean(np.array(F1)):.4f}")
                    else:
                        print("警告: 过滤后无有效样本用于BERTScore计算")
                        P, R, F1 = [0.0], [0.0], [0.0]
                except Exception as e:
                    print(f"BERTScore计算失败: {e}")
                    import traceback
                    traceback.print_exc()
                    P, R, F1 = [0.0], [0.0], [0.0]

                # 保存每一轮的评估指标
                results['epoch'].append(epoch + 1)
                results['train_loss'].append(train_epoch_loss)
                results['dev_loss'].append(dev_epoch_loss)
                results['test_loss'].append(test_epoch_loss)
                results['rouge_1_r'].append(rouge_scores['rouge-1']['r'])
                results['rouge_1_p'].append(rouge_scores['rouge-1']['p'])
                results['rouge_1_f'].append(rouge_scores['rouge-1']['f'])
                results['rouge_2_r'].append(rouge_scores['rouge-2']['r'])
                results['rouge_2_p'].append(rouge_scores['rouge-2']['p'])
                results['rouge_2_f'].append(rouge_scores['rouge-2']['f'])
                results['rouge_l_r'].append(rouge_scores['rouge-l']['r'])
                results['rouge_l_p'].append(rouge_scores['rouge-l']['p'])
                results['rouge_l_f'].append(rouge_scores['rouge-l']['f'])
                results['bleu_1'].append(bleu_1)
                results['bleu_2'].append(bleu_2)
                results['bleu_3'].append(bleu_3)
                results['bleu_4'].append(bleu_4)
                results['bertscore_P'].append(np.mean(np.array(P)))
                results['bertscore_R'].append(np.mean(np.array(R)))
                results['bertscore_F1'].append(np.mean(np.array(F1)))

        print()
        # 构建文件名
        results_file_name = f"Conformer_{skip_step_one}_b{batch_size}_ker{num_kernels}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_training_results.xlsx"
        #results_file_name = f"Conformer-no-conv_{skip_step_one}_b{batch_size}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_training_results.xlsx"
        # 保存每一轮的结果到Excel文件
        df = pd.DataFrame(results)
        df.to_excel(f"./results/{results_file_name}", index=False)

    print(f"Train losses: {train_losses}")
    print(f"Val losses: {val_losses}")
    print(f"Test losses: {test_losses}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    return model,test_losses

if __name__ == '__main__':
    args = get_config('train_decoding')
    ''' config param'''
    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = False

    if use_random_init and skip_step_one:
        step2_lr = 5 * 1e-4
    print(f'[INFO]using model: {model_name}')
    print(f'[INFO]using use_random_init: {use_random_init}')

    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

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
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    tasks_loaded = []

    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        tasks_loaded.append('task1')
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset_wRaw.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        tasks_loaded.append('task2')
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset_wRaw.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        tasks_loaded.append('task3')
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset_wRaw.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        tasks_loaded.append('taskNRv2')
        print(f'[INFO]loaded {len(tasks_loaded)} tasks: {", ".join(tasks_loaded)}')
    print()

    if model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                             eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice,
                           eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                            eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(
        dev_set), 'test': len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))

    # Allows to pad and get real size of eeg vectors
    def pad_and_sort_batch(data_loader_batch):
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, word_contents, word_contents_attn, subject = tuple(
            zip(*data_loader_batch))
        input_embeddings_padded = pad_sequence(input_embeddings, batch_first=True, padding_value=0)

        return input_embeddings_padded, seq_len, input_masks, input_mask_invert, target_ids, target_mask,  word_contents, word_contents_attn, subject

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, collate_fn=pad_and_sort_batch)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_and_sort_batch)
    # test dataloader
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_and_sort_batch)
    # dataloaders
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader, 'test': test_dataloader}

    ''' set up model '''
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        model = BrainTranslator(pretrained, in_feature=1024, decoder_embedding_size=1024,
                                additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)
    model.to(device)

    num_kernels = model.conv_module.num_kernels
    num_layers = model.num_layers

    if skip_step_one:
        save_name = f'Conformer_{skip_step_one}_{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_ker{num_kernels}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'Conformer_{skip_step_one}_{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_ker{num_kernels}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    #if skip_step_one:
        #save_name = f'Conformer-no-conv_{skip_step_one}_{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    #else:
        #save_name = f'Conformer-no-conv_{skip_step_one}_{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_layers{num_layers}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt'
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt'

    """save config"""
    with open(f'./config/decoding_raw/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    ''' training loop '''

    ######################################################
    '''step one trainig'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = './checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_2steptraining_b8_25_25_5e-05_5e-05_unique_sent.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:
        model.to(device)
        test_losses = []

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.CyclicLR(optimizer_step1, base_lr=step1_lr, max_lr=5e-4, mode="triangular2")

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()
        model.freeze_pretrained_bart()

        print('=== start Step1 training ... ===')
        # return best loss model from step1 training
        model, test_losses = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1,
                            tokenizer=tokenizer, dataset_sizes=dataset_sizes,num_epochs=num_epochs_step1,
                            checkpoint_path_best=output_checkpoint_name_best,
                            checkpoint_path_last=output_checkpoint_name_last, stepone=True)

        train_writer.flush()
        train_writer.close()
        val_writer.flush()
        val_writer.close()
        dev_writer.flush()
        dev_writer.close()

    ######################################################
    '''step two trainig'''
    ######################################################

    model.freeze_pretrained_brain()

    ''' set up optimizer and scheduler'''
    #optimizer_step2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step2_lr, momentum=0.9,weight_decay=1e-5)

    # ReduceLROnPlateau
    #exp_lr_scheduler_step2 = ReduceLROnPlateau(optimizer_step2,mode='min', factor=0.05, patience=1, threshold=0.005,cooldown=1,min_lr=5e-6 )

    optimizer_step2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.CyclicLR(optimizer_step2, base_lr=0.0000005, max_lr=0.00005, mode="triangular2")

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 training ... ===')
    model.to(device)

    '''main loop'''
    trained_model, test_losses_step2 = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2,tokenizer=tokenizer,
                                dataset_sizes=dataset_sizes,num_epochs=num_epochs_step2,
                                checkpoint_path_best=output_checkpoint_name_best,
                                checkpoint_path_last=output_checkpoint_name_last, stepone=False)

    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()
    dev_writer.flush()
    dev_writer.close()