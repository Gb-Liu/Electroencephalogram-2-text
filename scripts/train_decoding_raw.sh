python train_decoding.py --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 10 \
    --num_epoch_step2 10 \
    -lr1 0.00005 \
    -lr2 0.00005 \
    -b 4 \
    -s ./checkpoints/decoding_raw/Conformer \
    -cuda "cuda:0"


#    bash ./scripts/train_decoding_raw.sh
#    bash ./scripts/eval_decoding_raw.sh