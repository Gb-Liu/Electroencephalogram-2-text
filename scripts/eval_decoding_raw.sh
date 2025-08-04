python eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding_raw/Conformer/last/NoConformer_task1_task2_taskNRv2_b8_layers0_5_5_5e-05_5e-05_unique_sent.pt \
    --config_path ./config/decoding_raw/NoConformer_task1_task2_taskNRv2_b8_layers0_5_5_5e-05_5e-05_unique_sent.json \
    -cuda cuda:0
