# Create Environment
Run conda env create -f environment1.yml to create the conda environment.

# Code
## ZuCo datasets.
Download ZuCo v1.0 and v2.0 to the address you specified.

### Preprocess datasets
run bash ./scripts/prepare_dataset_raw.sh to preprocess .mat files and prepare sentiment labels.

For each task, all .mat files will be converted into one .pickle file.
### Usage Example
To train an EEG-To-Text decoding model, run bash ./scripts/train_decoding_raw.sh.

To evaluate the trained EEG-To-Text decoding model from above, run bash ./scripts/eval_decoding_raw.sh.
