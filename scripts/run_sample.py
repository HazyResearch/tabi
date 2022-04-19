import subprocess

# set hyperparameters

# number of epochs for each round of sampling
n_epochs = 1

# first epoch is in-batch negatives
num_neg_rounds = 3

# tabi-specific
type_weight = 0.1

# model params
max_context_length = 32
lr = 1e-5
temperature = 0.05
add_types_in_desc = True
seed = 1234
batch_size = 32
eval_batch_size = 32
neg_sample_batch_size = 32
entity_batch_size = 32
num_negatives = 1  # number of hard negatives to use for training
num_negatives_orig = 20  # total number of hard negatives to fetch (fetch extra since we filter gold ids and optionally based on counts)
filter_negatives = False  # whether to filter hard negatives

# machine params
gpus = "0"
ngpus = 1
distributed = False

# set paths
home_dir = "tabi"
data_dir = f"sample_data"
train_file = "train.jsonl"
dev_file = "dev.jsonl"
test_file = "dev.jsonl"
entity_file = "entity.jsonl"
type_file = "figer_types.txt"

base_log_dir = "logs"
run_name = f"sample_run"

log_dir = f"{base_log_dir}/{run_name}"

preprocess = True
tokenized_entity_data = f"{log_dir}/1_preprocess/entity_data.npy"

python_str = "python"
if distributed:
    python_str = f"python -m torch.distributed.launch --nproc_per_node={ngpus}"

# preprocess entity data (generate and save tokens for BERT entity input)
if preprocess:
    subprocess.run(
        f"python {home_dir}/preprocess_entity.py \
        --batch_size {batch_size} \
        --add_entity_type_in_description {add_types_in_desc} \
        --log_dir {log_dir}/1_preprocess \
        --type_file {data_dir}/{type_file} \
        --num_workers 12 \
        --entity_file {data_dir}/{entity_file}",
        shell=True,
        check=True,
    )

# train model
subprocess.run(
    f"CUDA_VISIBLE_DEVICES={gpus} {python_str} {home_dir}/train.py \
    --train_data_file {data_dir}/{train_file} \
    --train_entity_file {data_dir}/{entity_file} \
    --dev_data_file {data_dir}/{dev_file} \
    --dev_entity_file {data_dir}/{entity_file} \
    --type_file {data_dir}/{type_file} \
    --n_epochs {n_epochs} \
    --log_dir {log_dir}/1_train \
    --temperature {temperature} \
    --batch_size {batch_size} \
    --add_entity_type_in_description {add_types_in_desc} \
    --distributed {distributed} \
    --tokenized_entity_data {tokenized_entity_data} \
    --max_context_length {max_context_length} \
    --alpha {type_weight} \
    --seed {seed} \
    --lr {lr}",
    shell=True,
    check=True,
)

# generate entities
subprocess.run(
    f"CUDA_VISIBLE_DEVICES={gpus} python {home_dir}/extract_entity.py \
    --entity_file {data_dir}/{entity_file} \
    --model_checkpoint {log_dir}/1_train/best_model.pth \
    --batch_size {entity_batch_size} \
    --log_dir {log_dir}/1_entity \
    --add_entity_type_in_description {add_types_in_desc} \
    --distributed {distributed} \
    --tokenized_entity_data {tokenized_entity_data} \
    --type_file {data_dir}/{type_file}",
    shell=True,
    check=True,
)

# eval
subprocess.run(
    f"CUDA_VISIBLE_DEVICES={gpus} python {home_dir}/eval.py \
    --test_data_file {data_dir}/{test_file} \
    --entity_file {data_dir}/{entity_file} \
    --log_dir {log_dir}/1_eval \
    --temperature {temperature} \
    --batch_size {eval_batch_size} \
    --model_checkpoint {log_dir}/1_train/best_model.pth \
    --entity_emb_path {log_dir}/1_entity/embs.npy \
    --entity_map_file {log_dir}/1_entity/entity_map.pkl \
    --add_entity_type_in_description {add_types_in_desc} \
    --max_context_length {max_context_length} \
    --mode eval",
    shell=True,
    check=True,
)

# hard negative sampling rounds
for round in range(1, num_neg_rounds + 1):

    # decay the lr by 1/2 each round
    neg_lr = lr * (0.5) ** (round)

    # generate train negative samples
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={gpus} {python_str} {home_dir}/eval.py \
        --test_data_file {data_dir}/{train_file} \
        --entity_file {data_dir}/{entity_file} \
        --log_dir {log_dir}/{round}_neg_sample_train \
        --temperature {temperature} \
        --batch_size {neg_sample_batch_size} \
        --model_checkpoint {log_dir}/{round}_train/best_model.pth \
        --entity_emb_path {log_dir}/{round}_entity/embs.npy \
        --entity_map_file {log_dir}/{round}_entity/entity_map.pkl \
        --add_entity_type_in_description {add_types_in_desc} \
        --max_context_length {max_context_length} \
        --distributed {distributed} \
        --top_k {num_negatives} \
        --type_file {data_dir}/{type_file} \
        --orig_num_negatives {num_negatives_orig} \
        --filter_negatives {filter_negatives} \
        --mode neg_sample",
        shell=True,
        check=True,
    )

    # generate dev negative samples
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={gpus} {python_str} {home_dir}/eval.py \
        --test_data_file {data_dir}/{dev_file} \
        --entity_file {data_dir}/{entity_file} \
        --log_dir {log_dir}/{round}_neg_sample_dev \
        --temperature {temperature} \
        --batch_size {neg_sample_batch_size} \
        --model_checkpoint {log_dir}/{round}_train/best_model.pth \
        --entity_emb_path {log_dir}/{round}_entity/embs.npy \
        --entity_map_file {log_dir}/{round}_entity/entity_map.pkl \
        --max_context_length {max_context_length} \
        --add_entity_type_in_description {add_types_in_desc} \
        --distributed {distributed} \
        --top_k {num_negatives} \
        --type_file {data_dir}/{type_file} \
        --mode neg_sample",
        shell=True,
        check=True,
    )

    # train model
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={gpus} {python_str} {home_dir}/train.py \
        --seed {seed} \
        --train_data_file {data_dir}/{train_file} \
        --train_entity_file {data_dir}/{entity_file} \
        --dev_data_file {data_dir}/{dev_file} \
        --dev_entity_file {data_dir}/{entity_file} \
        --type_file {data_dir}/{type_file} \
        --n_epochs {n_epochs} \
        --log_dir {log_dir}/{round+1}_train \
        --temperature {temperature} \
        --batch_size {batch_size} \
        --add_entity_type_in_description {add_types_in_desc} \
        --distributed {distributed} \
        --tokenized_entity_data {tokenized_entity_data} \
        --alpha {type_weight} \
        --max_context_length {max_context_length} \
        --num_negatives {num_negatives} \
        --model_checkpoint {log_dir}/{round}_train/best_model.pth \
        --train_neg_sample_file {log_dir}/{round}_neg_sample_train/neg_samples.json \
        --dev_neg_sample_file {log_dir}/{round}_neg_sample_dev/neg_samples.json \
        --lr {neg_lr}",
        shell=True,
        check=True,
    )

    # generate entities for eval
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={gpus} python {home_dir}/extract_entity.py \
        --entity_file {data_dir}/{entity_file} \
        --model_checkpoint {log_dir}/{round+1}_train/best_model.pth \
        --batch_size {entity_batch_size} \
        --log_dir {log_dir}/{round+1}_entity \
        --add_entity_type_in_description {add_types_in_desc} \
        --tokenized_entity_data {tokenized_entity_data} \
        --type_file {data_dir}/{type_file} \
        --distributed {distributed}",
        shell=True,
        check=True,
    )

    # eval
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={gpus} python {home_dir}/eval.py \
        --test_data_file {data_dir}/{test_file} \
        --entity_file {data_dir}/{entity_file} \
        --log_dir {log_dir}/{round+1}_eval \
        --temperature {temperature} \
        --batch_size {eval_batch_size} \
        --model_checkpoint {log_dir}/{round+1}_train/best_model.pth \
        --entity_emb_path {log_dir}/{round+1}_entity/embs.npy \
        --max_context_length {max_context_length} \
        --entity_map_file {log_dir}/{round+1}_entity/entity_map.pkl \
        --add_entity_type_in_description {add_types_in_desc} \
        --mode eval",
        shell=True,
        check=True,
    )
