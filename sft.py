import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer # 自动加载因果语言模型
from data import D3Dataset, SFTData
import random
from datasets import Dataset as HFDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 学习率调度器
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,# 采样数量（-1表示全部）
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 2,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 64,  # 确保这里写了 : int = 64
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,   # 序列最大长度
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    
    category: str="",
    train_from_scratch: bool = False, # 是否从头开始训练

):
    # 1. 设置随机种子
    set_seed(seed)
    # 2. 设置WandB项目
    os.environ['WANDB_PROJECT'] = wandb_project
    # 3. 类别名称映射
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Yelp": "restaurants", "Toys_and_Games": "toys and games"}
    print(category)
    category = category_dict[category]
    # 4. 检查必须参数
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    # 1. 计算梯度累积步数
    #gradient_accumulation_steps = batch_size // micro_batch_size
    # 2. 分布式训练设置
    device_map = "auto"
    # 获取GPU数量
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 #是否使用分布式数据并行
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        # 从预训练权重加载模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    # 从配置从头开始训练（随机初始化权重）
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
    # 4. 分词器设置
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # 1. 使用SFTData类加载数据
    train_data = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    # 2. 转换为Hugging Face Dataset格式
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    # 1. 禁用模型缓存（训练时需要）
    model.config.use_cache = False
    # 2. 开始训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # 3. 保存模型
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
