import torch.nn as nn
from dataprocess import create_dataloader
from tqdm import tqdm
import logging
import torch
import swanlab  
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from utils import *
from torch.amp import autocast, GradScaler
from metric import F1Calculator,F1Calculator_for_code
from collections import Counter
# 设置当前进程可见的 GPU（必须在所有 CUDA 操作之前调用）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 1 号 GPU
# 注释掉swanlab初始化
swanlab.init(
    workspace="",  # 你的工作空间名
    project="",  # 项目名称
    name="",  # 运行名称
)

# 配置日志，输出到log.txt文件中
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log_add/"), logging.StreamHandler()])
logger = logging.getLogger(__name__)
scaler = GradScaler()
def train(model, train_loader, optimizer, epoch, device, scheduler):
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch", ncols=100)

    for batch_idx, (batch_input, batch_mask, batch_labels) in enumerate(pbar):
        batch_input = batch_input.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播使用 AMP
        with autocast(dtype=torch.bfloat16,device_type="cuda"):
            outputs = model(
                batch_input, 
                attention_mask=batch_mask, 
                labels=batch_labels, 
                return_loss=True
            )
            loss = outputs.loss
        
        # AMP 梯度缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()##进行梯度更新，使用梯度检查点来进行更新，这样可以节省不少的显存
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        total_loss += loss.item()
        num_batches += 1

        global_step = epoch * len(train_loader) + batch_idx
        swanlab.log({"learning_rate": current_lr}, step=global_step)
        pbar.set_description(f'epoch: {epoch }')
        pbar.set_postfix({'loss': f'{loss.item():.5f}'})
    
    avg_loss = total_loss / num_batches
    print(f"train epoch:{epoch}\tloss:{avg_loss:.5f}")
    swanlab.log({"train_epoch_loss": avg_loss}, step=epoch)

    return avg_loss
def dev(hypernum, model, dev_loss_loader, dev_f1_loader, epoch, device, tokenizer, output_file, dataset_name, answer_output):
    model.eval()
    total_loss = 0.0
    num_loss_batches = 0
    f1_calculator = F1Calculator_for_code(save_file=answer_output)

    # 第一阶段：计算loss
    if dev_loss_loader is not None:
        pbar_loss = tqdm(dev_loss_loader, desc=f"Calculating Loss-{dataset_name}", unit="batch", ncols=115)
        with torch.no_grad():
            for batch_idx, (batch_input, batch_input_mask, gold_labels) in enumerate(pbar_loss):
                batch_input = batch_input.to(device)
                batch_input_mask = batch_input_mask.to(device)
                gold_labels = gold_labels.to(device)

                # 使用 AMP 前向传播
                with autocast(dtype=torch.bfloat16,device_type="cuda"):
                    outputs = model(
                        batch_input,
                        attention_mask=batch_input_mask,
                        labels=gold_labels,
                        return_loss=True
                    )
                    loss = outputs.loss.mean()

                total_loss += loss.item()
                num_loss_batches += 1

                pbar_loss.set_postfix({
                    'Loss': f'{loss.item():.5f}',
                    'Avg Loss': f'{total_loss / num_loss_batches:.5f}'
                })

    # 第二阶段：计算F1
    if dev_f1_loader is not None:
        pbar_f1 = tqdm(dev_f1_loader, desc=f"Calculating F1-{dataset_name}", unit="batch", ncols=150)
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n\n========== Epoch {epoch} Evaluation Outputs ==========\n")
            with torch.no_grad():
                for batch_idx, (batch_input, batch_input_mask, gold_labels) in enumerate(pbar_f1):
                    batch_input = batch_input.to(device)
                    batch_input_mask = batch_input_mask.to(device)
                    batch_labels = gold_labels

                    # 生成阶段使用 AMP
                    with autocast(dtype=torch.bfloat16,device_type="cuda"):
                        output = model.generate(
                            batch_input,
                            attention_mask=batch_input_mask,
                            max_new_tokens=hypernum.max_new_tokens,
                            temperature=0.5,
                            num_beams=hypernum.num_beams,
                            do_sample=hypernum.do_sample,
                            top_p=hypernum.top_p,
                            top_k=hypernum.top_k,
                            early_stopping=hypernum.early_stopping,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id  
                        )

                    bsz, sql = batch_input.shape
                    decoded_output = tokenizer.batch_decode(output[:, sql:], skip_special_tokens=True)

                    f_out.write(f"\n--- Batch {batch_idx} ---\n")
                    f_out.write("Predictions:\n")
                    for i, pred in enumerate(decoded_output):
                        f_out.write(f"  Sample {i}: {pred}\n")
                    f_out.write("\nTrue Labels:\n")
                    for i, label in enumerate(batch_labels):
                        f_out.write(f"  Sample {i}: {label}\n")

                    f1_calculator.update(decoded_output, batch_labels)
                    tp, fp, fn, f1 = f1_calculator.compute_f1()

                    pbar_f1.set_postfix({
                        'F1': f'{f1:.5f}',
                        'TP': tp, 'FP': fp, 'FN': fn
                    })

    avg_loss = total_loss / num_loss_batches if num_loss_batches > 0 else 0

    logger.info(
        f"dev (Epoch {epoch}, Dataset: {dataset_name}): "
        f"F1={f1:.4f}, Loss={avg_loss:.4f}, TP={tp}, FP={fp}, FN={fn}"
    )

    swanlab.log({
        f"{dataset_name}/dev_epoch_f1": f1,
        f"{dataset_name}/dev_epoch_loss": avg_loss
    }, step=epoch)

    return f1, avg_loss


def test(hypernum, model, test_loaders, epoch, device, tokenizer, output_file, dataset_names, answer_output):
    model.eval()  # 设置模型为评估模式，禁用dropout等训练时的操作
    f1_calculator = F1Calculator_for_code(save_file=answer_output)  # 初始化F1计算器，用于计算F1分数
    pbar = tqdm(zip(*test_loaders.values()), desc=f"Testing-{', '.join(dataset_names)}", unit="batch", ncols=115)  # 设置进度条

    # 打开文件进行结果输出，文件追加模式
    with open(output_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n\n========== Epoch {epoch} Evaluation Outputs ==========\n")

        # 初始化一个字典来存储每个数据集的预测结果
        all_predictions = {dataset_name: [] for dataset_name in dataset_names}
        all_labels = []  # 用于存储所有样本的真实标签

        with torch.no_grad():  # 禁用梯度计算
            # 遍历每个批次
            for batch_idx, batch_alls in enumerate(pbar):
                # batch_inputs 为每个数据集对应的一个批次
                # 每个数据集的输入数据是独立的，需要分别处理
                batch_predictions = []  # 当前批次在所有数据集的预测结果
                gold_labels = None
                # 遍历每个数据集的DataLoader并独立推理
                for dataset_idx,(dataset_name, batch_all) in enumerate(zip(dataset_names, batch_alls)):
                    batch_input = batch_all[0].to(device)  # 将输入数据传到GPU上
                    batch_input_mask = batch_all[1].to(device)  # 获取当前数据集的attention mask
                    batch_labels = batch_all[2]  # 获取当前数据集的标签（可能在不同数据集上有差异）

                    # 使用自动混合精度（AMP）进行前向传播，降低计算开销
                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        output = model.generate(
                            batch_input,  # 输入文本
                            attention_mask=batch_input_mask,  # 输入的mask
                            max_new_tokens=hypernum.max_new_tokens,
                            temperature=0.5,
                            num_beams=hypernum.num_beams,
                            do_sample=hypernum.do_sample,
                            top_p=hypernum.top_p,
                            top_k=hypernum.top_k,
                            early_stopping=hypernum.early_stopping,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    bsz, sql = batch_input.shape
                    # 假设输出为(batch_size, seq_length)的logits，进行解码
                    decoded_output = tokenizer.batch_decode(output[:, sql:], skip_special_tokens=True)

                    # 将当前数据集的预测结果添加到对应的字典中
                    all_predictions[dataset_name].append(decoded_output)
                    batch_predictions.append(decoded_output)

                    ##存储最后一种类型的真实标签，方便最后计算F1
                    if dataset_idx == 2:
                        gold_labels = batch_labels

                # 投票集成：对每个样本进行投票（从不同数据集的预测结果中选择最常见的标签）
                final_preds = []
                for i in range(len(batch_labels)):  # 遍历每个样本
                    # 对当前样本，收集所有数据集的预测，按照batch中的每条数据为最小单位进行处理
                    unique_votes = list(set([batch_predictions[j][i] for j in range(len(dataset_names))]))

                    ##在这里对所有的votes的格式进行统一操作
                    ##待做！！！
                    unique_format = process_and_convert_entity_list(unique_votes)
                    # 执行多数投票：选择出现次数最多的预测结果
                    majority_vote = apply_majority_voting_rule(unique_format)
                    final_preds.append(majority_vote)

                # 更新F1计算器，计算当前批次的F1分数
                f1_calculator.update(final_preds, gold_labels)

                # 计算当前批次的TP, FP, FN和F1分数
                tp, fp, fn, f1 = f1_calculator.compute_f1()

                # 更新进度条
                pbar.set_postfix({
                    'F1': f'{f1:.5f}',
                    'TP': tp, 'FP': fp, 'FN': fn
                })

                # 写入每个批次的预测结果和真实标签到文件
                f_out.write(f"\n--- Batch {batch_idx} ---\n")
                f_out.write("Predictions:\n")
                for i, pred in enumerate(final_preds):
                    f_out.write(f"  Sample {i}: {pred}\n")
                f_out.write("\nTrue Labels:\n")
                for i, label in enumerate(batch_labels):
                    f_out.write(f"  Sample {i}: {label}\n")

        # 记录并打印最终的F1得分
        logger.info(f"Test F1 (Epoch {epoch}, Datasets: {', '.join(dataset_names)}): {f1:.4f}")
        swanlab.log({f"{', '.join(dataset_names)}/test_epoch_f1": f1}, step=epoch)

    return f1

import re

def process_and_convert_entity_list(entity_list):
    converted = []

    for entity_str in entity_list:
        # 去除首尾空白
        entity_str = entity_str.strip()

        # 先按 ";" 分割成多个可能的实体片段
        parts = entity_str.split(";")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # ----------- 情况 1：there is no XXX entity -----------
            no_entity_match = re.search(
                r"there is no\s+(\w+)\s+entity",
                part,
                flags=re.IGNORECASE
            )

            if no_entity_match:
                entity_type = no_entity_match.group(1)
                converted.append(f'EntityList.add("there is no {entity_type} entity");')
                continue

            # ----------- 情况 2：匹配 append/new XXX("name") ----------
            match = re.search(
                r'(append|push_back|add)\s*\(\s*(?:new\s+)?(\w+)\("([^"]+)"\)',
                part,
                flags=re.IGNORECASE
            )

            if match:
                entity_type = match.group(2)
                entity_name = match.group(3)
                converted.append(f'EntityList.add(new {entity_type}("{entity_name}"));')

    return converted



def apply_majority_voting_rule(predictions):##投票策略1：将出现大于等于2次的实体算作最后的实体(降低FP)
    vote_count = Counter(predictions)

    # 找到所有出现次数 ≥ 2 的实体
    majority_entities = [entity for entity, count in vote_count.items() if count >= 2]

    # 如果没有符合条件的，返回空列表（不再返回字符串）
    return majority_entities
def merge_voting(lines):##所有的内容都要
    seen = set()
    result = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result

def test(hypernum, model, test_loader, epoch, device, tokenizer, output_file, dataset_name, answer_output):
    model.eval()
    f1_calculator = F1Calculator_for_code(save_file=answer_output)
    pbar = tqdm(test_loader, desc=f"Testing-{dataset_name}", unit="batch", ncols=115)

    # 打开文件保存输出
    with open(output_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n\n========== Epoch {epoch} Evaluation Outputs ==========\n")
        
        # f1_func_name = f"compute_f1_{dataset_name}_1"
        # f1_func = getattr(bio_metric, f1_func_name)

        with torch.no_grad():
            for batch_idx, (batch_input, batch_input_mask, gold_labels) in enumerate(pbar):
                batch_input = batch_input.to(device)
                batch_input_mask = batch_input_mask.to(device)
                batch_labels = gold_labels

                # AMP 前向生成
                with autocast(dtype=torch.bfloat16,device_type="cuda"):
                    output = model.generate(
                        batch_input,
                        attention_mask=batch_input_mask,
                        max_new_tokens=hypernum.max_new_tokens,
                        temperature=0.5,
                        num_beams=hypernum.num_beams,
                        do_sample=hypernum.do_sample,
                        top_p=hypernum.top_p,
                        top_k=hypernum.top_k,
                        early_stopping=hypernum.early_stopping,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id  
                    )

                bsz, sql = batch_input.shape
                decoded_output = tokenizer.batch_decode(output[:, sql:], skip_special_tokens=True)

                # 写入文件
                f_out.write(f"\n--- Batch {batch_idx} ---\n")
                f_out.write("Predictions:\n")
                for i, pred in enumerate(decoded_output):
                    f_out.write(f"  Sample {i}: {pred}\n")
                f_out.write("\nTrue Labels:\n")
                for i, label in enumerate(batch_labels):
                    f_out.write(f"  Sample {i}: {label}\n")

                f1_calculator.update(decoded_output, batch_labels)
                tp, fp, fn, f1 = f1_calculator.compute_f1()

                pbar.set_postfix({
                        'F1': f'{f1:.5f}',
                        'TP': tp, 'FP': fp, 'FN': fn
                    })

    logger.info(f"Test F1 (Epoch {epoch }, Dataset: {dataset_name}): {f1:.4f}")
    swanlab.log({f"{dataset_name}/test_epoch_f1": f1}, step=epoch)

    return f1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernum = Hypernum.from_yaml("./config/config.yaml")
    ##修改全局的tokenizer
    tokenizer_new = AutoTokenizer.from_pretrained(hypernum.model_path)
    # NEW_ENTITY_TOKENS = ["<bc5cdr_disease>"]
    # tokenizer_new.add_tokens(NEW_ENTITY_TOKENS)
    train_loader, dev_loss_loaders, dev_f1_loaders , test_loaders = create_dataloader(hypernum,tokenizer=tokenizer_new)

    #加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        hypernum.model_path, 
        torch_dtype=torch.bfloat16, 
        attention_dropout=0.05
    ).to(device)
    model.config.use_cache = False

    # 配置LoRA参数
    peft_config = LoraConfig(
        inference_mode=False,
        r=hypernum.r,
        lora_alpha=hypernum.lora_alpha,
        lora_dropout=hypernum.lora_dropout,
        target_modules=hypernum.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    

    if hypernum.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # ★★ 关键：给输入打上需要梯度的钩子，避免梯度断流
        model.enable_input_require_grads()
   

    # 将模型转换为PeftModel以支持LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数数量

    epochs = hypernum.epochs
    # 只优化可训练参数（LoRA参数）
    #optimizer = torch.optim.Adam(model.parameters(), lr=hypernum.lr)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hypernum.lr
    )
    len_train_loader = len(train_loader)
    num_training_steps = len_train_loader * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 初始化全局最佳F1和模型权重
    best_avg_f1 = -1
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch, device, scheduler)
        

        logger.info(f"\nEpoch {epoch }/{epochs} 验证阶段开始:")

        total_f1 = 0
        dataset_count = 0
        total_loss = 0

        for dataset_name in dev_loss_loaders.keys():  # 使用dev_loss_loaders的key
            output_file = hypernum.dev_pre_path.format(dataset=dataset_name)
            answer_output = hypernum.dev_clean_path.format(dataset=dataset_name)
            
            # 获取对应的dataloader
            dev_loss_loader = dev_loss_loaders[dataset_name]
            dev_f1_loader = dev_f1_loaders[dataset_name]
            
            
            val_f1, val_loss = dev(
                hypernum,
                model,
                dev_loss_loader,  # 传入dev_loss_loader
                dev_f1_loader,    # 传入dev_f1_loader
                epoch,
                device,
                tokenizer_new,  # 使用全局的tokenizer，而不是dataset的tokenizer
                output_file,
                dataset_name,
                answer_output
            )

            logger.info(f"[验证集 {dataset_name}] F1: {val_f1:.4f}, Loss: {val_loss:.4f}")
            total_f1 += val_f1
            total_loss += val_loss
            dataset_count += 1

        avg_f1 = total_f1 / dataset_count
        avg_loss = total_loss / dataset_count
        logger.info(f"平均验证结果 => F1: {avg_f1:.4f} | Loss: {avg_loss:.4f}")

        # swanlab.log({
        #     "avg_f1": avg_f1,
        #     "avg_loss": avg_loss
        # }, step=epoch)

        # 如果当前平均F1更优，保存模型
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}##将模型权重先保存在CPU，来节省显存的消耗
            logger.info(f"新最佳模型保存（平均验证F1提升至 {best_avg_f1:.4f}）")

        logger.info(f"Epoch {epoch } 完成, 训练 Loss: {train_loss:.4f}")

    # 保存最佳模型
    if best_model_state is not None:
        save_path = hypernum.save_path
        torch.save(best_model_state, save_path)
        logger.info(f"最优模型（基于所有验证集的平均 F1）已保存到 {save_path}")
        

    # 测试阶段：加载统一的最优模型并测试所有测试集
    logger.info("开始测试阶段...")

    # 加载最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("未找到最佳模型状态，使用当前模型进行测试")

    dataset_names = list(test_loaders.keys())  # 获取所有数据集的名称
    # 每 3 个数据集为一组进行 test
    for i in range(0, len(dataset_names), 3):

        # 取当前组（如 [1,2,3]）
        datasets_to_vote = dataset_names[i:i + 3]
        test_loaders_to_vote = {name: test_loaders[name] for name in datasets_to_vote}

        logger.info(f"当前推理组（用于投票）: {datasets_to_vote}")

        # ★ 当前要真正生成输出的主数据集，是这个组的第一个数据集
        main_dataset = datasets_to_vote[0]

        output_file = hypernum.test_pre_path.format(dataset=main_dataset)
        answer_output = hypernum.test_clean_path.format(dataset=main_dataset)

        # ★ test() 会使用 test_loaders_to_vote 中的所有数据集进行投票推理
        #   但最终结果只输出 main_dataset 对应的答案
        test_f1 = test(
            hypernum,
            model,
            test_loaders_to_vote,   # 三个数据集一起推理
            -1,
            device,
            tokenizer_new,
            output_file,
            datasets_to_vote,       # 三个数据集名称
            answer_output
        )

        logger.info(f"[测试集 {main_dataset}] 测试 F1: {test_f1:.4f}")

    logger.info("全部测试完成！")
