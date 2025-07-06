#!/usr/bin/env python
# coding: utf-8
# %%

import sys, argparse, time, os
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def get_model_and_tokenizer(model_path, ft_layers, lora_config):
    lora_r, lora_alpha, lora_dropout = lora_config
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    if model_path == "unsloth/Llama-3.2-11B-Vision-Instruct": # start from scratch, not checkpoint
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = ft_layers[0], # False if not finetuning vision layers
            finetune_language_layers   = ft_layers[1], # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers
        
            r = lora_r,           # The larger, the higher the accuracy, but might overfit
            lora_alpha = lora_alpha,  # Recommended alpha == r at least
            lora_dropout = lora_dropout,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )
    return (model, tokenizer)


def get_dataset(task):
    if task == "informative":
        ds = load_dataset("xiaoxl/crisismmd2inf")
        dataset = ds["train"]
#         instruction = "You are an data expert in crisis management with many years of experience. \
# You are classifying the following tweet containing both text and image for crisis management. There are two categories to label the tweets: \
# '1: informative', '0: not_informative'. \
#  if a tweet text and image both provide relevant and useful information that could help crisis management during a crisis, the tweet is informative, \
# then reply 'informative' and only that, no additional words, otherwise, the tweet is not informative, reply 'not_informative' and only that, no additional words. Tweet text is: {}, \
# the classification is:"
        instruction = """Does the image give useful information that could help during a crisis?
Respond with '1' if this image provides any information or details about a crisis, and '0' if it does not.

Instructions: 
  - You should prioritize identifying texts that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text has any relevant crisis-related information, response with '1'.
  - When the meaning of the image is unclear, response with '0'.
  - Do not output any extra text.

The classification is:
"""
        converted_dataset = [convert_to_conversation(instruction, sample) for sample in dataset]
        return (dataset, converted_dataset)
    # --------------------------------------------------------------------------------------------------------------------    
    elif task == "humanitarian":
        ds = load_dataset("xiaoxl/crisismmd2hum")
        dataset = ds["train"]
        label_dict = {'affected_individuals': 0, 
                      'rescue_volunteering_or_donation_effort': 1,
                      'infrastructure_and_utility_damage': 2, 
                      'other_relevant_information': 3,
                      'not_humanitarian': 4}
        def process_dataset(example):
            example['label'] = label_dict[example['label']]
            return example
        dataset = dataset.map(process_dataset)
        
        instruction = (
            "You are an expert in disaster response and humanitarian aid image analysis. "
            "Examine this image carefully and classify it into exactly one of these categories (0-4). "
            "Respond with ONLY the number, no other text or explanation.\n\n"
            "Categories:\n"
            "0: HUMAN IMPACT - Must show PEOPLE who are clearly affected by the disaster: injured, displaced, "
            "evacuated, in temporary shelters, or waiting in lines for aid. People must be visibly impacted.\n"
            
            "1: RESPONSE EFFORTS - Must show active RESCUE operations, aid distribution, medical treatment, "
            "VOLUNTEER work, DONATION, or evacuation in progress. Look for emergency responders, relief workers, or "
            "organized aid activities.\n"
            
            "2: INFRASTRUCTURE DAMAGE - Must show clear physical damage to buildings, roads, bridges, power lines, "
            "VEHICLES or other structures that was caused by the disaster. The damage should be obvious and significant.\n"
            
            "3: OTHER CRISIS INFO - Shows verified crisis-related content that doesn't fit above categories: "
            "maps of affected areas, emergency warnings, official updates, or documentation of crisis conditions. "
            "Must have clear connection to the current disaster.\n"
            
            "4: NOT CRISIS-RELATED - Use this for:\n"
            "- Images where you're unsure if it's related to the crisis\n"
            "- General photos that could be from any time/place\n" 
            "- Images without clear crisis impact or response\n"
            "- Stock photos or promotional images\n"
            "- Any image that doesn't definitively fit categories 0-3\n\n"
            
            "Important:\n"
            "- If there's ANY sign of rescue or donation, pick 1.\n"
            "- If there's ANY sign of damage, pick 2.\n"
            "- If there's ANY sign of obviously distressed or harmed people, pick 0.\n"
            "- If it's definitely about a crisis but you DO NOT see rescue/damage/impacted people, pick 3.\n"
            "- Otherwise, pick 4. Also, when you are not sure which number to pick, pick 4.\n"
            "Answer with just a single digit (0–4).\n"
            "The classification is:"
        )
        converted_dataset = [convert_to_conversation(instruction, sample) for sample in dataset]
        return (dataset, converted_dataset)
    else:
        print(f"Invalid task name '{task}'. Must be either 'informative' or 'humanitarian'. Exiting...")
        return (None, None)

def convert_to_conversation(instruction, sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["label"]} ]
        },
    ]
    return { "messages" : conversation }

def start_training(model, tokenizer, data_collator, converted_dataset, learning_rate, epoch, batch_size, output_dir, gradient_accumulation_steps, logging_steps):
    FastVisionModel.for_training(model) # Enable for training!
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = data_collator, # Must use!
        train_dataset = converted_dataset, #
        args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = 5,
            # max_steps = 30,
            num_train_epochs = epoch, # Set this instead of max_steps for full training runs
            learning_rate = learning_rate, # change this
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none", 
            # report_to = "tensorboard",
        
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )
    
    
    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()

    model.save_pretrained(output_dir) 
    tokenizer.save_pretrained(output_dir)
    
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Training configuration parameters')
    parser.add_argument('--model_path', type=str, default='unsloth/Llama-3.2-11B-Vision-Instruct',
                        help='Directory to load model (default: "unsloth/Llama-3.2-11B-Vision-Instruct")')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--epoch', type=float, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save output files (default: ./output)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Number of updates steps to accumulate before performing a backward/update pass (default: 1)')
    ''' --vision, --language --attention --mlp
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers
    '''
    parser.add_argument('--vision_layers', type=bool, default=False, help='default=False')
    parser.add_argument('--language_layers', type=bool, default=True, help='default=True')
    ''' --lora_r, --lora_alpha, --lora_dropout
        lora: r, lora_alpha, lora_dropout
    '''
    parser.add_argument('--lora_r', type=int, default=16, help='lora_r: default=16')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora_alpha: default=16')
    parser.add_argument('--lora_dropout', type=float, default=0, help='lora_dropout (float): default=0')
    '''
        --task: informative, humanitarian
    '''
    parser.add_argument('--task', type=str, default="informative", help='task: "informative" or "humanitarian", default="informative"')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging_steps: default=100')
    
    args = parser.parse_args()
    model_path = args.model_path
    learning_rate = args.learning_rate
    epoch = args.epoch
    batch_size = args.batch_size
    output_dir = args.output_dir
    gradient_accumulation_steps = args.gradient_accumulation_steps
    ft_layers = [args.vision_layers, args.language_layers]
    lora_config = [args.lora_r, args.lora_alpha, args.lora_dropout]
    task = args.task
    logging_steps = args.logging_steps
    print(f"\n{'-'*80}\ncommand: {args}\n{'-'*80}\n")
    
    dataset, converted_dataset = get_dataset(task)
    if dataset and converted_dataset:
        model, tokenizer = get_model_and_tokenizer(model_path, ft_layers, lora_config)
        data_collator = UnslothVisionDataCollator(model, tokenizer)
        start_training(model, tokenizer, data_collator, converted_dataset, learning_rate, epoch, batch_size, output_dir, gradient_accumulation_steps, logging_steps)
    end_time = time.time()
    print(f"\n{'-'*80}\ncommand: {args}\n{'-'*80}\n")
    print(f"Total training time: {(end_time-start_time)/60:.2f} mins.")
