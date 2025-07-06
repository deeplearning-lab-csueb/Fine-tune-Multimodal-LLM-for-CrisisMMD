#!/usr/bin/env python
# coding: utf-8

import json
import time
import argparse
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TextStreamer

def get_model_and_tokenizer(model_path):
    # load saved model for inference
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = False, # Set to False for 16bit LoRA
    )
    print(f"Loading model from {model_path}")
    FastVisionModel.for_inference(model) # Enable for inference!
    return (model, tokenizer)

def get_dataset(tokenizer, task, split):
    if split not in ['train', 'dev', 'test']:
        print(f"Invalid split name '{split}'. Must be either 'test', 'dev' or 'train'. Exiting...")
        return (None, None)
    if task == "informative":
        ds = load_dataset("xiaoxl/crisismmd2inf")
        dataset = ds[split]
        instruction = """Does the image give useful information that could help during a crisis?
Respond with '1' if this image provides any information or details about a crisis, and '0' if it does not.

Instructions: 
  - You should prioritize identifying texts that provide relevant details, even if they are brief or incomplete.
  - Avoid being overly restrictive. If the text has any relevant crisis-related information, response with '1'.
  - When the meaning of the image is unclear, response with '0'.
  - Do not output any extra text.

The classification is:
"""
        converted_dataset = [tokenizer.apply_chat_template(convert_to_conversation(instruction, example), add_generation_prompt = True) for example in dataset]
        return (dataset, converted_dataset)

    elif task == "humanitarian":
        ds = load_dataset("xiaoxl/crisismmd2hum")
        dataset = ds[split]
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
        converted_dataset = [tokenizer.apply_chat_template(convert_to_conversation(instruction, example), add_generation_prompt = True) for example in dataset]
        return (dataset, converted_dataset)
        
    else:
        print(f"Invalid task name '{task}'. Must be either 'informative' or 'humanitarian'. Exiting...")
        return (None, None)

def convert_to_conversation(instruction, example):
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]

def start_inference(model, tokenizer, dataset, converted_dataset, model_path, split):
    results = []
    result_path = model_path.strip('/') + f"/results_{split}.json"
    counter, invalid_count = 0, 0
    for input_text, data in zip(converted_dataset, dataset):
        image = data['image']
        y_true = data['label']
        text = data['tweet_text']
        counter += 1
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        
        # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        # output = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
        #                    use_cache = True, temperature = 1.5, min_p = 0.1)
        # output = model.generate(**inputs, max_new_tokens = 128,
        #                    use_cache = True, temperature = temperature, min_p = 0.1)
        output = model.generate(**inputs, max_new_tokens = 128,
                                use_cache = True, do_sample=False)
        # result = tokenizer.decode(output[0])
        # y_pred = result.split("assistant<|end_header_id|>")[1][:-10].strip("\n")
        y_pred = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        # print(f"counter: y_pred\n{counter}: {y_pred}\n")
        
        if y_pred is None: # ignore the invalid results
            invalid_count += 1
            continue
            
        results.append({
            "text": text,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if counter % 10 == 0:
            print(f"counter: {counter}")
            
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nPerformed {counter} test cases. Invalid count: {invalid_count}. Predictions saved to {result_path}")
    return results

if __name__ == "__main__":
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Inference configuration parameters')
    parser.add_argument('--model_path', type=str, default='./lora_model',
                    help='Directory to load model (default: ./lora_model)')
    # parser.add_argument('--temperature', type=float, default='0.1',
    #                 help='Inference temperature (default: 0.1)')
    parser.add_argument('--task', type=str, default="informative", 
                        help='task: "informative" or "humanitarian", default="informative"')
    parser.add_argument('--split', type=str, default="test", 
                        help='split: "test" or "dev" or "train" or "all", default="test"')
    args = parser.parse_args()
    model_path = args.model_path
    # temperature = args.temperature
    task = args.task
    split = args.split
    print(f"\n{'-'*80}\ncommand: {args}\n{'-'*80}\n")
    
    model, tokenizer = get_model_and_tokenizer(model_path)
    if split == "all":
        for split in ['test', 'dev', 'train']:
            print(f"inferencing on {split} dataset.\n")
            dataset, converted_dataset = get_dataset(tokenizer, task, split)
            start_inference(model, tokenizer, dataset, converted_dataset, model_path, split)
    else:
        dataset, converted_dataset = get_dataset(tokenizer, task, split)
        if dataset and converted_dataset:
            start_inference(model, tokenizer, dataset, converted_dataset, model_path, split)
    
    end_time = time.time()
    print(f"\n{'-'*80}\ncommand: {args}\n{'-'*80}\n")
    print(f"total inference runtime: {(end_time-start_time)/60}mins.")




