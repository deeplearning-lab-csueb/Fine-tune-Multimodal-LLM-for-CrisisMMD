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
        ds = load_dataset('json', data_files={'train': 'processed_dataset/informative_train.json',
                                              'dev': 'processed_dataset/informative_dev.json',
                                              'test': 'processed_dataset/informative_test.json'})
        dataset = ds[split]
        instruction = "You are an data expert in crisis management with many years of experience. \
You are classifying the following tweet containing only text for crisis management. There are two categories to label the tweets: \
'1: informative', '0: not_informative'. \
 if the tweet text provides relevant and useful information that could help crisis management during a crisis, the tweet is informative, \
then reply '1' and only that, no additional words, otherwise, the tweet is not informative, reply '0' and only that, no additional words. Tweet text is: {}, \
the classification is:"
        converted_dataset = [tokenizer.apply_chat_template(convert_to_conversation_text_only(instruction, example), add_generation_prompt = True) for example in dataset]
        return (dataset, converted_dataset)

    elif task == "humanitarian":
        ds = load_dataset('json', data_files={'train': 'processed_dataset/humanitarian_train.json',
                                              'dev': 'processed_dataset/humanitarian_dev.json',
                                              'test': 'processed_dataset/humanitarian_test.json'})
        dataset = ds[split]
        
        instruction = (
            "You are an expert in disaster response and humanitarian aid data analysis. "
            "Examine this text delimited by triple quotes (\"\"\" \"\"\") carefully and classify it into exactly one of these categories (0-4). "
            "Respond with ONLY the number, no other text or explanation.\n\n"
            "Categories:\n"
            "0: HUMAN IMPACT - Must show direct human suffering or hardship:\n"
            "- Deaths, injuries, or missing people\n"
            "- People struggling without basic needs (food, water, shelter)\n"
            "- Displaced or evacuated people\n"
            "- Personal stories of survival or loss\n"
            "- People stranded or waiting for rescue\n"
            
            "1: RESPONSE EFFORTS - Any organized help effort, no matter how small:\n"
            "- Rescue operations and emergency response\n"
            "- Aid collection or distribution activities\n"
            "- Donations of money, supplies, or services\n"
            "- Volunteer work and relief efforts\n"
            "- Medical assistance\n"
            "- Fundraising events for disaster relief\n"
            
            "2: INFRASTRUCTURE DAMAGE - Must describe specific physical destruction:\n"
            "- Destroyed or damaged buildings and homes\n"
            "- Damaged roads, bridges, or transportation systems\n"
            "- Disrupted power lines or water systems\n"
            "- Damaged vehicles or equipment\n"
            "- Before/after comparisons showing destruction\n"
            
            "3: CRISIS UPDATES - Must be specific to the crisis but not fit above categories:\n"
            "- Weather forecasts and disaster warnings\n"
            "- Maps or descriptions of impact areas\n"
            "- Official announcements about the disaster\n"
            "- Statistics and data about crisis impact\n"
            "- Crisis reporting without specific damage/casualties/response\n"
            
            "4: NOT CRISIS-RELATED - Use when no other category clearly fits:\n"
            "- General discussion without crisis specifics\n"
            "- Personal opinions about non-crisis aspects\n"
            "- Promotional or commercial content\n"
            "- Unclear connection to crisis\n"
            "- Content that could apply to non-crisis situations\n\n"
            
            "Important Decision Rules:\n"
            "- If you see ANY mention of help, rescue, or donations → Pick 1\n"
            "- If you see ANY mention of human casualties, suffering, or displacement but not related to volunteer, rescue, donation... → Pick 0\n"    
            "- If you see ANY specific physical destruction of properties → Pick 2\n"
            "- If it's clearly about the crisis but doesn't fit 0-2 → Pick 3\n"
            "- If multiple categories could apply, use the one that best fits to the text meaning.\n"
            "- Only use 4 when you are COMPLETELY SURE no other category fits\n"
            "Answer with just a single digit (0–4).\n\n"
            
            "Tweet text is: {}.\n"
            "the classification is:"
        )   
        converted_dataset = [tokenizer.apply_chat_template(convert_to_conversation_text_only(instruction, example), add_generation_prompt = True) for example in dataset]
        return (dataset, converted_dataset)
        
    else:
        print(f"Invalid task name '{task}'. Must be either 'informative' or 'humanitarian'. Exiting...")
        return (None, None)

def convert_to_conversation_text_only(instruction, example):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": instruction.format(example['text'])}
        ]}
    ]

def start_inference(model, tokenizer, dataset, converted_dataset, model_path, split):
    results = []
    result_path = model_path.strip('/') + f"/results_{split}.json"
    counter, invalid_count = 0, 0
    for input_text, data in zip(converted_dataset, dataset):
        y_true = data['label']
        text = data['text']
        counter += 1
        inputs = tokenizer(
            images=None,
            text=input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")

        output = model.generate(**inputs, max_new_tokens = 128,
                                use_cache = True, do_sample=False)
        y_pred = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        # print(f"input_text: {input_text}")
        # print(f"y_pred: {y_pred}")
        
        if y_pred is None: # ignore the invalid results
            invalid_count += 1
            continue
            
        results.append({
            "text": text,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        # print(f"y_true: {y_true}; y_pred: {y_pred}\n")
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
    # dataset, converted_dataset = get_dataset(tokenizer, task, split)
    
    # if dataset and converted_dataset:
    #     start_inference(model, tokenizer, dataset, converted_dataset, model_path, split)
    
    end_time = time.time()
    print(f"\n{'-'*80}\ncommand: {args}\n{'-'*80}\n")
    print(f"total inference runtime: {(end_time-start_time)/60}mins.")




