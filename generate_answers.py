# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import copy
import json
import os
import random
import openai
import dataclasses
import logging
import tiktoken
from tqdm import tqdm
from typing import Optional, Sequence, Union, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from prompt_templates import ConversationPrompt, ConversationPromptAnswer, ConversationPromptTask, ConversationPromptTask_2
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.1
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='add_generated_instructions_1.json,add_generated_instructions_2.json', help="one or more input files, separated by comma.")
    parser.add_argument("--save_file", type=str,
                        default='add_answers.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    args.save_file = os.path.join(args.path, args.save_file)
    random.seed(args.seed)
    
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))
    
    template = ConversationPromptAnswer()
    decoding_args = OpenAIDecodingArguments()

    # read all the input files
    all_instances = []
    data_files = args.data_files.split(",")
    for data_file in data_files:
        data_file = os.path.join(args.path, data_file)
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                instances = json.load(f)
                all_instances.extend(instances)
        else:
            raise ValueError("Input file {} does not exist.".format(data_file))
    
    # group instances by id
    id2instances = {}  # {"SuperNI-task497-d1bb2eb02d7749849340515edb593540": {"input":..., "instructions":[...]}}
    for instance in all_instances:
        ins_id = instance["id"].rsplit("-", 1)[0]
        if id2instances.get(ins_id,None) is None:
            id2instances[ins_id] = {"input": instance["input"], "instructions": instance["instructions"], "cost": instance["cost"]}
        else:
            id2instances[ins_id]["instructions"].extend(instance["instructions"])
            id2instances[ins_id]["cost"] += instance["cost"]
    
    # annotate the instructions 
    target_datas, skip_num, complete_num = [], 0, 0
    # TODO: only test the first 100 instances
    # id2instances = dict(random.sample(list(id2instances.items())[10:], 40))
    for input_id, input_ins in tqdm(id2instances.items(), total=len(id2instances)):
        input, instructions, all_cost = input_ins["input"], input_ins["instructions"], input_ins["cost"]
        # delete identical instructions to save money
        instructions = list(set(instructions))
        annotated_instances = []
        for idx, instruction in enumerate(instructions):
            input_value = {"input": input, "instruction": instruction}
            content, cost = openai_chat_completion(input_value, template, decoding_args)
            if content is None:
                skip_num += 1
                continue
            input_value["output"] = content
            input_value["cost"] = cost
            annotated_instance = copy.deepcopy(input_value)
            annotated_instance.pop("input")
            annotated_instances.append(annotated_instance)
            complete_num += 1
            all_cost += cost
        target_data = {"id": input_id, "input": input, "instances": annotated_instances, "cost": all_cost}
        target_datas.append(target_data)
            
    # write the output files
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(target_datas, f, indent=2)
    
    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, complete_num))
        
if __name__ == "__main__":
    main()