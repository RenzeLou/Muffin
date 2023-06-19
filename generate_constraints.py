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

from prompt_templates import ConversationPromptConstraint, ConversationPromptConstraint_2
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    

def filter_same_instructions(instructions: List[str]):
    instructions = list(set(instructions))  # Remove exact duplicate elements

    # remove instructions that are substrings of other instructions
    filtered_instructions = []
    for i in range(len(instructions)):
        included = False
        for j in range(len(instructions)):
            if i != j and instructions[i] in instructions[j]:
                included = True
                break
        if not included:
            filtered_instructions.append(instructions[i])
            
    instructions = filtered_instructions
    
    return instructions


def get_shuffled_demons(all_demons:list):
        # shuffle the demons
        random.shuffle(all_demons)
        demons_dict = {}
        for id, demon in enumerate(all_demons):
            demon_instruct, demon_constraint = demon["instruction"], demon["constraints"]
            demons_dict["instruction_{}".format(id+1)] = demon_instruct
            demons_dict["constraint_{}".format(id+1)] = demon_constraint
            if demon.get("input", None) is not None:
                demons_dict["input_{}".format(id+1)] = demon["input"]
        
        return demons_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo", help="the name of the api model.")
    parser.add_argument("--choice", type=int, required=True, help="which prompt template to use: [0, 1]")
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='add_generated_instructions_1.json,add_generated_instructions_2.json', help="one or more input files, separated by comma.")
    parser.add_argument("--save_file", type=str,
                        default='add_constraints.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances (input) to annotate.")
    parser.add_argument("--demo_instructions", type=str, default="/scratch/rml6079/project/Tk-Instruct/data/valid_x/constraints_few_manual.json",
                        help="path to the demonstration instructions w/ constraints.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    args.save_file = os.path.join(args.path, args.save_file)
    random.seed(args.seed)
    
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))
    
    if args.choice == 0:
        # given instruction, generate constraints
        template = ConversationPromptConstraint()
    elif args.choice == 1:
        # given both instruction and input, generate constraints
        template = ConversationPromptConstraint_2()
    else:
        raise ValueError("Choice must be 0 or 1.")
    decoding_args = OpenAIDecodingArguments()
    
    # load the demonstration instructions
    with open(args.demo_instructions, "r") as f:
        all_demons = json.load(f)
        all_demons = all_demons[:template.demonstrations_num]  # 8-shot demonstrations 

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
    skip_num, complete_num = 0, 0
    new_id2instances = {}
    id2instances = dict(random.sample(list(id2instances.items()), min(args.instance_num, len(id2instances)))) if args.instance_num is not None else id2instances
    for input_id, input_ins in tqdm(id2instances.items(), total=len(id2instances)):
        input, instructions, all_cost = input_ins["input"], input_ins["instructions"], input_ins["cost"]
        # delete identical instructions to save money
        instructions = filter_same_instructions(instructions)
        new_instructions, constraints = [], []  # instructions where the constraints are added
        new_input_ins = copy.deepcopy(input_ins)  # all remain the same except for the `instructions`.
        for idx, instruction in enumerate(instructions):
            demons_dict = get_shuffled_demons(all_demons)  # since the response of LLMs depends on the order of the demons, shuffle it each time to avoid bias
            input_value = copy.deepcopy(demons_dict)
            input_value["target_instruction"] = instruction
            input_value["target_input"] = input
            content, cost = openai_chat_completion(input_value, template, decoding_args, model_name=args.api_name)
            if content is None:
                skip_num += 1
                continue
            # new_instruction = f"{instruction} Output constraint: {content}"
            new_instruction = instruction
            new_instructions.append(new_instruction)
            constraints.append(content)
            complete_num += 1
            all_cost += cost
        new_input_ins["instructions"] = new_instructions
        new_input_ins["constraints"] = constraints
        new_input_ins["cost"] = all_cost
        new_id2instances[input_id] = new_input_ins
            
    # write the output files
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(new_id2instances, f, indent=2)
    
    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, complete_num))
        
if __name__ == "__main__":
    main()