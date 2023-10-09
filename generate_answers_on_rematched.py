# used for annotating the answers on the rematched data
# the only difference between this file and `generate_answers.py` is that, this file support the input file that contains outputs (official matched pairs)
import argparse
import copy
import json
import os
import random
import string
import openai
import dataclasses
import logging
import tenacity
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
# from post_process.compute_metrics import rougeL_score

import sys
sys.path.append("./post_process")
from compute_metrics import rougeL_score # noqa


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.1
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    

def filter_same_instructions(instructions: List[str]):
    ori_len = len(instructions)
    
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
    
    final_len = len(instructions)
    deleted_num = ori_len - final_len
    
    return instructions, deleted_num

def filter_highly_overlapping_instructions(instructions: List[str], threshold=0.6):
    ori_len = len(instructions)
    
    # remove instructions that are highly overlapping with other instructions
    filtered_instructions = []
    for instruction in instructions:
        included = True
        for filtered_instruction in filtered_instructions:
            if rougeL_score(instruction, filtered_instruction) > threshold:
                included = False
                break
        if included:
            filtered_instructions.append(instruction)
        
    instructions = filtered_instructions
    
    final_len = len(instructions)
    deleted_num = ori_len - final_len
    
    return instructions, deleted_num


def process_input_files(args):
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
    
    # group instances by id (gather the same input)
    id2instances = {}  
    for instance in all_instances:
        ins_id = instance["id"].rsplit("-", 1)[0] if not args.no_hint_id else instance["id"]
        if id2instances.get(ins_id,None) is None:
            output = instance.get("output", None)
            id2instances[ins_id] = {"input": instance["input"], "instructions": instance["instructions"], "cost": instance["cost"], "outputs": [output]}
        else:
            id2instances[ins_id]["instructions"].extend(instance["instructions"])
            id2instances[ins_id]["cost"] += instance["cost"]
            id2instances[ins_id]["outputs"].append(instance.get("output", None))
            
    return id2instances

def save_intermediate_results(all_items, args, message):
    file_name = os.path.basename(args.save_file)
    file_name = file_name.rsplit(".", 1)[0] + f".{message}.json"
    terminate_save_path = os.path.join(args.path, "terminated_results")
    os.makedirs(terminate_save_path, exist_ok=True)
    with open(os.path.join(terminate_save_path, file_name), "w") as f:
        json.dump(all_items, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo", help="the name of the api model.")
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='add_generated_instructions_1.json,add_generated_instructions_2.json', help="one or more input files, separated by comma.")
    parser.add_argument("--save_file", type=str,
                        default='add_answers.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances (input) to annotate.")
    parser.add_argument("--constraint_added", action="store_true", help="whether the constraints have been added to the input file.")
    parser.add_argument("--rouge_threshold", type=float, default=0.6, help="the rouge threshold for filtering highly overlapping instructions.")
    parser.add_argument("--cancel_filter", action="store_true", help="do filtering by default to save money (we dont want to annotate the same instruction twice); but sometimes we have to cancel the filtering")
    parser.add_argument("--no_hint_id", action="store_true", help="set to true if the id looks like 'SuperNI-task291-ab63f213cfd240ce80a71eee0a329a83', no '-hint1' suffix.")

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

    id2instances = process_input_files(args) # {"SuperNI-task497-d1bb2eb02d7749849340515edb593540": {"input":..., "instructions":[...], "cost": ...}}
    
    # annotate the instructions 
    target_datas, skip_num, complete_num = [], 0, 0
    identical_del_num = []
    id2instances = dict(random.sample(list(id2instances.items()), min(args.instance_num, len(id2instances)))) if args.instance_num is not None else id2instances
    
    try:
        for input_id, input_ins in tqdm(id2instances.items(), total=len(id2instances)):
            input, instructions, all_cost = input_ins["input"], input_ins["instructions"], input_ins["cost"]
            outputs = input_ins["outputs"]
            assert len(instructions) == len(outputs), "the length of instructions and outputs should be the same."
            constraints = input_ins.get("constraints", None)
            # delete identical instructions to save money
            if not args.constraint_added and not args.cancel_filter:
                # if the constraints have been added, there is no need to filter (already doen in generate_constraints.py)
                instructions, del_num_1 = filter_same_instructions(instructions)
                instructions, del_num_2 = filter_highly_overlapping_instructions(instructions, args.rouge_threshold)
                del_num = del_num_1 + del_num_2
                identical_del_num.append(del_num)
            annotated_instances = []
            for instruction, output in zip(instructions, outputs):
                # constraint = constraints[idx] if constraints is not None else None
                constraint = None
                query_instruction = f"{instruction} Output constraint: {constraint}" if constraint is not None else instruction
                input_value = {"input": input, "instruction": query_instruction}
                if output is None:
                    content, cost = openai_chat_completion(input_value, template, decoding_args, model_name=args.api_name)
                else:
                    content, cost = output, 0
                if content is None:
                    skip_num += 1
                    continue
                input_value["output"] = content
                input_value["cost"] = cost
                annotated_instance = copy.deepcopy(input_value)
                annotated_instance.pop("input")
                annotated_instance["instruction"] = instruction
                annotated_instance["constraint"] = constraint if constraint is not None else ""
                annotated_instances.append(annotated_instance)
                complete_num += 1
                all_cost += cost
            target_data = {"id": input_id, "input": input, "instances": annotated_instances, "cost": all_cost}
            target_datas.append(target_data)
    except KeyboardInterrupt as e:
        # save the intermediate results
        print("==> Error: {}".format(e))
        print("\nUser terminated the program\n")
        save_intermediate_results(target_datas, args, "KeyboardInterrupt")
        sys.exit(0)  # Exit the program gracefully
    # except openai.error.RateLimitError as e:
    except tenacity.RetryError as e:
        print("==> Error: {}".format(e))
        print("\nOpenAI API rate limit reached. Please increase the waiting/retry times in the tenacity decorator.\n")
        save_intermediate_results(target_datas, args, "RateLimitError")
        sys.exit(0)  # Exit the program gracefully    
        
            
    # write the output files
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(target_datas, f, indent=2)
    
    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, complete_num))
    if len(identical_del_num) > 0:
        print("==> identical instructions deleted num: {}, avg del for each input: {}".format(sum(identical_del_num), sum(identical_del_num)/len(identical_del_num)))
    # save above screen print to a file
    file_name = args.save_file.split("/")[-1].split(".")[0]
    screen_save_path = os.path.join(args.path, "screen_print")
    os.makedirs(screen_save_path, exist_ok=True)
    with open(os.path.join(screen_save_path, file_name + ".txt"), "w") as f:
        f.write("==> saved to {}\n".format(save_file))
        f.write("==> skip: {} ; complete: {}\n".format(skip_num, complete_num))
        if len(identical_del_num) > 0:
            f.write("==> identical instructions deleted num: {}, avg del for each input: {}".format(sum(identical_del_num), sum(identical_del_num)/len(identical_del_num)))
        
        
if __name__ == "__main__":
    main()