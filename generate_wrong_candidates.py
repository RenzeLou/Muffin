# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import copy
import json
import os
import random
import string
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

from prompt_templates import ConversationPrompt, ConversationPromptWrongOutputs
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 3096
    temperature: float = 0.3
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo", help="the name of the api model.")
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='filtered_full.json', help="one or more input files, separated by comma.")
    parser.add_argument("--save_file", type=str,
                        default='filtered_full_with_wrong_candidates.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances (input) to annotate.")
    parser.add_argument("--length_threshold", type=int, default=None, help="whether to set a length threshold that doesn't generate wrong candidates when the correct output is too long.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    args.save_file = os.path.join(args.path, args.save_file)
    random.seed(args.seed)
    
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))
    
    template = ConversationPromptWrongOutputs()
    decoding_args = OpenAIDecodingArguments()

    # read the source files
    with open(os.path.join(args.path, args.data_files), "r") as f:
        source_datas = json.load(f)
    
    source_datas = random.sample(source_datas, min(args.instance_num, len(source_datas))) if args.instance_num is not None else source_datas 
    
    # process the source files
    target_datas, skip_num, complete_num = [], 0, 0
    for source_data in tqdm(source_datas, total=len(source_datas)):
        instances = source_data["instances"]
        id, input, all_cost = source_data["id"], source_data["input"], source_data["cost"]
        new_instances = []
        for idx, ins in enumerate(instances):
            # instruction, output, cost = ins["instruction"], ins["output"], ins["cost"]
            if args.length_threshold is not None and len(ins["output"].split()) > args.length_threshold:
                # if this task requires long generations, we don't anticipate it can be constructed into classifcation tasks
                ins["wrong_outputs"] = []
                cost = 0
                skip_num += 1
            else:
                input_value = copy.deepcopy(ins)
                input_value["input"] = input
                content, cost = openai_chat_completion(input_value, template, decoding_args, model_name=args.api_name)
                if content is None:
                    skip_num += 1
                    continue
                ins["wrong_outputs"] = content
                ins["cost"] += cost
                complete_num += 1
            new_instances.append(ins)
            all_cost += cost
            
        source_data["instances"] = new_instances
        source_data["cost"] = all_cost
        target_datas.append(source_data)
            
    # write the output files
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(target_datas, f, indent=2)
    
    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, complete_num))
        
if __name__ == "__main__":
    main()