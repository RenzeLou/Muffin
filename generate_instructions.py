# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import copy
import json
import os
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

from prompt_templates import ConversationPromptTask,ConversationPrompt
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 3800
    temperature: float = 0.2
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 1.99
    frequency_penalty: float = 0.0
    # suffix: Optional[str] = None
    # logit_bias: Optional[dict] = None
    # echo: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--data_file", type=str,
                        default='./data/dummy/SuperNI_attributes.json')
    # parser.add_argument("--save_file", type=str,
    #                     default='./data/dummy/attributes.json')
    parser.add_argument("--seed", type=int, default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    openai.api_key = args.api_key
    template = ConversationPromptTask()
    decoding_args = OpenAIDecodingArguments()

    # read the input files
    if os.path.exists(args.data_file):
        with open(args.data_file, "r") as f:
            instances = json.load(f)
    else:
        raise ValueError("Input file {} does not exist.".format(args.data_file))
    
    all_instances = []
    for ins in instances:
        id, x, atts, cost = ins["id"], ins["input"], ins["content"], ins["cost"]  # ins["attributes"]  
        for idx, att in enumerate(atts):
            all_instances.append({"id": id + f"-hint{idx}", "input": x, "hint": att, "cost": cost})
            
    outputs = []
    for i, instance in tqdm(enumerate(all_instances), total=len(all_instances)):
        content, cost = openai_chat_completion(instance, template, decoding_args)
        instance.update({"instructions": content})
        instance["cost"] += cost
        # print(instance)
        # exit()
        outputs.append(instance)

    # write the output files
    save_file = args.data_file.replace("_attributes.json", "_instructions.json")
    with open(save_file, "w") as f:
        json.dump(outputs, f, indent=2)
        
if __name__ == "__main__":
    main()