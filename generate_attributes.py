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

from prompt_templates import ConversationPromptAttribute,ConversationPrompt
from chat_completion import openai_chat_completion


def default_stop() -> List[str]:
    return ["None.", "None", "none.", "none"]

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
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
    parser.add_argument("--data_file", type=str,
                        default='SuperNI.json')
    parser.add_argument("--save_file", type=str,
                        default='attributes.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances (input) to annotate.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    args.data_file = os.path.join(args.path, args.data_file)
    args.save_file = os.path.join(args.path, args.save_file)
    random.seed(args.seed)
    
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))

    template = ConversationPromptAttribute()
    decoding_args = OpenAIDecodingArguments()

    # read the input files
    if os.path.exists(args.data_file):
        with open(args.data_file, "r") as f:
            instances = json.load(f)
    else:
        raise ValueError("Input file {} does not exist.".format(args.data_file))

    # test_instances = [
    #     {"id": "1", "input": "This is what I read when it comes from an EWEB commish, 'Shut up and take it!'"},
    #     {"id": "2", "input": "It's a very nice kit, it came with all the accessories, BUT my waterproof case was broken, the thing that closes it was broken so I can't close the case and now the case is useless. And I bought this kit just because of the waterproof case.... The rest was fine as announced."},
    #     {"id": "3", "input": "['U', '6923', 'y', 'm', 'v', 'M', 'Y', '87667', 'E', '6059', 'p']"}]
    outputs, skip_num = [], 0
    # randomly sample subset of instances (when testing)
    instances = random.sample(instances, min(args.instance_num, len(instances))) if args.instance_num is not None else instances
    for i, instance in tqdm(enumerate(instances), total=len(instances)):
        content, cost = openai_chat_completion(instance, template, decoding_args, model_name=args.api_name)
        if content is None:
            skip_num += 1
            continue
        instance.update({"attributes": content, "cost": cost})
        outputs.append(instance)

    # write the output files
    # save_file = args.data_file.replace(".json", "_attributes.json")
    save_file = args.save_file
    with open(save_file, "w") as f:
        json.dump(outputs, f, indent=2)

    print("==> saved to {}".format(save_file))
    print("==> skip: {} ; complete: {}".format(skip_num, len(outputs)))


if __name__ == "__main__":
    main()
