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
    stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # suffix: Optional[str] = None
    # logit_bias: Optional[dict] = None
    # echo: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--data_file", type=str,
                        default='./data/dummy/SuperNI.json')
    # parser.add_argument("--save_file", type=str,
    #                     default='./data/dummy/attributes.json')
    parser.add_argument("--seed", type=int, default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    openai.api_key = args.api_key
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
    outputs = []
    for i, instance in tqdm(enumerate(instances), total=len(instances)):
        content, cost = openai_chat_completion(instance, template, decoding_args)
        instance.update({"attributes": content, "cost": cost})
        # print(instance)
        # exit()
        outputs.append(instance)

    # write the output files
    save_file = args.data_file.replace(".json", "_attributes.json")
    with open(save_file, "w") as f:
        json.dump(outputs, f, indent=2)
    # for instance in outputs:
    #     print(instance + '\n')


if __name__ == "__main__":
    main()
