
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
# from generate_attributes import OpenAIDecodingArguments

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages.
    See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def construct_prompt(input_dic: dict, template: ConversationPrompt, max_tokens=2048, model="gpt-3.5-turbo"):
    '''
    # cut long completion
    # assert the max length of chatgpt is 4096
    # therefore, 4096 = completion (max_tokens) + messages
    '''
    user_content = template.query_prompt.format_map(input_dic)
    messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": user_content}
        ]
    message_tok_num = num_tokens_from_messages(messages=messages, model=model)
    # the sum of tokens of messages and completion should be less than 4096
    if message_tok_num + max_tokens > 4096:
        max_tokens = max(4096 - message_tok_num - 100, 0) # 100 is a buffer
        logging.warning("since the message is too long ({}), reduce the max_tokens of completion to {}".format(message_tok_num, max_tokens))

    return messages, max_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    '''
    # Retry with exponential backoff
    # See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    '''
    result = openai.ChatCompletion.create(**kwargs)
    
    return result


def openai_chat_completion(
    input_dic: dict,
    template: ConversationPrompt,
    decoding_args,
    model_name="gpt-3.5-turbo",
    **decoding_kwargs,
):
    '''
    for each input x, do single-turn chat completion
    '''
    batch_decoding_args = copy.deepcopy(decoding_args)
    # construct the prompt, and try to reduce max_tokens of completion if the message is too long
    messages, batch_decoding_args.max_tokens = construct_prompt(input_dic, template, max_tokens=batch_decoding_args.max_tokens, model="gpt-3.5-turbo-0301")
    shared_kwargs = dict(
        model=model_name,
        messages=messages,
        **batch_decoding_args.__dict__,
        **decoding_kwargs,
    )
    completion = completion_with_backoff(**shared_kwargs)
    # completion = openai.ChatCompletion.create(**shared_kwargs)
    choices = completion.choices
    reponse = choices[0].message.content
    cost = completion.usage.total_tokens

    # extract the contents from the response
    content = template.extract_content(reponse)

    return content, cost