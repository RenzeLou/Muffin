'''
Use this script to expand more classification task
Simply convert each original task into a classification paradigm by mixing correct and wrong outputs
Also use different symbols to represent the options
'''

import argparse
import copy
import json
import math
import os
import random
import re
import string

from tqdm import tqdm

def convert_integer_to_char(num):
    # convert a integer to a char, like 0 -> A, 1 -> B, 25 -> Z, 26 -> AA, 27 -> AB, 51 -> AZ, 52 -> BA, ...
    result = ""
    while num >= 0:
        remainder = num % 26
        result = chr(remainder + 65) + result
        num = (num // 26) - 1
        if num < 0:
            break
    return result

def convert_char_to_integer(char):
    num = 0
    for c in char:
        num = num * 26 + ord(c) - 64
    return num - 1


# INTEGER = [str(i) for i in range(10)]
# ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] + \
#             ["K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"] + \
#             ["U", "V", "W", "X", "Y", "Z"]
# using much more candidates symbols is verified to be slightly better
INTEGER = [str(i) for i in range(18278)] # from '0' to '18277'
ALPHABET = [convert_integer_to_char(i) for i in range(18288)] # from 'A' to 'ZZZ'
MARK = ["@", "#", "$", "%", "^", "&", "*", "+", "!", "?"]
OPTION_SYMBOLS = [ALPHABET, INTEGER, MARK]

CONNECT_SYMBOLS = [":"]  # ["-", "_", ":", "=", "~", ".", "|"] TODO: using less connect symbols is verified to be slightly better
BRACKET_SYMBOLS = [("(", ")"),("'", "'")]  # [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("'", "'"), ("", "")] TODO: using less bracket symbols is verified to be slightly better

CLS_CONSTRANTS_LANGUAGE = ["Your answer should be a single letter from ",
                       "The options are ",
                       "Output constraints: ",
                       "The answer should be one of ",
                       "Avoid answers outside of ",
                       "The answer should be a character from ",
                       "Choose one of ",
                       "Here are the options: ",
                       "Get your answer from ",
                       "Do not generate options except ",
                       "Never use labels other than ",
                       "Try to answer with ",
                       "You should only use these symbols to represent your final answer: ",
                       "Your answer must be a single letter chosen from ",
                        "Choose one of the following options: ",
                        "Please select an option from the following: ",
                        "Make your selection from the options: ",
                        "Select one of the following: ",
                        "Pick a letter from the options: ",
                        "Your response should be one of the following: ",
                        "Choose a character from the following: ",
                        "Please provide a single letter from the options: ",
                        "Pick one of the following choices: ",
                        "Select an answer from the given options: ",
                        "Your answer should match one of the following: ",
                        "Please use only the characters listed here: "
                    ]
GEN_CONSTRANTS_LANGUAGE = ["Output constraints: ", "Output requirements: ", "Output restrictions: ", "Output limitations: ", "Output demands: ", "Output prerequisites: ",
                           "Answer constraints: ", "Answer requirements: ", "Answer restrictions: ", "Answer limitations: ", "Answer demands: ", "Answer prerequisites: ",
                           "Constraints: ", "Requirements: ", "Restrictions: ", "Limitations: ", "Demands: ", "Prerequisites: ",
                           "The constraints are: ", "The requirements are: ", "The restrictions are: ", "The limitations are: ", "The demands are: ", "The prerequisites are: ",
                           "The constraints are as follows: ", "The requirements are as follows: ", "The restrictions are as follows: ", "The limitations are as follows: ", "The demands are as follows: ", "The prerequisites are as follows: ",
                           "The constraints are listed below: ", "The requirements are listed below: ", "The restrictions are listed below: ", "The limitations are listed below: ", "The demands are listed below: ", "The prerequisites are listed below: "]


def match_none(input_string):
    pattern = r"(?i)none[\W]*"
    return bool(re.match(pattern, input_string))


def construct_classification_task(output:str, wrong_outputs:list, add_constraints:bool=False):
    max_num = max([len(ALPHABET), len(INTEGER), len(MARK)])
    target_num = len(wrong_outputs) + 1
    if target_num > max_num:
        # cut the wrong outputs to the max number
        print("WARNING: The number of outputs {} is larger than the max number {}. Cut it to {}.".format(target_num, max_num, max_num))
        wrong_outputs = wrong_outputs[:max_num - 1]
        target_num = len(wrong_outputs) + 1
    all_symbols = []
    while len(all_symbols) < target_num:
        # randomly choose a category of symbols
        all_symbols = random.choice(OPTION_SYMBOLS)
    # randomly choose the option symbols in the category
    target_symbols = random.sample(all_symbols, target_num)
    # randomly choose the connect and bracket symbol
    connect_symbol = random.choice(CONNECT_SYMBOLS)
    bracket_symbol = random.choice(BRACKET_SYMBOLS)
    # combine symbols with outputs
    final_instructions = []
    assert len(target_symbols) == len(wrong_outputs) + 1
    correct_option = target_symbols[0]
    for ans, symbol in zip([output] + wrong_outputs, target_symbols):
        option = bracket_symbol[0] + symbol + bracket_symbol[1] + connect_symbol + " " + ans
        if option[-1] not in string.punctuation:
            option += "."
        final_instructions.append(option)
    # shuffle the options to avoid the short-cut
    random.shuffle(final_instructions)
    random.shuffle(target_symbols)
    # construction the final instruction and constraints
    option_instruction = "\n".join(final_instructions)
    option_constraints = random.choice(CLS_CONSTRANTS_LANGUAGE)
    constraints_bracket_symbol = random.choice(BRACKET_SYMBOLS)
    option_constraints += constraints_bracket_symbol[0]
    for id, symbol in enumerate(target_symbols):
        option_constraints += symbol
        if id != len(target_symbols) - 1:
            option_constraints += ", "
    option_constraints += constraints_bracket_symbol[1] + "."
    
    option_instruction = option_instruction + "\n" + option_constraints if add_constraints else option_instruction
    
    return option_instruction, correct_option
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='filtered_wrong_outs.json', help="file with wrong outputs added.")
    parser.add_argument("--save_file", type=str,
                        default='classification_added.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--instance_num", type=int, default=None, help="number of instances used for training.")
    parser.add_argument("--cls_num", type=int, default=1, help="number of classification tasks to add for each instance. Different classification tasks will use various templates (options).")
    parser.add_argument("--add_constraints", action="store_true", help="whether to add constraints to the classification tasks.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    # read the input file
    with open(os.path.join(args.path, args.data_files), "r") as f:
        source_datas = json.load(f)
        
    target_datas = []
    ori_ins_num, new_cls_ins_num = 0, 0
    for source_data in tqdm(source_datas, total=len(source_datas)):
        id, input, all_cost = source_data["id"], source_data["input"], source_data["cost"]
        instances = source_data["instances"]
        for idx, ins in enumerate(instances):
            new_source_data = {}
            new_source_data["id"] = id + "-" + str(idx)
            new_source_data["input"], new_source_data["cost"] = input, all_cost
            new_source_data["instances"] = []
            # put gen instance first
            first_ins = copy.deepcopy(ins)
            first_ins.pop("wrong_outputs")
            constraint = first_ins.pop("constraint")
            if constraint != "" and not match_none(constraint) and args.add_constraints:
                temp = random.choice(GEN_CONSTRANTS_LANGUAGE)
                first_ins["instruction"] += " " + temp + constraint
            new_source_data["instances"].append(first_ins)
            ori_ins_num += 1
            # add more cls-version instances
            for i in range(args.cls_num):
                new_ins = copy.deepcopy(ins)
                if len(new_ins["wrong_outputs"]) > 0:
                    # we only construct classification tasks when there are wrong outputs
                    option_instruction, correct_option = construct_classification_task(new_ins["output"], new_ins["wrong_outputs"], args.add_constraints)
                    new_ins["instruction"] += "\n" + option_instruction
                    new_ins["output"] = correct_option
                    new_ins.pop("wrong_outputs")
                    new_ins.pop("constraint")
                    new_source_data["instances"].append(new_ins)
                    new_cls_ins_num += 1
            target_datas.append(new_source_data)
            
    # if args.instance_num is not None:
    #     # uniformly sample the instances from each input, not recommended when there is a huge variance of the number of instances for each input
    #     print("*** Note that you choose to use only {} instances for training.".format(args.instance_num))
    #     random.shuffle(target_datas)
    #     avg_ins_per_input = math.ceil(args.instance_num / float(len(target_datas)))
    #     new_target_datas = []
    #     for input in target_datas:
    #         new_input = copy.deepcopy(input)
    #         new_input["instances"] = random.sample(new_input["instances"], min(avg_ins_per_input, len(new_input["instances"])))
    #         new_target_datas.append(new_input)
    #     target_datas = new_target_datas  
    
    # if args.instance_num is not None:
    #     # just simply use all the gen and cls tasks of each input, then better set cls_num to 1 to ensure the balance
    #     print("*** Note that you choose to use only {} instances for training.".format(args.instance_num))
    #     random.shuffle(target_datas)
    #     count = 0
    #     new_target_datas = []
    #     for input in target_datas:
    #         new_target_datas.append(input)
    #         count += len(input["instances"])
    #         if count >= args.instance_num:
    #             break
    #     target_datas = new_target_datas  
    
    if args.instance_num is not None:
        # randomly choose one task from the (one gen + multi cls) tasks of each input, better set cls_num to 1 to ensure the balance
        print("*** Note that you choose to use only {} instances for training.".format(args.instance_num))
        random.shuffle(target_datas)
        count = 0
        new_target_datas = []
        for input in target_datas:
            new_input = copy.deepcopy(input)
            new_input["instances"] = random.sample(new_input["instances"], 1)
            new_target_datas.append(new_input)
            count += 1
            if count >= args.instance_num:
                break
        target_datas = new_target_datas  
        
    # count how many instances are remained
    instances_num_list = []
    for input in target_datas:
        all_ins = input["instances"]
        instances_num_list.append(len(all_ins))
        
    print("Instance num:")
    print("==> all input: {}".format(len(target_datas)))
    print("==> all instances: {}, avg num for each input: {}".format(sum(instances_num_list), sum(instances_num_list)/len(instances_num_list) if len(instances_num_list) > 0 else 0))


    # name the save file as the instance num
    args.save_file = args.save_file.split(".")[0] + f"_{args.cls_num}.json"
    args.save_file = os.path.join(args.path, args.save_file)
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))
    
    # save the filtered data
    print("Save the filtered data at {}.".format(args.save_file))
    with open(args.save_file, "w", encoding="utf-8") as f:
        json.dump(target_datas, f, indent=2)
    
if __name__ == "__main__":    
    main()
    