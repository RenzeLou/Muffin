import argparse
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ['NLTK_DATA'] = '/scratch/rml6079/nltk_data'
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

from tqdm import tqdm
# from post_process.classification_expansion import convert_integer_to_char
import sys
sys.path.append("./post_process")
from compute_metrics import rougeL_score # noqa

'''
# of input
# of input (from superni)  # not sure, perhaps put outside the table
# of input (from dolma)  # not sure, perhaps put outside the table
# of instructions (# of instances)
# of intructions from superNI
# of intructions from brainstorm
# of expanded classification instructions
# of instructions per input

ave. input length (in words)
ave. instruction length (in words)
ave. output length (in words)

# of unique instructions from superNI  # this is the unique number (# of tasks), while the actual instance number (after rematching) is much larger
'''

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

def plot_length_distribution(string_lengths, save_file="./string_length_distribution.pdf", x_name="String Length", y_name="Number of Samples", outlier_threshold=5, frequency_threshold=None):
    # delete some outliers
    # first count the frequency of each length
    length_freq = {}
    for length in string_lengths:
        if length not in length_freq:
            length_freq[length] = 0
        length_freq[length] += 1
    # then delete those lengths that only appear very few times
    if frequency_threshold is not None:
        new_string_lengths = []
        for length in string_lengths:
            if length_freq[length] <= frequency_threshold:
                new_string_lengths.append(length)
        string_lengths = new_string_lengths
    new_string_lengths = []
    for length in string_lengths:
        if length < outlier_threshold:
            new_string_lengths.append(length)
    string_lengths = new_string_lengths
    # Using numpy histogram function to get the frequency and bins
    freq, bins = np.histogram(string_lengths, bins=range(min(string_lengths), max(string_lengths)+2))

    # Plotting the histogram
    plt.bar(bins[:-1], freq, width=1, edgecolor='#636efa', align='center')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.title('String Length Distribution')
    plt.tight_layout()

    # Saving the figure as a PDF
    plt.savefig(save_file, format='pdf')
    
def add_input(instruction:str, instruction2input:dict, rouge_threshold:float=0.6, input:str=None):
    find_flag = False
    for ins in instruction2input.keys():
        if rougeL_score(instruction, ins) >= rouge_threshold:
            instruction2input[ins].add(input)
            find_flag = True
    if not find_flag:
        instruction2input[instruction] = set()
        instruction2input[instruction].add(input)
    
    return instruction2input
        


INTEGER = [str(i) for i in range(18278)] # from '0' to '18277'
ALPHABET = [convert_integer_to_char(i) for i in range(18288)] # from 'A' to 'ZZZ'
MARK = ["@", "#", "$", "%", "^", "&", "*", "+", "!", "?"]
OPTION_SYMBOLS = ALPHABET + MARK + INTEGER

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/SuperNI_v8/scale_up_data.part_0.68014.json")
    parser.add_argument("--rematch_data_path", type=str, default="./data/SuperNI_v7/rematched.round_1_2_3_4.filtered_chatgpt.json")
    parser.add_argument("--rouge_threshold", type=float, default=0.6, help="the rouge threshold to decide if the two instruction are the same or not.")
    parser.add_argument("--use_rouge", action="store_true", help="if set to True, use rougeL to match the instructions, otherwise use string matching.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    input_num, input_num_from_superni, input_num_from_dolma = 0, 0, 0
    instruction_num = 0
    instruction_num_from_superni = 0
    instruction_num_from_superni_unique = 0
    expanded_instruction_num = 0
    
    input_length_list = []
    instruction_length_list = []
    output_length_list = []
    
    # read the final data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    for item in tqdm(data):
        id = item["id"]
        input = item["input"]
        input_len = len(word_tokenize(input))
        input_length_list.append(input_len)
        
        input_num += 1
        if id.startswith("SuperNI"):
            input_num_from_superni += 1
        else:
            input_num_from_dolma += 1
        
        instruction_num += len(item["instances"])
        
        for ins in item["instances"]:
            instruction = ins["instruction"]
            output = ins["output"]
            intruction_len = len(word_tokenize(instruction))
            output_len = len(word_tokenize(output))
            instruction_length_list.append(intruction_len)
            output_length_list.append(output_len)
            if output in OPTION_SYMBOLS:
                # for those classification expansion instructions
                expanded_instruction_num += 1
    
    # read the rematched data
    with open(args.rematch_data_path, "r") as f:
        rematched_data = json.load(f)
    all_unique_superni_instructions = set()
    for item in tqdm(rematched_data):
        instruction_num_from_superni += len(item["instances"])
        for ins in item["instances"]:
            instruction = ins["instruction"]
            all_unique_superni_instructions.add(instruction)
    instruction_num_from_superni_unique = len(all_unique_superni_instructions)
    
    # calculate the input num per instruction
    instruction2input = {}
    for item in tqdm(data):
        id = item["id"]
        input = item["input"]
        for ins in item["instances"]:
            instruction = ins["instruction"]
            instruction = instruction.strip()
            if not args.use_rouge:
                # simply string matching
                if instruction not in instruction2input:
                    instruction2input[instruction] = set()
                    instruction2input[instruction].add(input)
                else:
                    instruction2input[instruction].add(input)
            else:
                # use rougeL to match
                instruction2input = add_input(instruction, instruction2input, args.rouge_threshold, input)
                
    input_num_per_instruction = []
    for instruction, inputs in instruction2input.items():
        input_num_per_instruction.append(len(inputs))
            
    
    print("input_num: {}".format(input_num))
    print("input_num_from_superni: {}".format(input_num_from_superni))
    print("input_num_from_dolma: {}".format(input_num_from_dolma))
    print("instruction_num: {}".format(instruction_num))
    print("instruction_num_from_superni: {}".format(instruction_num_from_superni))
    print("instruction_num_from_superni_unique: {}".format(instruction_num_from_superni_unique))
    print("expanded_instruction_num: {}".format(expanded_instruction_num))
    print("instruction_num_from_brainstorm: {}".format(instruction_num - instruction_num_from_superni - expanded_instruction_num))
    print("instruction_num_per_input: {:.4f}".format(instruction_num / input_num))
    print("input_num_per_instruction: {:.4f}".format(sum(input_num_per_instruction) / len(input_num_per_instruction)))
    print("ave. input length (in words): {:.4f}".format(sum(input_length_list) / len(input_length_list)))
    print("ave. instruction length (in words): {:.4f}".format(sum(instruction_length_list) / len(instruction_length_list)))
    print("ave. output length (in words): {:.4f}".format(sum(output_length_list) / len(output_length_list)))
    
    # print("the min output length (in words): {}".format(min(output_length_list)))
    # exit()
    
    # plot length distribution
    os.makedirs("./utils/data_statistics", exist_ok=True)
    input_len_save_file = "./utils/data_statistics/input_length_distribution.pdf"
    instruction_len_save_file = "./utils/data_statistics/instruction_length_distribution.pdf"
    output_len_save_file = "./utils/data_statistics/output_length_distribution.pdf"
    plot_length_distribution(input_length_list, save_file=input_len_save_file, x_name="Input Length", y_name="# Inputs", outlier_threshold=1500, frequency_threshold=40)
    plot_length_distribution(instruction_length_list, save_file=instruction_len_save_file, x_name="Instruction Length", y_name="# Instructions", outlier_threshold=650)
    plot_length_distribution(output_length_list, save_file=output_len_save_file, x_name="Output Length", y_name="# Outputs", outlier_threshold=200, frequency_threshold=15000)
    

    
if __name__ == "__main__":
    main()
        
    