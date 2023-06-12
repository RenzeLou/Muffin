'''
Use this script to conduct a simple filtering on generated instruction data
Two rules:
(1). delete the same instructions for each input
(2). delete those no-answer instructions (invalid instructions)
(3). [optional] delete those input with only a few instructions (threshold); the threshold is set empirically (according to the data distribution)

Also print the statistics of the data after filtering.
'''

import argparse
import json
import os
import re

from tqdm import tqdm


def del_exist_instructions(current_instructions:list, new_instructions:list):
    '''
    Delete the same instructions for each input
    '''
    new_instructions = [ins for ins in new_instructions if ins not in current_instructions]
    return new_instructions

def check_no_answer_reponse(input_string:str):
    '''
    Check ChatGPT's meaningless response, such as:
    "Sorry, I cannot create a quiz as I am a language model AI..."
    '''
    pattern = r"(?i)Sorry.*(?:AI Language model|Language model AI|Language AI Model)"
    return bool(re.search(pattern, input_string))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
                        default='./data/dummy/', help='source file & target save path.')
    parser.add_argument("--data_files", type=str,
                        default='add_answers.json,add_answers_2.json', help="one or more input files, separated by comma.")
    parser.add_argument("--save_file", type=str,
                        default='filtered.json')
    parser.add_argument("--seed", type=int, default=42)
    # TODO: I don't know if these meanningless responses are also helpful for tuning the model (alignments)
    parser.add_argument("--del_no_answer", action="store_true", help="whether to delete ChatGPT's no-answer response, such as 'Sorry, I cannot create a quiz as I am a language model AI...'.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    # read all the input files
    all_inputs = []
    data_files = args.data_files.split(",")
    for data_file in data_files:
        data_file = os.path.join(args.path, data_file)
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                inputs = json.load(f)
                all_inputs.extend(inputs)
        else:
            raise ValueError("Input file {} does not exist.".format(data_file))
        
    
    # for each input, delete the same instructions
    print("Start filtering...")
    print("==> delete the same instructions for each input")
    all_inputs_del = []
    same_inst_del_num_list = []
    for input in tqdm(all_inputs):
        all_instructions = input["instances"]
        # each element in all_instructions is a dict, like
        # {
        #     "instruction": "Identify all the uppercase letters in the input.",
        #     "output": "['S', 'I', 'D', 'R', 'R']",
        #     "cost": 174
        #   }
        # delete the elemtents with the same instruction
        all_instructions_del = []
        same_inst_del_num = 0
        for idx, instruction in enumerate(all_instructions):
            if instruction["instruction"] not in [ins["instruction"] for ins in all_instructions_del]:
                all_instructions_del.append(instruction)
            else:
                same_inst_del_num += 1
        same_inst_del_num_list.append(same_inst_del_num)
        input["instances"] = all_instructions_del
        all_inputs_del.append(input)
    
    print("==> delete those no-answer instructions")
    all_inputs_del_2 = []
    no_answer_del_num_list = []
    for input in tqdm(all_inputs_del):
        all_instructions = input["instances"]
        all_instructions_del = []
        no_answer_inst_del_num = 0
        for instruction in all_instructions:
            output = instruction["output"]
            # delete empty output
            if output == "":
                no_answer_inst_del_num += 1
                continue
            # delete meaningless response
            if args.del_no_answer and check_no_answer_reponse(output):
                no_answer_inst_del_num += 1
                continue
            all_instructions_del.append(instruction)
        no_answer_del_num_list.append(no_answer_inst_del_num)
        input["instances"] = all_instructions_del
        all_inputs_del_2.append(input)
    
    # TODO: delete those input with only a few instructions (threshold); but I am not sure if this is helpful.
    
    print("Delete num:")
    print("==> identical instructions: {}, avg del num for each input: {}".format(sum(same_inst_del_num_list), sum(same_inst_del_num_list)/len(same_inst_del_num_list) if len(same_inst_del_num_list) > 0 else 0))
    print("==> no-answer instructions: {}, avg del num for each input: {}".format(sum(no_answer_del_num_list), sum(no_answer_del_num_list)/len(no_answer_del_num_list) if len(no_answer_del_num_list) > 0 else 0))
    
    # count how many instances are remained
    instances_num_list = []
    for input in all_inputs_del_2:
        instructions = input["instances"]
        instances_num_list.append(len(instructions))
        
    print("Instance num:")
    print("==> all input: {}".format(len(all_inputs_del_2)))
    print("==> all instances: {}, avg num for each input: {}".format(sum(instances_num_list), sum(instances_num_list)/len(instances_num_list) if len(instances_num_list) > 0 else 0))


    # name the save file as the instance num
    args.save_file = args.save_file.split(".")[0] + f"_{sum(instances_num_list)}.json"
    args.save_file = os.path.join(args.path, args.save_file)
    if os.path.exists(args.save_file) and not args.overwrite:
        raise ValueError("Save file {} already exists, set --overwrite to overwrite it.".format(args.save_file))
    
    # save the filtered data
    print("Save the filtered data at {}.".format(args.save_file))
    with open(args.save_file, "w") as f:
        json.dump(all_inputs_del_2, f, indent=2)
    
if __name__ == "__main__":    
    main()
    