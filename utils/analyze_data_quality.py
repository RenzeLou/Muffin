import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/SuperNI_v8/scale_up_data.part_0.68014.json")
    parser.add_argument("--select_num", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_path", type=str, default="./utils/data_quality") # same as `--path` by default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled", action="store_true", help="if set to True, means the input data is already sampled, there is no need to do the further sampling")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    if not args.sampled:
        select_num = min(args.select_num, len(data))
        selected_data = random.sample(data, select_num)
        print("==> totaly {} input, select {} input".format(len(data), len(selected_data)))
        
        new_selected_data = []
        for item in selected_data:
            # randomly sample 1 instance from each input
            item["instances"] = [random.choice(item["instances"])]
            new_selected_data.append(item)
        # save the sampled data for further sharing
        with open(os.path.join(args.save_path, "sampled_data_{}.json".format(select_num)), "w") as f:
            json.dump(new_selected_data, f, indent=2)
    else:
        new_selected_data = data
        print("==> begin annotating {} instances".format(len(new_selected_data)))
    
    selected_data_annotated = []
    for idx, item in enumerate(new_selected_data):
        print("\n*** {} / {} examples ***".format(idx, len(new_selected_data)))
        # print the instruction, input, output on the screen
        print("=" *20 + " instruction: " + "=" * 20)
        print(item["instances"][0]["instruction"])
        print("=" * 20 + " input: " + "=" * 20)
        print(item["input"])
        print("=" * 20 + " output: " + "=" * 20)
        print(item["instances"][0]["output"])
        print("=" * 60)
        # Q1: Does the instruction describe a valid task that can be answered?
        while True:
            print("Q1: Does the instruction describe a valid task that can be answered? ('y'/'n')")
            print("A1: ")
            answer = input()
            if answer == "y":
                break
            elif answer == "n":
                break
            else:
                print("please input 'y' or 'n'")
        item["A1"] = answer
        # Q2: Is this instruction appropriately matches the input?
        while True:
            print("Q2: Is this instruction appropriately matches the input? ('y'/'n')")
            print("A2: ")
            answer = input()
            if answer == "y":
                break
            elif answer == "n":
                break
            else:
                print("please input 'y' or 'n'")
        item["A2"] = answer
        # Q3: Is the output acceptably responds to the instruction and input?
        while True:
            print("Q3: Is the output acceptably responds to the instruction and input? ('y'/'n')")
            print("A3: ")
            answer = input()
            if answer == "y":
                break
            elif answer == "n":
                break
            else:
                print("please input 'y' or 'n'")
        item["A3"] = answer
        selected_data_annotated.append(item)
    
    # save the annotated data
    with open(os.path.join(args.save_path, "annotated_data_{}.json".format(len(selected_data_annotated))), "w") as f:
        json.dump(selected_data_annotated, f, indent=2)
    
    print("\n===> save the annotated data to {}".format(os.path.join(args.save_path, "annotated_data_{}.json".format(len(selected_data_annotated)))))
    

if __name__ == "__main__":
    main()
        