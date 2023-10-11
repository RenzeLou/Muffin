import argparse
import copy
import json
import os
import random

from tqdm import tqdm


def process_model_predictions(predictions:list):
    processed_predictions = []
    for idx, item in tqdm(enumerate(predictions)):
        yes_prob = item.pop("yes_prob")
        no_prob = item.pop("no_prob")
        instructions = item.pop("instructions")
        outputs = item.pop("outputs")
        gt = item.pop("ground_truth")
        assert gt[0] == "Unknown", "the weired ground truth `{}`".format(gt[0])
        instruction = instructions[0]
        output = outputs[0]
        
        new_item = copy.deepcopy(item)
        new_item["instruction"] = instruction
        new_item["output"] = output
        # new_item["judge"] = "Unknown"  # dont have to set it as `Unknown`, just use this model prediction results
        
        processed_predictions.append(new_item)
    return processed_predictions

def process_to_annotate_answer(source_data:list):
    # process the judged data into a new format, that can be used for query GPT to annotate the answer.
    new_source_data = []
    for idx, item in tqdm(enumerate(source_data)):
        judge = item.pop("judge")
        if judge == "yes":
            # directly add the official instruction
            instructions = [item.pop("instruction")]
            item["instructions"] = instructions
            assert isinstance(item["output"], list)
            if len(item["output"]) > 0:
                item["output"] = random.choice(item["output"])
            else:
                item.pop("output")
            # following fields are not needed (placeholder)
            item["hint"] = ""
            item["cost"] = 0
            item["example_1"], item["example_2"], item["example_3"] = "", "", ""
            new_source_data.append(item)
        elif judge == "no":
            pass
        else:
            # warning 
            Warning("the weired judge `{}`of id `{}`".format(judge, item["id"]))
            pass
        
    return new_source_data  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_files", type=str, nargs="+", 
                        default=['superni_rematched.gpt.json', 'round_2/superni_rematched.gpt.json'],
                        help='path to the source files.')
    parser.add_argument("--model_predict_file", type=str, default=None,
                        help="optional, finally we only need those gpt annotation, but sometimes we also have to use local model's annotation.") # /scratch/rml6079/project/Instruct_dataset_training_code/out_classify_applicable/flan-t5-xl_3e-05_10_5050/predict_classify_answers.json
    parser.add_argument("--save_path", type=str, 
                        default='./data/SuperNI_v7', help='path to the save file.')
    parser.add_argument("--save_file", type=str,default="rematched.waited_annotate_y.json")  # file waited for gpt to annotate
    parser.add_argument("--overwrite",action="store_true", help="whether to overwrite the existed file")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, args.save_file)
    
    if os.path.exists(save_file) and not args.overwrite:
        raise ValueError("Save file already exists uder the {}, set --overwrite to overwrite it.".format(save_file))
    
    args.source_files = [os.path.join(args.save_path, source_file) for source_file in args.source_files]
    
    # process source data
    print("=== process the GPT predictions ===")
    source_data = []
    for source_file in args.source_files:
        with open(source_file, "r") as f:
            source_data.extend(json.load(f))
    print("==> total number of source predictions: {}".format(len(source_data)))
    target_data = process_to_annotate_answer(source_data)
    print("==> total number of matched pairs (pos predictions): {}".format(len(target_data)))

    
    # process the local model predictions (additional source data)
    if args.model_predict_file is not None:
        print("\n=== process the local model predictions ===")
        with open(args.model_predict_file, "r") as f:
            predictions = json.load(f)
        predictions = process_model_predictions(predictions)
        print("==> total number of model predictions: {}".format(len(predictions)))
        additional_target_data = process_to_annotate_answer(predictions)
        print("==> total number of matched pairs (pos predictions): {}".format(len(additional_target_data)))
        target_data.extend(additional_target_data)
        print("\ntotal number of matched instances: {}".format(len(target_data)))
    
    # save the target data
    with open (save_file, "w") as f:
        json.dump(target_data, f, indent=2)
        
    print("file saved to {}".format(save_file))
    

if __name__ == "__main__":
    main()
        
    
    