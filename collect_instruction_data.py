'''
load and unify all the instruction data
currently only support the SuperNI
'''

import argparse
import copy
import json
import jsonlines
import os
import random
import numpy as np
from tqdm import tqdm

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    result = []
    for p in all_path:
        if not os.path.isdir(p) and sufix in p:
            result.append(os.path.join(task_path,p))
            
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path",type=str,default='../Tk-Instruct/data/tasks/def_segmentation')
    parser.add_argument("--split_path",type=str,default="../Tk-Instruct/data/splits/default")
    parser.add_argument("--save_path",type=str,default="./data/dummy")
    parser.add_argument("--num_dataset",type=int,default=20)
    parser.add_argument("--num_instance",type=int,default=10)
    parser.add_argument("--seed",type=int,default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    training_data_path = args.training_data_path
    split_path = args.split_path
    save_path = args.save_path
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    del_ent = lambda x:x[:-1]
    PREFIX = len(training_data_path) + 1
    train_num = 0
    train_tk_name = []
    
    print("==> read all the training tasks...")
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        # random select some tasks
        all_tr_tasks = random.sample(all_tr_tasks,min(args.num_dataset,len(all_tr_tasks)))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        
    all_tasks_pt = FindAllSuffix(training_data_path,"json")
    ins_cnt = 0
    resulting_data = []
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_tr_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            train_tk_name.append(tk_name)
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                instruction = tk_info["Definition"]
                instances = tk_info["Instances"]
                # random select some instances
                instances = random.sample(instances,min(args.num_instance,len(instances)))
                for ins in instances:
                    ins_cnt += 1
                    new_ins = {"id": "SuperNI-" + ins["id"],
                               "instruction": instruction[0],
                               "input": ins["input"],
                               "output": ins["output"]
                            }
                    resulting_data.append(new_ins)
    
    assert train_num == args.num_dataset, f"training dataset num {train_num} shoud be equal to arg {args.num_dataset}"
    
    # save data
    os.makedirs(save_path, exist_ok=True)
    
    print("==> save SuperNI data at {}, total {} instances".format(save_path, ins_cnt))
    
    with open(os.path.join(save_path, "SuperNI.json"), "w") as f:
        json.dump(resulting_data, f, indent=2)
        
if __name__ == "__main__":
    main()