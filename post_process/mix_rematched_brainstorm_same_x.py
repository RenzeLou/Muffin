'''
mix the brainstorm data and rematched data, and also combine those instances with the same x
'''

import argparse
import copy
import json
import os
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainstorm_data", type=str, default="./data/SuperNI_v4/mix_cls_1.json")
    parser.add_argument("--remathced_data", type=str, default="./data/SuperNI_v7/rematched.round_1_2_3_4.filtered_chatgpt.json")
    parser.add_argument("--mix_save_file", type=str, default="mix_remacthed_brainstorm.json")
    parser.add_argument("--mix_save_path", type=str, default="./data/SuperNI_v7")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    brainstorm_data = args.brainstorm_data
    remathced_data = args.remathced_data
    os.makedirs(args.mix_save_path, exist_ok=True)
    mix_save_file = os.path.join(args.mix_save_path, args.mix_save_file)
    shuffle = args.shuffle
    overwrite = args.overwrite
    seed= args.seed
    random.seed(seed)

    # process the brainstorm data
    with open(brainstorm_data, 'r') as f:
        brainstorm_data = json.load(f)

    new_brainstorm_data = []
    for item in tqdm(brainstorm_data):
        instances = item['instances']
        assert len(instances) <= 2, "mix_cls_1.json should have 2 instances (one gen, one cls)"
        # randomly select one instruction from cls and gen paradigms
        selected_ins = random.choice(instances)
        item['instances'] = [selected_ins]
        new_brainstorm_data.append(item)
        
    print("\n==> for brainstorm data")
    print("total {} instance".format(len(new_brainstorm_data)))

    # reformulate the brainstorm data (same as rematched data), combine all the task instructions with the same input x
    input_id2ins = {}
    for item in tqdm(new_brainstorm_data):
        id = item['id']
        input_id = id.rsplit("-", 1)[0]
        # print("input_id: {}".format(input_id))
        # exit()
        if input_id not in input_id2ins:
            input_id2ins[input_id] = [item]
        else:
            input_id2ins[input_id].append(item)
            
    reformulated_brainstorm_data = []
    for input_id, items in input_id2ins.items():
        instances = []
        for item in items:
            instances.extend(item['instances'])
        reformulated_brainstorm_data.append({
            'id': input_id,
            'input': items[0]['input'],
            'instances': instances
        })

    print("\n==> for reformulated brainstorm data")
    print("total {} input".format(len(reformulated_brainstorm_data)))
        
        
    # process the rematched data
    with open(remathced_data, 'r') as f:
        remathced_data = json.load(f)
        
    merged_data = []
    find_cnt, no_find_cnt = 0, 0
    ins_cnt_list = []
    reformulated_brainstorm_data_remain = copy.deepcopy(reformulated_brainstorm_data)
    for item in tqdm(remathced_data):
        input_id = item['id']
        new_item = copy.deepcopy(item)
        # find the brainstorm data with the same input id
        find_flag = False
        for brainstorm_item in reformulated_brainstorm_data:
            if brainstorm_item['id'] == input_id:
                # combine the instances
                all_instances = brainstorm_item['instances'] + item['instances']
                if shuffle:
                    random.shuffle(all_instances)
                new_item['instances'] = all_instances
                find_flag = True
                # delete this item from the brainstorm data
                reformulated_brainstorm_data_remain.remove(brainstorm_item)
                break
        merged_data.append(new_item)
        ins_cnt_list.append(len(new_item['instances']))
        if find_flag:
            find_cnt += 1
        else:
            no_find_cnt += 1
            
    print("\n=====================")
    print("finally, find {} same input from brainstormed data, no find {} input".format(find_cnt, no_find_cnt))
    print("total {} input, {} instances, {} instructions per input".format(len(merged_data), sum(ins_cnt_list), sum(ins_cnt_list)/len(merged_data)))

    # dont forget the remaining brainstorm data
    merged_data.extend(reformulated_brainstorm_data_remain)

    # save the merged data
    mix_save_file = mix_save_file.replace(".json", ".{}.json".format(sum(ins_cnt_list)))
    if os.path.exists(mix_save_file) and not overwrite:
        raise ValueError("file {} already exists, please set overwrite=True".format(mix_save_file))

    with open(mix_save_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
        
    print("\n==> save the merged data to {}".format(mix_save_file))
    


if __name__ == "__main__":
    main()
