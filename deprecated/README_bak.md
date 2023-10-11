### prompt template
the prompt template can be found in `prompt_templates.py`
the querying procedure can be found in `chat_completion.py` (shared by `generate_attributes.py`, `generate_instructions.py`, `generate_answers.py`, and so on)

### source data collection
`collect_instruction_data.py` is used to load and reorganize the data, currently only support the SuperNI dataset.

### instance (input, instructions, outputs) generation
run `python generate_attributes.py --api_key xxx` to generate the attributes.
run `python generate_instructions.py --api_key xxx` to generate the instructions based on the attributes.
run `python generate_constraints.py` to add additional constraints to the instructions. (optional)
run `python generate_answers.py` to annotate the answers based on the (input, instruction, constraint).

### filtering 
use `python post_process/filtering.py` to filter the data. The resulting data will be saved as f`filtered_{instance_num}.json`

### classification expansion
use `python generate_wrong_candidates.py` to generate more wrong output candidates for each task. Then, use `python post_process/classification_expansion.py` to construct more classification instances by combining the wrong candidates with correct outputs. 

## About API
use `gpt-3.5-turbo-0301` for `generate_attributes.py`, `generate_instructions.py`.
use `gpt-3.5-turbo-0613` for `generate_constraints.py`, `generate_answers.py`, `generate_wrong_candidates.py`.

## TODO:

```bash
python generate_attributes.py --path ./data/SuperNI_v8 --data_file part_1.json --save_file part_1.add_attributes.json --api_name gpt-3.5-turbo-0301 --template 2
python generate_instructions.py --path ./data/SuperNI_v8 --data_file part_1.add_attributes.json --save_file part_1.add_generated_instructions_5.json --template 5 --api_name gpt-3.5-turbo-0301
python generate_instructions.py --path ./data/SuperNI_v8 --data_file part_1.add_attributes.json --save_file part_1.add_generated_instructions_6.json --template 6 --api_name gpt-3.5-turbo-0301
python generate_answers.py --path ./data/SuperNI_v8 --data_files part_1.add_generated_instructions_5.json,part_1.add_generated_instructions_6.json --save_file part_1.add_answers_full.json --api_name gpt-3.5-turbo-0613
python post_process/filtering.py --path ./data/SuperNI_v8 --data_files part_1.add_answers_full.json --save_file part_1.filtered_full.json
python generate_wrong_candidates.py --api_name gpt-3.5-turbo-0613 --path ./data/SuperNI_v8 --data_files part_1.filtered_full.json --save_file part_1.add_classification_candidates.json --length_threshold 100
python post_process/classification_expansion.py --path ./data/SuperNI_v8 --data_files part_1.add_classification_candidates.json --save_file part_1.mix_cls.json --cls_num 1 --add_constraints
```

```bash
python generate_attributes.py
python generate_instructions_CLS.py --path ./data/SuperNI_v4 --data_file add_attributes.json --save_file add_generated_instructions_cls.json --api_name gpt-3.5-turbo-0301 --overwrite
python generate_answers.py --path ./data/SuperNI_v4 --data_files add_generated_instructions_cls.json --save_file add_answers_cls.json --api_name gpt-3.5-turbo-0613 --overwrite
python post_process/filtering.py
python mix_additional_cls_data.py ("remember to change the path")
```


```bash
python classify_cls_instruction_validation.py --path ./data/SuperNI_v4 --data_file original_collection.json --save_file classified_superni_cls_instructions.json --api_name gpt-3.5-turbo-0613 --overwrite
python generate_answers.py --path ./data/SuperNI_v4 --data_files classified_superni_cls_instructions.json  --save_file add_answers_superni_cls_instructions.json --api_name gpt-3.5-turbo-0613 --cancel_filter --overwrite --no_hint_id --api_key xxx  # (our group's api key)
```


### rematch superni

```bash
python data/split_ori_collection_2.py --annotate_instruction_num 100 --overwrite  # split the classification procedure (judge whether the instruction and the given input are matched) into two parts, one part use openai and another part use local model)

python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v6 --data_file superni_rematched_waited_classify.gpt.json --save_file superni_rematched.gpt.json --overwrite # (the first half,gpt-annotated part)

# python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v6 --data_file superni_rematched_waited_classify.local.json --save_file superni_rematched.local.json --overwrite # (the second half, or local-model-annotated part)
```

after using gpt-4 annotate all these subset of instruction (100/756), train a local model (flan-t5) to annotate the rest of the instructions

```bash
cd /scratch/rml6079/project/Instruct_dataset_training_code

# training 
sh scripts/classify_applicable.sh 7 6 google/flan-t5-xl SuperNI_v7/t5_superni_rematched.gpt.json SuperNI_v7/t5_dev_superni_rematched.gpt.json SuperNI_v7/t5_test_superni_rematched_waited_classify.local.json out_classify_applicable 5050  # have fixed the hyperparameters, use `classify_applicable_tune_basic_hyper.sh` to decide hyperparameters at first
# testing 
sh scripts/classify_applicable_predict.sh 7 36 flan-t5-xl_3e-05_10_5050 SuperNI_v7/t5_test_superni_rematched_waited_classify.local.json 0.2
```

then process the local model's prediction, use GPT to do the 2nd round annotation

```bash
# process the local model's prediction, randomly select 45% of the data to annotate (10k)
python data/process_local_model_results.py --save_path ./data/SuperNI_v7/round_2 --margin 0.5 --random_ratio 0.45 --overwrite

# 2nd round annotation
python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v7/round_2 --data_file superni_rematched_waited_classify.gpt.json --save_file superni_rematched.gpt.json
```

then do the same things like before: use all the current gpt-annotated data to train a stronger local small model, and do prediction on the rest of the data, and 3rd round annotation

```bash
# 3rd round annotation
python data/process_local_model_results.py --model_predict_file /scratch/rml6079/project/Instruct_dataset_training_code/out_classify_applicable/flan-t5-xl_3e-05_10_13495/predict_classify_answers.json --save_path ./data/SuperNI_v7/round_3 --margin 0.2 --random_ratio 0.75 --overwrite
python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v7/round_3 --data_file superni_rematched_waited_classify.gpt.json --save_file superni_rematched.gpt.json
```

```bash
# 4th round annotation
python data/process_local_model_results.py --model_predict_file /scratch/rml6079/project/Instruct_dataset_training_code/out_classify_applicable/flan-t5-xl_3e-05_5_21793/predict_classify_answers.json --save_path ./data/SuperNI_v7/round_4 --margin 0.0 --random_ratio 1 --overwrite
python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v7/round_4 --data_file superni_rematched_waited_classify.gpt.json --save_file superni_rematched.gpt.json
```


```bash
# merge all-round gpt-annotated data together, reformat them to train T5
python data/merge_gpt4_annotation_all_round.py --source_path ./data/SuperNI_v7 --source_files superni_rematched.gpt.json round_2/superni_rematched.gpt.json --save_path ./data/SuperNI_v7/round_2 --save_file superni_rematched.gpt.json --format_process --local_file superni_rematched_waited_classify.local.json --neg_num 7000  --overwrite
python data/merge_gpt4_annotation_all_round.py --source_path ./data/SuperNI_v7 --source_files superni_rematched.gpt.json round_2/superni_rematched.gpt.json round_3/superni_rematched.gpt.json --save_path ./data/SuperNI_v7/round_3 --save_file superni_rematched.gpt.json --format_process --local_file superni_rematched_waited_classify.local.json --neg_num 11000 --overwrite
```

finally, after getting all the GPT-anntated samples, collect those pos samples and format data file, and annotate y

```bash
# gather all-round gpt-annotated data together, reformat them
python data/process_rematched_results.py

python generate_answers_on_rematched.py --api_name gpt-3.5-turbo-0613 --path ./data/SuperNI_v7 --data_files rematched.round_1.waited_annotate_y.json --save_file rematched.round_1.add_answers.json --overwrite --cancel_filter --api_key xxx

python post_process/filtering_rematch.py --path ./data/SuperNI_v7 --data_files rematched.round_1.add_answers.json --save_file rematched.round_1.filtered.json
```


**below procedures are optional**. Since 100/756 is also expensive, and it constantly faces with openai api error. So we split the 100/756 into 5 parts **again**, use different group accounts to annotate the instructions. Then, merge the all these small pieces of annotations into one file.

```bash
python data/split_gpt4_annotation.py  # due to so poor, split this subset again... use different group accounts to annotate the instructions
python classify_instruction_validation.py --api_name gpt-4-0613 --path ./data/SuperNI_v7/split_data_dueto_poor --data_file split_3_superni_rematched_waited_classify.gpt.json --save_file split_3_superni_rematched.gpt.json --api_key xxx
python data/merge_gpt4_annotation.py --overwrite --format_process  # merge the annotations from different group accounts
```


### merge rematched & brainstorm data

after finishing annotating both rematched data and brainstorm data, we can merge them together. there are two ways:

#### 1. merge according to the same input (recommend)

following code will use rematched data as the base, and finding those instructions from brainstorm data that share the same input with rematched data, and merge those same-x instructions together.

(if there is no sahred x, then the final file is the same as rematched data)

```bash
python data/mix_rematched_brainstorm_same_x.py --brainstorm_data ./data/SuperNI_v7/mix_cls_long_1.json --remathced_data ./data/SuperNI_v7/rematched.round_1_2_3_4.filtered_chatgpt.json --mix_save_path ./data/SuperNI_v7 --mix_save_file mix_v2.remacthed_brainstorm.long.json
```

then after scaling (use the input text from dolma), merge the above same_x_mix data with the new brainstorm data

```bash
python data/mix_rematched_brainstorm_more_data.py # as for the args, see the file, should be changed accordinglly (incrementallt scale up)
```

#### 2. simply combine

while the following code just simply combine these two data, you can also set `--instance_num` to decie how many brainstorm data is added into rematch data

```bash
cd data
python mix_rematched_brainstorm.py  # mod the args in this file
```

----------------------------------------------------------------------------------------

# about scaling up -- adding more input x

use the following scripts to process and collect free-form text (x) from dolma:

```bash
python pre_process/process_dolma.py --select_num 3500 --length_diversity --num_per_part 750
```