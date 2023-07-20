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
python generate_attributes.py --path ./data/SuperNI_v4 --data_file original_collection.json --save_file add_attributes.json --api_name gpt-3.5-turbo-0301 --overwrite
python generate_instructions.py --path ./data/SuperNI_v4 --data_file add_attributes.json --save_file add_generated_instructions_4.json --template 4 --api_name gpt-3.5-turbo-0301 --overwrite
python generate_instructions.py --path ./data/SuperNI_v4 --data_file add_attributes.json --save_file add_generated_instructions_3.json --template 3 --api_name gpt-3.5-turbo-0301 --overwrite
python generate_answers.py --path ./data/SuperNI_v4 --data_files add_generated_instructions_3.json,add_generated_instructions_4.json --save_file add_answers_full.json --api_name gpt-3.5-turbo-0613 --overwrite
python post_process/filtering.py --path ./data/SuperNI_v4 --data_files add_answers_full.json --save_file filtered_full.json --overwrite
python generate_wrong_candidates.py --api_name gpt-3.5-turbo-0613 --path ./data/SuperNI_v4 --data_files filtered_full.json --save_file add_classification_candidates.json --length_threshold 100 --overwrite
python post_process/classification_expansion.py --path ./data/SuperNI_v4 --data_files add_classification_candidates.json --save_file mix_cls.json --cls_num 1 --add_constraints --overwrite
```

```bash
python generate_attributes.py
python generate_instructions_CLS.py --path ./data/SuperNI_v4 --data_file add_attributes.json --save_file add_generated_instructions_cls.json --api_name gpt-3.5-turbo-0301 --overwrite
python generate_answers.py --path ./data/SuperNI_v4 --data_files add_generated_instructions_cls.json --save_file add_answers_cls.json --api_name gpt-3.5-turbo-0613 --overwrite
python post_process/filtering.py
mix
```