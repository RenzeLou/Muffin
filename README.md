`collect_instruction_data.py` is used to load and reorganize the data, currently only support the SuperNI dataset.

run `python generate_attributes.py --api_key xxx` to generate the attributes.
run `python generate_instructions.py --api_key xxx` to generate the instructions based on the attributes.
run `python generate_answers.py` to annotate the answers.

the prompt template can be found in `prompt_templates.py`
the querying procedure can be found in `chat_completion.py` (shared by both `generate_attributes.py`, `generate_instructions.py` and `generate_answers.py`)


use `python filter/filtering.py` to filter the data. The resulting data will be saved as f`filtered_{instance_num}.json`

TODO:

```bash
python generate_instructions.py --path ./data/SuperNI_v2 --data_file add_attributes.json --save_file add_generated_instructions_2.json --template 2 --instance_num 200

python generate_instructions.py --path ./data/SuperNI_v2 --data_file add_attributes.json --save_file add_generated_instructions_1.json --template 1 --instance_num 200
python generate_answers.py --path ./data/SuperNI_v2 --data_files add_generated_instructions_1.json,add_generated_instructions_2.json --save_file add_answers_full.json
```