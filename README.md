`collect_instruction_data.py` is used to load and reorganize the data, currently only support the SuperNI dataset.

run `python generate_attributes.py --api_key xxx` to generate the attributes.
run `python generate_instructions.py --api_key xxx` to generate the instructions based on the attributes.

the prompt template can be found in `prompt_templates.py`
the querying procedure can be found in `chat_completion.py` (shared by both `generate_attributes.py` and `generate_instructions.py`)
