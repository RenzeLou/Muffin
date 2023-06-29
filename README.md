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

TODO:

```bash

```