data_file=$1
api_key=$2


# Instruction Rematching
echo "Instruction rematching..."
python classify_instruction_validation.py \
 --api_name gpt-4-0613 \
 --path ./data \
 --data_file ${data_file} \
 --save_file rematched.gpt.json \
 --overwrite \
 --api_key ${api_key} # better to set environment variable OPENAI_API_KEY, and remove this line


# reformat the data file
python post_process/process_rematched_results.py \
 --source_files rematched.gpt.json \
 --save_path ./data \
 --save_file rematched.waited_annotate_y.json \
 --overwrite


# Output Annotation
python generate_answers_on_rematched.py \
 --api_name gpt-3.5-turbo-0613 \
 --path ./data \
 --data_files rematched.waited_annotate_y.json \
 --save_file rematched.add_answers.json \
 --overwrite \
 --cancel_filter \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line

# Instruction Filtering
python post_process/filtering_rematch.py \
 --path ./data \
 --data_files rematched.add_answers.json \
 --save_file rematched.json \
 --overwrite