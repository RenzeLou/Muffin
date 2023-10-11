data_file=$1
api_key=$2

# Facets Recognition
echo "Facets Recognition..."
python generate_attributes.py --path ./data \
 --data_file ${data_file} \
 --save_file add_attributes.json \
 --api_name gpt-3.5-turbo-0301 \
 --template 2 \
 --overwrite \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line


# Instruction Brainstorm
echo "Instruction Brainstorm (hint prompt)..."
python generate_instructions.py --path ./data \
 --data_file add_attributes.json \
 --save_file add_generated_instructions_5.json \
 --template 5 \
 --api_name gpt-3.5-turbo-0301 \
 --overwrite \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line

echo "Instruction Brainstorm (shift attribute)..."
python generate_instructions.py --path ./data \
 --data_file add_attributes.json \
 --save_file add_generated_instructions_6.json \
 --template 6 \
 --api_name gpt-3.5-turbo-0301 \
 --overwrite \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line


# Output Annotation
echo "Output Annotation..."
python generate_answers.py --path ./data \
 --data_files add_generated_instructions_5.json,add_generated_instructions_6.json \
 --save_file add_answers_full.json \
 --api_name gpt-3.5-turbo-0613 \
 --overwrite \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line


# Instruction Filtering
echo "Instruction Filtering..."
python post_process/filtering.py --path ./data \
 --data_files add_answers_full.json \
 --save_file filtered_full.json \
 --overwrite


# Classification Expansion
echo "Classification Expansion..."
python generate_wrong_candidates.py --path ./data \
 --api_name gpt-3.5-turbo-0613 \
 --data_files filtered_full.json \
 --save_file add_classification_candidates.json \
 --length_threshold 100 \
 --overwrite \
 --api_key ${api_key}  # better to set environment variable OPENAI_API_KEY, and remove this line

python post_process/classification_expansion.py --path ./data \
 --data_files add_classification_candidates.json \
 --save_file brainstorm.json \
 --cls_num 1 \
 --add_constraints \
 --overwrite