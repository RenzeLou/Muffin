# This file contains the prompt templates used in chat completion.
import re

# parent class
class ConversationPrompt(object):
    def __init__(self):
        self.system = (
            "You are a creative assistant. " +
            "Your responsibility is to understand the user's instructions and help brainstorm novel ideas."
        )
        # filter based on keywords that are not suitable for language models.
        self.blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "poster",
            "music",
            "flowchart",
            "diagram",
            # "media",
            # "media post",
        ]

    def extract_content(self, content:str):
        # Remove the "None" at the end
        # content = re.sub(r"(None\.|None|none\.|none)", "", content) 
        content = re.sub(r"(None\.?|none\.?)+$", "", content)
        items = re.split(self.seperator, content)
        # Remove any empty elements from the list
        items = [item.strip() for item in items if item.strip() != ""]
        # Remove any elements that has words in the blacklist
        items = [item for item in items if not any([word in item for word in self.blacklist])] 
        return items
    

# used for generating textual facets (attributes) 
class ConversationPromptAttribute(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Natural Language Processing (NLP)."
        )
        self.seperator = r"Attribute \d+:"  ## Attribute 1: Attribute 2: .. Attribute n:
        self.requirement_prompt = (
            "1. Please brainstorm as many textual attributes as possible. If you think there are no more suitable attributes, end up with 'None'.\n" +
            "2. Be creative. Any interesting perspectives are welcome!\n" +
            "3. Each attribute must concisely summarize one specific aspect of this input.\n" +
            "4. Feel free to ignore the tedious and specific content. Just focus on some general textual attributes!\n" +
            "5. Please prioritize your most confident predictions.\n"
        )
        self.query_prompt = (
            "You are an expert in Natural Language Processing (NLP). Carefully read the given input (consists of one or several pieces of text), and find out the textual attributes of this input.\n\n" +
            "### Input:\n" +
            "{input}\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Attribute 1:\n"            
        )
        
# used for generating textual attributes (generally more attributes than the first template)
class ConversationPromptAttribute_2(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.seperator = r"Attribute \d+:"  ## Attribute 1: Attribute 2: .. Attribute n:
        self.requirement_prompt = (
            "1. Please brainstorm as many textual attributes as possible. If you think there are no more suitable attributes, end up with 'None'.\n" +
            "2. Be creative. Any interesting perspectives are welcome!\n" +
            "3. Each attribute must concisely summarize one specific aspect of this input, such as language, length, intent, etc.\n" +
            "4. Feel free to ignore the tedious and specific content. Just focus on some general textual attributes!\n" +
            "5. Please prioritize your most confident predictions.\n" 
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Instruction:\n" +
            "Given the above input, what kind of textual attributes it has?\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Attribute 1:\n"
        )
        
        
# used for generating task instructions     
class ConversationPromptTask(ConversationPrompt):
    ''' following the hint to brainstorm novel task instructions '''
    def __init__(self):
        super().__init__()
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" + 
            "2. You'll need to look at the hint as a direction to guide your thinking. Your responses should strictly be based on the hint!\n" +
            "3. Each task must be indivisible (one task, one intention).\n" +
            "4. Avoid tasks requiring additional context, i.e., tasks must be easily solved/answered using only the input.\n" +
            "5. Please prioritize your most confident predictions.\n"
            "6. You should explicitly mention any necessary output constraints, such as answer options or length limitations.\n" +
            "7. But do not disclose the final answer!\n"
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Instruction:\n" +
            "What kind of textual tasks can you develop that can be applied to the input?\n\n" +
            "### Hint:\n" +
            "{hint}\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Task 1:\n"
        )


class ConversationPromptTask_2(ConversationPrompt):
    ''' shifting the attribute to generate task instructions '''
    def __init__(self):
        super().__init__()
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" + 
            "2. Your tasks can be applied to the input to shift its attribute.\n" +
            "3. Each task must be indivisible (one task, one intention).\n" +
            "4. Avoid tasks requiring additional context, i.e., tasks must be easily solved/answered using only the input.\n" +
            "5. Please prioritize your most confident predictions.\n"
            "6. You should explicitly mention any necessary output constraints, such as answer options or length limitations.\n" +
            "7. But do not disclose the final answer!\n"
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Attribute:\n" +
            "{hint}\n\n" +
            "### Instruction:\n" +
            "Based on the above information, what kind of textual tasks can you develop that can shift the input's attribute?\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Task 1:\n"
        )
        

class ConversationPromptTask_5(ConversationPrompt):
    ''' following the hint to brainstorm novel task instructions, also add 3-shot in-domain demonstrations '''
    def __init__(self):
        super().__init__()
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" +
            "2. You'll need to look at the hint as a direction to guide your thinking. Your responses should strictly be based on this hint!\n" +
            "3. Each task must be indivisible (one task, one intention).\n" +
            "4. Please prioritize your most confident predictions.\n" +
            "5. Avoid tasks requiring additional context, i.e., tasks must be easily solved/answered using only the input.\n" +
            "6. Your tasks should sufficiently describe how this input is expected to be mapped to an output text, i.e., elaborating the tasks in detail.\n" +
            "7. But do not disclose the final answer!\n"
        )
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n\n" +
            "### In a word, you should first describe what the input is, and what textual attribute it has, then elaborate on the task intent, and finally exactly describe what kind of output you expect and mention any necessary output constraints (e.g., formats, options).\n"
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Hint:\n" +
            "{hint}\n\n" +
            "### Instruction:\n" +
            "Based on the above hint, what kind of textual tasks can you develop that can be applied to the input?\n\n" +
            "### Format Examples (Imitate their formats, ignore the contents):\n" +
            self.demonstrations + "\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Task 1:\n"
        )

    
class ConversationPromptTask_6(ConversationPrompt):
    ''' shifting the attribute to generate task instructions, also add 3-shot in-domain demonstrations '''
    def __init__(self):
        super().__init__()
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" +
            "2. You'll need to look at the attribute as a direction to guide your thinking. \n" +
            "3. Each task must be indivisible (one task, one intention).\n" +
            "4. Please prioritize your most confident predictions.\n" +
            "5. Avoid tasks requiring additional context, i.e., tasks must be easily solved/answered using only the input.\n" +
            "6. Your tasks should sufficiently describe how this input is expected to be mapped to an output text, i.e., elaborating the tasks in detail.\n" +
            "7. But do not disclose the final answer!\n"
        )
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n\n" +
            "### In a word, you should first describe what the input is, and what textual attribute it has, then elaborate on the task intent, and finally exactly describe what kind of output you expect and mention any necessary output constraints (e.g., formats, options).\n"
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Attribute:\n" +
            "{hint}\n\n" +
            "### Instruction:\n" +
            "Based on the above information, what kind of textual tasks can you develop that can shift the input's attribute?\n\n" +
            "### Format Examples (Imitate their formats, ignore the contents):\n" +
            self.demonstrations + "\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "Task 1:\n"
        )           

        
class ConversationPromptTask_3(ConversationPrompt):
    ''' following the hint to brainstorm novel task instructions, also add 3-shot in-domain demonstrations '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a creative prompt engineer."
        )
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" +
            "2. Please prioritize your most confident predictions.\n" +
            "3. Avoid generating tasks that require accessing external tools or APIs.\n" +
            "4. Ensure your textual tasks are detailed and sufficient, e.g., including any necessary contexts beyond the given input.\n" +
            "5. Do not disclose the output (answer) of your tasks.\n"
        )
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n\n" +
            "### In a word, you should first describe what the input is, and what textual attribute it has, then elaborate on the task intent, and finally exactly describe what kind of output you expect and mention any necessary output constraints (e.g., formats, options).\n"
        )
        self.query_prompt = (
            "You are a creative prompt engineer responsible for designing textual tasks to test the readers' problem-solving ability.\n" +
            "A 'textual task' can be simply understood as a 'mapping function', which defines how to process an 'input' (source text) into a certain 'output' (target text).\n\n" +
            "### Input (waits to be processed):\n" +
            "{input}\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "### Hint:\n" +
            "{hint}\n\n" +
            "What kind of textual tasks can you develop to process the input?\n\n" +
            "Task 1: "
        )

    
class ConversationPromptTask_4(ConversationPrompt):
    ''' shifting the attribute to generate task instructions, also add 3-shot in-domain demonstrations '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a creative prompt engineer."
        )
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" +
            "2. Please prioritize your most confident predictions.\n" +
            "3. Avoid generating tasks that require accessing external tools or APIs.\n" +
            "4. Ensure your textual tasks are detailed and sufficient, e.g., including any necessary contexts beyond the given input.\n" +
            "5. Do not disclose the output (answer) of your tasks.\n"
        )
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n\n" +
            "### In a word, you should first describe what the input is, and what textual attribute it has, then elaborate on the task intent, and finally exactly describe what kind of output you expect and mention any necessary output constraints (e.g., formats, options).\n"
        )
        self.query_prompt = (
            "You are a creative prompt engineer responsible for designing textual tasks to test the readers' knowledge.\n" +
            "A 'textual task' can be simply understood as a 'mapping function', which defines how to process an 'input' (source text) into a certain 'output' (target text).\n\n" +
            "### Input (waits to be processed):\n" +
            "{input}\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "### Attribute:\n" +            
            "{hint}\n\n" +
            "Your tasks should also try to shift the above attribute of the input. So, what kind of textual tasks can you develop?\n\n" +
            "Task 1: "
        )   


# used for generating classification task instructions
class ConversationPromptTask_CLS(ConversationPrompt):
    ''' use 3-shot demonstrations '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Natural Language Processing (NLP)."
        )
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = ()
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n"
        )
        self.query_prompt = (
            "You are an expert in Natural Language Processing (NLP). I want you to design a high-quality classification task based on the given information.\n" +
            "A good classification task should clearly include any necessary information, e.g., all the possible answer options (in random order) and how to decide the category of a given input. If the input has already included the answer options, you don't have to repeat them.\n\n" +
            "### Input:\n" +
            "{input}\n\n" +
            "### Attribute:\n" +
            "{hint}\n\n" +
            "Using the attribute as a clue, what classification task can you develop for this input?\n\n" +
            "### Example Tasks (your task should imitate their formats):\n" +
            self.demonstrations + "\n" +
            "### Your task (do not disclose the answer):\n"
        )   
        
    def extract_content(self, content:str):
        content = content.strip()
        return [content]  # return a list to ensure the consistency with other templates
    

# used for generating classification task instructions
class ConversationPromptTask_CLS_2(ConversationPrompt):
    ''' use 3-shot demonstrations (slightly tune the prompt) '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Natural Language Processing (NLP)."
        )
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = ()
        self.demonstrations = (
            "1. {example_1}\n" +
            "2. {example_2}\n" +
            "3. {example_3}\n"
        )
        self.query_prompt = (
            "You are an expert in Natural Language Processing (NLP). I want you to design a high-quality classification task based on the given information.\n" +
            "A good classification task should clearly include any necessary information, e.g., all the possible answer options and how to decide the category of a given input. If the input has already included the answer options, you don't have to repeat them.\n\n" +
            "### Input:\n" +
            "{input}\n\n" +
            "### Attribute:\n" +
            "{hint}\n\n" +
            "Using the attribute as a clue (just use it to guide your thinking but do not mention it in your task), what classification task can you develop for this input?\n\n" +
            "### Example Tasks (your task should imitate their formats):\n" +
            self.demonstrations + "\n" +
            "### Your task (do not disclose the answer):\n"
        )   
        
    def extract_content(self, content:str):
        content = content.strip()
        return [content]  # return a list to ensure the consistency with other templates    


# used for annotating the output
class ConversationPromptAnswer(ConversationPrompt):
    ''' generate the answer by following the instruction '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a helpful assistant. " +
            "Your responsibility is to follow the user's instruction and generate the output."
        )
        # don't need to filter the keywords when annotating the output
        self.blacklist = [
        ]
        
        self.requirement_prompt = (
            "1. Conclude your final output without reasoning.\n" +
            "2. If you think it is impossible to answer the instruction by giving the existing information (e.g., require additional context, not a textual task), simply generate 'None'.\n"
        )
        self.query_prompt = (
            # "### Input:\n" + 
            # "{input}\n\n" +
            # "### Instruction:\n" +
            # "{instruction}\n\n" +
            # "### Requirements:\n" +
            # self.requirement_prompt + "\n" +
            # "### Output:\n"
            "Given a task instruction and an input, please generate the output (answer) according to the requirements mentioned in the instruction.\n" +
            "If you cannot answer the instruction base on the given information, simply generate 'None'.\n\n" +
            "### Instruction:\n" +
            "{instruction}\n\n" +
            "### Input:\n" + 
            "{input}\n\n" +
            "### Output:\n"
        )

    def extract_content(self, content:str):
        # Remove the "None" at the end
        # content = re.sub(r"(None\.|None|none\.|none)", "", content) 
        content = re.sub(r"(None\.?|none\.?)+$", "", content)
        content = content.strip()
         
        return content
        
# used for annotating the output (the original version, that asks the chatgpt to not reasoning)
class ConversationPromptAnswer_2(ConversationPrompt):
    ''' generate the answer by following the instruction '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a helpful assistant. " +
            "Your responsibility is to follow the user's instruction and generate the output."
        )
        # don't need to filter the keywords when annotating the output
        self.blacklist = [
        ]
        
        self.requirement_prompt = (
            "1. Conclude your final output without reasoning.\n" +
            "2. If you think it is impossible to answer the instruction by giving the existing information (e.g., require additional context, not a textual task), simply generate 'None'.\n"
        )
        self.query_prompt = (
            "### Input:\n" + 
            "{input}\n\n" +
            "### Instruction:\n" +
            "{instruction}\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "### Output:\n"
        )

    def extract_content(self, content:str):
        # Remove the "None" at the end
        # content = re.sub(r"(None\.|None|none\.|none)", "", content) 
        content = re.sub(r"(None\.?|none\.?)+$", "", content)
        content = content.strip()
         
        return content


# the prompt used in evaluating GPT performances on eval benchmarks (baseline prompt)
class ConversationPromptAnswer_GPT_EVAL(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a helpful assistant. " +
            "Your responsibility is to follow the user's instruction and generate the output."
        )
        # don't need to filter the keywords when annotating the output
        self.blacklist = [
        ]
        self.query_prompt = (
            "### Instruction:\n" +
            "{instruction}\n\n" +
            "### Input:\n" + 
            "{input}\n\n" +
            "### Output:\n"
        )

    def extract_content(self, content:str):
        # Remove the "None" at the end
        # content = re.sub(r"(None\.|None|none\.|none)", "", content) 
        content = re.sub(r"(None\.?|none\.?)+$", "", content)
        content = content.strip()
         
        return content

        
# used for adding output constraints
class ConversationPromptConstraint(ConversationPrompt):
    ''' summarize an output constraint from the instruction '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a helpful assistant. "
        )
        
        self.demonstrations_num = 8
        self.query_prompt = (
            "Read and summarize the output constraints from the given instruction. Generate 'None', if there are no clear constraints required by the instruction.\n\n" + 
            "### Example 1:\n" +
            "Instruction: {instruction_1}\n" +
            "Constraints: {constraint_1}\n\n" +
            "### Example 2:\n" +
            "Instruction: {instruction_2}\n" +
            "Constraints: {constraint_2}\n\n" +
            "### Example 3:\n" +
            "Instruction: {instruction_3}\n" +
            "Constraints: {constraint_3}\n\n" +
            "### Example 4:\n" +
            "Instruction: {instruction_4}\n" +
            "Constraints: {constraint_4}\n\n" +
            "### Example 5:\n" +
            "Instruction: {instruction_5}\n" +
            "Constraints: {constraint_5}\n\n" +
            "### Example 6:\n" +
            "Instruction: {instruction_6}\n" +
            "Constraints: {constraint_6}\n\n" +
            "### Example 7:\n" +
            "Instruction: {instruction_7}\n" +
            "Constraints: {constraint_7}\n\n" +
            "### Example 8:\n" +
            "Instruction: {instruction_8}\n" +
            "Constraints: {constraint_8}\n\n" +
            "### Example 9:\n" +
            "Instruction: {target_instruction}\n" +
            "Constraints: "
        )   
        
    def extract_content(self, content:str):
        # just simply return back the content
        content = content.strip()
         
        return content

class ConversationPromptConstraint_2(ConversationPrompt):
    ''' also add Input to the query prompt, since some of the constraints are coming from the input '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are a helpful assistant. "
        )
        
        self.demonstrations_num = 8
        self.query_prompt = (
            "Read the given task instruction and input, and summarize output constraints for this task. Generate 'None', if there are no clear constraints required by this task.\n\n" + 
            "### Example 1:\n" +
            "- Instruction: {instruction_1}\n" +
            "- Input: {input_1}\n" +
            "- Constraints: {constraint_1}\n\n" +
            "### Example 2:\n" +
            "- Instruction: {instruction_2}\n" +
            "- Input: {input_2}\n" +
            "- Constraints: {constraint_2}\n\n" +
            "### Example 3:\n" +
            "- Instruction: {instruction_3}\n" +
            "- Input: {input_3}\n" +
            "- Constraints: {constraint_3}\n\n" +
            "### Example 4:\n" +
            "- Instruction: {instruction_4}\n" +
            "- Input: {input_4}\n" +
            "- Constraints: {constraint_4}\n\n" +
            "### Example 5:\n" +
            "- Instruction: {instruction_5}\n" +
            "- Input: {input_5}\n" +
            "- Constraints: {constraint_5}\n\n" +
            "### Example 6:\n" +
            "- Instruction: {instruction_6}\n" +
            "- Input: {input_6}\n" +
            "- Constraints: {constraint_6}\n\n" +
            "### Example 7:\n" +
            "- Instruction: {instruction_7}\n" +
            "- Input: {input_7}\n" +
            "- Constraints: {constraint_7}\n\n" +
            "### Example 8:\n" +
            "- Instruction: {instruction_8}\n" +
            "- Input: {input_8}\n" +
            "- Constraints: {constraint_8}\n\n" +
            "### Example 9:\n" +
            "- Instruction: {target_instruction}\n" +
            "- Input: {target_input}\n" +
            "- Constraints: "
        )   
        
    def extract_content(self, content:str):
        # just simply return back the content
        content = content.strip()
         
        return content
    

# used for generating wrong output candidates, to expand more classifcation options
class ConversationPromptWrongOutputs(ConversationPrompt):
    ''' develop more outputs that are worse than the given output '''
    def __init__(self):
        super().__init__()
        self.seperator = r"Wrong Output \d+:" ## Wrong Output 1: Wrong Output 2: .. Wrong Output n:
        self.system = (
            "You are a helpful assistant. "
        )
        
        self.requirement_prompt = (
            "1. The output candidates you generate should be worse than the given correct output, e.g., wrong or imperfect answers.\n" +
            "2. You are encouraged to generate some challenging output candidates, that are close to the correct output but not the most desired one (i.e., containing certain errors).\n" +
            "3. You are encouraged to generate as many output candidates as possible; If you think there are no more suitable output candidates, end up with 'None'.\n"
        )
        self.query_prompt = (
            "Given a task, a task input, and a corresponding correct output, generate more output candidates for this task.\n\n" +
            "### Requirements:\n" +
            self.requirement_prompt + "\n" +
            "### Task:\n" +
            "{instruction}\n\n" +
            "### Input:\n" + 
            "{input}\n\n" +
            "### Output:\n" + 
            "{output}\n\n" +
            "Wrong Output 1:\n"
        )
        
    def extract_content(self, content:str):
        # Remove the "None" at the end
        # content = re.sub(r"(None\.|None|none\.|none)", "", content) 
        content = re.sub(r"(None\.?|none\.?)+$", "", content)
        items = re.split(self.seperator, content)
        # Remove any empty elements from the list
        items = [item.strip() for item in items if item.strip() != ""]
        
        return items
    

# used for classify whether an exsiting instruction can be applied to a given input
class ClassificationValidationPrompt(ConversationPrompt):
    ''' judge valid instruction or not (tested on gpt-4 and gpt-3.5-turbo-0613) '''
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Natural Language Processing (NLP) tasks."
        )
        self.query_prompt = (
            "You are an expert in Natural Language Processing (NLP) tasks.\n" +
            "Given a task description and a piece of text, your job is to determine whether this text can be used as input for this task.\n" +
            "If the text satisfies the input expectation of this task, answer 'Yes'; otherwise, if the text doesn't match the input description, answer 'No'.\n\n" +
            "### Text:\n" +
            "{input}\n\n" +
            "### Task Description:\n" +
            "{instruction}\n\n" +
            "### Your Answer:\n"
        )
        
    def extract_content(self, content:str):
        # just simply return back the content
        content = content.strip()
        content = re.sub(r'[^\w\s]', '', content) # remove all punctuations
        content = content.lower() # convert to lower case
         
        return content


if __name__ == "__main__":
    # prompt = ConversationPromptAttribute()
    # test_input = {"input": "This is a test input."}
    # print(prompt.query_prompt.format_map(test_input))
    # test_content = "Identify the total number of characters in the input.\n\n" + \
    #     "Attribute 2:\n" + \
    #     "Determine the number of uppercase letters in the input.\n\n" + \
    #     "Attribute 3:\n" + \
    #     "Count the number of lowercase letters in the input.\n\n" + \
    #     "Attribute 17:\n" + \
    #     "Count the number of characters that are neither vowels nor consonants (i.e., digits and special characters) in the input.\n\n" + \
    #     "None."
    # print(test_content)
    # print("\n\nExtracted attributes:")
    # print(prompt.extract_content(test_content))
    
    # test_content = "Identify the total number of characters in the input.\n\n" + \
    #     "Task 2:\n" + \
    #     "Determine the number of uppercase letters (such as 'None') in the input.\n\n" + \
    #     "Task 3:\n" + \
    #     "Count the number of lowercase letters in the input.\n\n" + \
    #     "Task 17:\n" + \
    #     "Count the number of characters that are neither vowels nor consonants (i.e., digits and special characters) in the input, e.g., none.\n\n" + \
    #     "None."
    # test_content = "None."
    
    # prompt = ConversationPromptTask()
    # prompt = ConversationPromptTask_2()
    # print(prompt.extract_content(test_content))
    
    # test_content = "Person A: Well that's sad. It might've been funny if it was fake.\n" + \
    #                 "Person B: Oh yes, because nothing brings joy to the world like a good fake tragedy. Maybe next time we can all gather around and laugh at a staged car accident.\n" + \
    #                 "Person B: Oh, my apologies. I thought you were suggesting we should all be entertained by the misfortunes of others. Silly me.\n" + \
    #                 "None."
                    
    # prompt = ConversationPromptAnswer()
    # test_input = {"input": "This is a test input.", "instruction": "This is a test instruction."}
    # print(prompt.query_prompt.format_map(test_input))
    # print(prompt.extract_content(test_content))
    
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptTask_3()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint.", "example_1": "This is a test example 1.", "example_2": "This is a test example 2.", "example_3": "This is a test example 3."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # print(10*"="+"\n\n")
    
    # prompt = ConversationPromptTask_4()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint.", "example_1": "This is a test example 1.", "example_2": "This is a test example 2.", "example_3": "This is a test example 3."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # print(10*"="+"\n\n")
    # prompt = ConversationPromptTask_CLS()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint.", "example_1": "This is a test example 1.", "example_2": "This is a test example 2.", "example_3": "This is a test example 3."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ClassificationValidationPrompt()
    # test_input = {"input": "This is a test input.", "instruction": "This is a test instruction."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptAttribute_2()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptTask_5()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint.", "example_1": "This is a test example 1.", "example_2": "This is a test example 2.", "example_3": "This is a test example 3."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptTask_6()
    # test_input = {"input": "This is a test input.", "hint": "This is a test hint.", "example_1": "This is a test example 1.", "example_2": "This is a test example 2.", "example_3": "This is a test example 3."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptConstraint()
    # test_input = {"instruction_1": "This is a test instruction 1.", "constraint_1": "This is a test constraint 1.",
    #               "instruction_2": "This is a test instruction 2.", "constraint_2": "This is a test constraint 2.",
    #               "instruction_3": "This is a test instruction 3.", "constraint_3": "This is a test constraint 3.",
    #               "instruction_4": "This is a test instruction 4.", "constraint_4": "This is a test constraint 4.",
    #               "instruction_5": "This is a test instruction 5.", "constraint_5": "This is a test constraint 5.",
    #               "instruction_6": "This is a test instruction 6.", "constraint_6": "This is a test constraint 6.",
    #               "instruction_7": "This is a test instruction 7.", "constraint_7": "This is a test constraint 7.",
    #               "instruction_8": "This is a test instruction 8.", "constraint_8": "This is a test constraint 8.",
    #               "target_instruction": "This is a test target instruction."}
    # print(prompt.query_prompt.format_map(test_input))
    
    # prompt = ConversationPromptConstraint_2()
    # test_input = {"instruction_1": "This is a test instruction 1.", "constraint_1": "This is a test constraint 1.", "input_1": "This is a test input 1.",
    #               "instruction_2": "This is a test instruction 2.", "constraint_2": "This is a test constraint 2.", "input_2": "This is a test input 2.",
    #               "instruction_3": "This is a test instruction 3.", "constraint_3": "This is a test constraint 3.", "input_3": "This is a test input 3.",
    #               "instruction_4": "This is a test instruction 4.", "constraint_4": "This is a test constraint 4.", "input_4": "This is a test input 4.",
    #               "instruction_5": "This is a test instruction 5.", "constraint_5": "This is a test constraint 5.", "input_5": "This is a test input 5.",
    #               "instruction_6": "This is a test instruction 6.", "constraint_6": "This is a test constraint 6.", "input_6": "This is a test input 6.",
    #               "instruction_7": "This is a test instruction 7.", "constraint_7": "This is a test constraint 7.", "input_7": "This is a test input 7.",
    #               "instruction_8": "This is a test instruction 8.", "constraint_8": "This is a test constraint 8.", "input_8": "This is a test input 8.",
    #               "target_instruction": "This is a test target instruction.", "target_input": "This is a test target input."}
    # print(prompt.query_prompt.format_map(test_input))
    
    prompt = ConversationPromptWrongOutputs()
    test_input = {"instruction": "This is a test instruction.", "input": "This is a test input.", "output": "This is a test output."}
    print(prompt.query_prompt.format_map(test_input))
    
    # test_content = "test content 1\n\n" + \
    #             "Wrong Output 2:\n" + \
    #             "test content 2\n\n" + \
    #             "Wrong Output 3:\n" + \
    #             "test content 3\n\n" + \
    #             "None."
    # test_content = "frience." + \
    #     "None."
    # print(prompt.extract_content(test_content))
    

