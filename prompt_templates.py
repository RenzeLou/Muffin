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
    

# used for generating textual attributes 
class ConversationPromptAttribute(ConversationPrompt):
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
        
        
class ConversationPromptTask_3(ConversationPrompt):
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

    
class ConversationPromptTask_4(ConversationPrompt):
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

if __name__ == "__main__":
    prompt = ConversationPromptAttribute()
    test_input = {"input": "This is a test input."}
    print(prompt.query_prompt.format_map(test_input))
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
    
    test_content = "Identify the total number of characters in the input.\n\n" + \
        "Task 2:\n" + \
        "Determine the number of uppercase letters (such as 'None') in the input.\n\n" + \
        "Task 3:\n" + \
        "Count the number of lowercase letters in the input.\n\n" + \
        "Task 17:\n" + \
        "Count the number of characters that are neither vowels nor consonants (i.e., digits and special characters) in the input, e.g., none.\n\n" + \
        "None."
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
    
    # prompt = ConversationPromptTask_4()
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
    

