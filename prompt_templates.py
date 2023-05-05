
import re

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
            "music",
            "flowchart",
            "diagram",
        ]
        

    def extract_content(self, content:str):
        content = re.sub(r"(None\.|None|none\.|none)", "", content) # Remove the "None" at the end
        attributes = re.split(self.seperator, content)
        # Remove any empty elements from the list
        attributes = [attr.strip() for attr in attributes if attr.strip() != ""]
        # Remove any elements that has words in the blacklist
        attributes = [attr for attr in attributes if not any([word in attr for word in self.blacklist])] 
        return attributes
    


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
        
        
class ConversationPromptTask(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.seperator = r"Task \d+:" ## Task 1: Task 2: .. Task n:
        self.requirement_prompt = (
            "1. Brainstorm as many textual tasks as possible. If you think there are no more suitable tasks, end up with 'None'.\n" + 
            "2. You'll need to look at the hint as a direction to guide your thinking. Your responses should strictly be based on the hint!\n" +
            "3. Each task must be indivisible (one task, one intention).\n" +
            "4. Avoid tasks requiring additional context, i.e., tasks must be easily solved/answered using only the input.\n" +
            "5. Please prioritize your most confident predictions.\n"
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

        
if __name__ == "__main__":
    # prompt = ConversationPromptAttribute()
    # print(prompt.att_prompt.format_map({"input": "This is a test input."}))
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
        "Determine the number of uppercase letters in the input.\n\n" + \
        "Task 3:\n" + \
        "Count the number of lowercase letters in the input.\n\n" + \
        "Task 17:\n" + \
        "Count the number of characters that are neither vowels nor consonants (i.e., digits and special characters) in the input.\n\n" + \
        "None."
    
    prompt = ConversationPromptTask()
    print(prompt.extract_content(test_content))
    

