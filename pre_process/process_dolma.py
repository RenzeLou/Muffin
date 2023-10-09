import argparse
import json
import math
import random
import os
os.environ['NLTK_DATA'] = '/scratch/rml6079/nltk_data'
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

import jsonlines
from tqdm import tqdm

blacklist = [
            # file suffix
            "file","files",
            # audio and vedio suffix
            "video","audio",".mp3",".wav",".mp4",".avi",".mov",".flv",".swf",".mkv",".wmv",".rmvb",".rm",
            # picture suffix
            "image","images","picture","pictures",".jpg",".jpeg",".png",".gif",".svg",".bmp",".tiff",".psd",".raw",".heif",".indd",".jpeg2000",".webp",".ai",
        ]

def select_data(files, path=None, length_threshold=500, do_length_filter=True):
    data = []
    uni_text = set()
    for file in files:
        file = os.path.join(path, file) if path is not None else file
        with jsonlines.open(file) as reader:
            # filter the reader: 
            # 1. if obj["text"] is an empty string, then ignore it
            # 2. if obj["text"] contains those information except for text (e.g., picture, music, etc.), then ignore it
            # 3. all the "text" are unique
            # 4. (optional) if obj["text"] is a very long string, then ignore it (not applicable for code data)
            if do_length_filter:
                reader = filter(lambda obj: obj["text"].strip() != "" and not any([item in obj["text"] for item in blacklist]) and len(obj["text"].split()) <= length_threshold, reader)
            else:
                reader = filter(lambda obj: obj["text"].strip() != "" and not any([item in obj["text"] for item in blacklist]), reader)
            for obj in tqdm(reader, desc="read {}".format(file)):
                if obj["text"] not in uni_text:
                    # ensure the texts are unique
                    uni_text.add(obj["text"])
                    data.append(obj)
        
    return data

def FindAllSuffix(task_path,file_substring="json"):
    # return a list that contains all the files with the given suffix
    result = []
    for root, dirs, files in os.walk(task_path):
        for file in files:
            if file.endswith(file_substring):
                result.append(os.path.join(root, file))
            
    return result


def control_length_balance(data):
    # use nltk to get the word/sentence length
    for item in tqdm(data):
        text = item["text"]
        word_length = len(word_tokenize(text))
        sent_length = len(sent_tokenize(text))
        item.update({"word_length": word_length, "sent_length": sent_length})
    # select the texts with diverse length
    # 1. count the frequency of each word length
    word_length2items = {}
    for item in data:
        word_length, sent_length = item["word_length"], item["sent_length"]
        if word_length not in word_length2items:
            word_length2items[word_length] = [item]
        else:
            word_length2items[word_length].append(item)
    # 2. get the avg frequency
    avg_word_freq = sum([len(items) for items in word_length2items.values()]) / len(word_length2items)
    # 3. filtering
    for word_length, items in word_length2items.items():
        if len(items) > avg_word_freq:
            # only remain avg frequency num items
            word_length2items[word_length] = random.sample(items, math.ceil(avg_word_freq))
    
    data_2 = []
    for items in word_length2items.values():
        data_2.extend(items)
    print("the avg frequency of each word length is {}; remain {} texts after promoting the length diversity".format(avg_word_freq, len(data_2)))
    
    return data_2

def combined_data_process(combined_data):
    processed_combined_data = []
    for obj in tqdm(combined_data, total=len(combined_data)):
        id = obj["source"] + "_" + obj["id"]
        text = obj["text"].strip()
        processed_combined_data.append({
            "id": id,
            "instruction": "",  # wait for annotation
            "input": text,
            "output": [] # wait for annotation
        })
    
    return processed_combined_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/web_text")
    parser.add_argument("--cc_files", type=str, nargs="+",
                        default=["cc_en_head-0000.json", "cc_en_middle-0109.json", "cc_en_tail-0001.json"])
    parser.add_argument("--c4_files", type=str, nargs="+",
                        default=["c4-0085.json"])
    parser.add_argument("--book_files", type=str, nargs="+",
                        default=["books-0002.json"])
    parser.add_argument("--wiki_files", type=str, nargs="+",
                        default=["en_simple_wiki-0001.json"])
    parser.add_argument("--s2_files", type=str, nargs="+",
                        default=["s2_v3-0000.json"])
    parser.add_argument("--stack_path", type=str, default="stack_code")
    parser.add_argument("--select_num", type=int, default=5000)
    parser.add_argument("--save_path", type=str, default="./data/web_text/select") # same as `--path` by default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_threshold", type=int, default=300)  # how many unique words in the text
    parser.add_argument("--overwrite", action="store_true", help="overwrite the save file if it exists.")
    parser.add_argument("--length_diversity", action="store_true", help="whether to select texts with different lengths.")
    parser.add_argument("--num_per_part", type=int, default=None, help="split the annotation parts due to so poor; each part has `num_per_part` texts to annotate.")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    # ratio ==> 0.3:0.2:0.2:0.2:0.1 == cc:c4:s2:wiki:stack
    cc_num = math.ceil(args.select_num * 0.3)
    c4_num = math.ceil(args.select_num * 0.2)
    # book_num = math.ceil(args.select_num * 0.2)
    s2_num = math.ceil(args.select_num * 0.2)
    stack_num = math.ceil(args.select_num * 0.1)
    wiki_num = math.ceil(args.select_num * 0.2)
    
    print("==> try to select {} texts from dolma".format(args.select_num))
    print("==> Note that you set the ratio as ``cc: c4: s2: wiki: stack = 3: 2: 2: 2: 1``")
    
    save_path = args.save_path + "_{}".format(args.select_num)
    os.makedirs(save_path, exist_ok=True)
    cc_save_file = os.path.join(save_path, "cc.json")
    c4_save_file = os.path.join(save_path, "c4.json")
    # book_save_file = os.path.join(save_path, "book.json")
    s2_save_file = os.path.join(save_path, "s2.json")
    wiki_save_file = os.path.join(save_path, "wiki.json")
    stack_save_file = os.path.join(save_path, "stack.json")
    combined_data_save_file = os.path.join(save_path, "texts_from_dolma.json")
    
    
    # read, process and select text from cc
    if os.path.exists(cc_save_file) and not args.overwrite:
        print("{} exists, simply load it.".format(cc_save_file))
        with open(cc_save_file, "r") as f:
            selected_cc_data = json.load(f)
    else:
        cc_data = select_data(args.cc_files, args.path, args.length_threshold)
        cc_num = min(cc_num, len(cc_data))
        print("totally {} texts from Common Crawl, now randomly select {} texts".format(len(cc_data), cc_num))
        selected_cc_data = random.sample(cc_data, cc_num)
        with open(cc_save_file, "w") as f:
            json.dump(selected_cc_data, f, indent=2)
            
    
    # read, process and select text from c4
    if os.path.exists(c4_save_file) and not args.overwrite:
        print("{} exists, simply load it.".format(c4_save_file))
        with open(c4_save_file, "r") as f:
            selected_c4_data = json.load(f)
    else:
        c4_data = select_data(args.c4_files, args.path, args.length_threshold)
        c4_num = min(c4_num, len(c4_data))
        print("totally {} texts from C4, now randomly select {} texts".format(len(c4_data), c4_num))
        selected_c4_data = random.sample(c4_data, c4_num)
        with open(c4_save_file, "w") as f:
            json.dump(selected_c4_data, f, indent=2)
            
    
    # TODO: ignore book data, because most of text are really long
    selected_book_data = []
    # read, process and select text from book
    # if os.path.exists(book_save_file) and not args.overwrite:
    #     print("{} exists, simply load it.".format(book_save_file))
    #     with open(book_save_file, "r") as f:
    #         selected_book_data = json.load(f)
    # else:
    #     book_data = select_data(args.book_files, args.path, args.length_threshold)
    #     book_num = min(book_num, len(book_data))
    #     print("totally {} texts from BookCorpus, now randomly select {} texts".format(len(book_data), book_num))
    #     selected_book_data = random.sample(book_data, book_num)
    #     with open(book_save_file, "w") as f:
    #         json.dump(selected_book_data, f, indent=2)
            
    
    # read, process and select text from s2
    if os.path.exists(s2_save_file) and not args.overwrite:
        print("{} exists, simply load it.".format(s2_save_file))
        with open(s2_save_file, "r") as f:
            selected_s2_data = json.load(f)
    else:
        s2_data = select_data(args.s2_files, args.path, args.length_threshold)
        s2_num = min(s2_num, len(s2_data))
        print("totally {} texts from Semantic Scholar, now randomly select {} texts".format(len(s2_data), s2_num))
        selected_s2_data = random.sample(s2_data, s2_num)
        with open(s2_save_file, "w") as f:
            json.dump(selected_s2_data, f, indent=2)
            
    
    # read, process and select text from wiki
    if os.path.exists(wiki_save_file) and not args.overwrite:
        print("{} exists, simply load it.".format(wiki_save_file))
        with open(wiki_save_file, "r") as f:
            selected_wiki_data = json.load(f)
    else:
        wiki_data = select_data(args.wiki_files, args.path, args.length_threshold)
        wiki_num = min(wiki_num, len(wiki_data))
        print("totally {} texts from Wikipedia, now randomly select {} texts".format(len(wiki_data), wiki_num))
        selected_wiki_data = random.sample(wiki_data, wiki_num)
        with open(wiki_save_file, "w") as f:
            json.dump(selected_wiki_data, f, indent=2)
            
    
    # read, process and select text from stack
    # get all the .json files under the `stack_path`
    if os.path.exists(stack_save_file) and not args.overwrite:
        print("{} exists, simply load it.".format(stack_save_file))
        with open(stack_save_file, "r") as f:
            selected_stack_data = json.load(f)
    else:
        stack_path = os.path.join(args.path, args.stack_path)
        stack_files = FindAllSuffix(stack_path, file_substring="json")
        stack_data = select_data(stack_files, path=None, length_threshold=args.length_threshold)
        statck_num = min(stack_num, len(stack_data))
        print("totally {} texts from Stack Overflow, now randomly select {} texts".format(len(stack_data), stack_num))
        selected_stack_data = random.sample(stack_data, stack_num)
        with open(stack_save_file, "w") as f:
            json.dump(selected_stack_data, f, indent=2)
        
        
    # combine all the selected data
    combined_data = selected_cc_data + selected_c4_data + selected_book_data + selected_s2_data + selected_wiki_data + selected_stack_data
    # process the combined data (used for the further annotation)
    print("\n==> now totally {} texts from dolma, process it...".format(len(combined_data)))
    processed_combined_data = combined_data_process(combined_data)
    # save the combined data
    if os.path.exists(combined_data_save_file) and not args.overwrite:
        print("==> {} exists, skip saving the combined data, please set `--overwrite` to overwrite it".format(combined_data_save_file))
    else:
        with open(combined_data_save_file, "w") as f:
            json.dump(processed_combined_data, f, indent=2)
        print("==> save the combined data to {}".format(combined_data_save_file))
        
    
    # process the final texts again, del some texts, to ensure the length diversity
    if args.length_diversity:
        print("\n==> now process the final texts again, to promote the length diversity")
        
        print("process cc data...")
        selected_cc_data_new = control_length_balance(selected_cc_data)
        print("process c4 data...")
        selected_c4_data_new = control_length_balance(selected_c4_data)
        # print("process book data...")
        # selected_book_data_new = control_length_balance(selected_book_data)
        print("process s2 data...")
        selected_s2_data_new = control_length_balance(selected_s2_data)
        print("process wiki data...")
        selected_wiki_data_new = control_length_balance(selected_wiki_data)
        # print("process stack data...")
        # selected_stack_data_new = control_length_balance(selected_stack_data)
        selected_stack_data_new = selected_stack_data  # don't need to control the length of code data
        
        # save all these length-controled data 
        save_path_new = os.path.join(save_path, "length_control")
        os.makedirs(save_path_new, exist_ok=True)
        cc_save_file_new = os.path.join(save_path_new, os.path.basename(cc_save_file))
        c4_save_file_new = os.path.join(save_path_new, os.path.basename(c4_save_file))
        # book_save_file_new = os.path.join(save_path_new, os.path.basename(book_save_file))
        s2_save_file_new = os.path.join(save_path_new, os.path.basename(s2_save_file))
        wiki_save_file_new = os.path.join(save_path_new, os.path.basename(wiki_save_file))
        stack_save_file_new = os.path.join(save_path_new, os.path.basename(stack_save_file))
        
        # save all the controled files (no need to judge overwrite here, as all these processing are fast)
        with open(cc_save_file_new, "w") as f:
            json.dump(selected_cc_data_new, f, indent=2)
        with open(c4_save_file_new, "w") as f:
            json.dump(selected_c4_data_new, f, indent=2)
        # with open(book_save_file_new, "w") as f:
        #     json.dump(selected_book_data_new, f, indent=2)
        with open(s2_save_file_new, "w") as f:
            json.dump(selected_s2_data_new, f, indent=2)
        with open(wiki_save_file_new, "w") as f:
            json.dump(selected_wiki_data_new, f, indent=2)
        with open(stack_save_file_new, "w") as f:
            json.dump(selected_stack_data_new, f, indent=2)
            
        
        combined_data_new = selected_cc_data_new + selected_c4_data_new + selected_s2_data_new + selected_wiki_data_new + selected_stack_data_new
        print("\n==> {} texts after length control".format(len(combined_data_new)))
        process_combined_data_new = combined_data_process(combined_data_new)
        combined_data_save_file_new = os.path.join(save_path_new, os.path.basename(combined_data_save_file))
        combined_data_save_file_new = combined_data_save_file_new.replace(".json", ".{}.json".format(len(combined_data_new)))
        with open(combined_data_save_file_new, "w") as f:
            json.dump(process_combined_data_new, f, indent=2)
            
        processed_combined_data = process_combined_data_new  # used for further splitting
    
    # split the data into several parts, that can be annotated seperately
    # by default, each part has 750 texts to annotate
    if args.num_per_part is not None:
        # shuffle the data
        random.shuffle(processed_combined_data)
        part_save_path = save_path if not args.length_diversity else save_path_new
        print("\n==> now split the data into several parts, each part has {} texts to annotate".format(args.num_per_part))
        num_parts = math.ceil(len(processed_combined_data) / args.num_per_part)
        for i in range(num_parts):
            part_data = processed_combined_data[i*args.num_per_part: (i+1)*args.num_per_part]
            part_save_file = os.path.join(part_save_path, "part_{}.json".format(i))
            with open(part_save_file, "w") as f:
                json.dump(part_data, f, indent=2)
            print("==> save part {} to {}".format(i, part_save_file))
        
        print("==> totally {} parts".format(num_parts))
        
        
        
            
    
if __name__ == "__main__":
    main()