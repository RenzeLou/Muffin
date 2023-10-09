import argparse
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


def plot_bar_chart(results:dict, save_file:str):
    # Input percentages
    percentages = list(results.values())

    # Corresponding names
    names = list(results.keys())

    # Color for each name
    colors = ['red', 'green', 'blue', 'violet']

    # Create a new figure
    plt.figure(figsize=[10, 8])

    # Create a bar plot
    bars = plt.bar(names, percentages, color=colors)

    # Add the percentage on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')

    # Add a legend
    plt.legend(bars, names, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    # Make the plot tighter
    plt.tight_layout()
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./utils/data_quality/annotated_data_200.json")
    parser.add_argument("--save_path", type=str, default="./utils/data_quality") # same as `--path` by default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled", action="store_true", help="if set to True, means the input data is already sampled, there is no need to do the further sampling")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    with open(args.data_path, "r") as f:
        data = json.load(f)
        
    a1_num, a2_num, a3_num = 0, 0, 0
    all_num = 0
    for item in tqdm(data):
        a1 = item["A1"]
        a2 = item["A2"]
        a3 = item["A3"]
        a1_num += 1 if a1 == "y" else 0
        a2_num += 1 if a2 == "y" else 0
        a3_num += 1 if a3 == "y" else 0
        all_num += 1 if a1 == "y" and a2 == "y" and a3 == "y" else 0
    
    print("A1: {} / {} = {:.4f}%".format(a1_num, len(data), a1_num / len(data) * 100))
    print("A2: {} / {} = {:.4f}%".format(a2_num, len(data), a2_num / len(data) * 100))
    print("A3: {} / {} = {:.4f}%".format(a3_num, len(data), a3_num / len(data) * 100))
    print("A1 & A2 & A3: {} / {} = {:.4f}%".format(all_num, len(data), all_num / len(data) * 100))
    
    
    # plot the bar chart
    results = {"A1": a1_num / len(data) * 100,
               "A2": a2_num / len(data) * 100,
               "A3": a3_num / len(data) * 100,
               "A1 & A2 & A3": all_num / len(data) * 100}
    save_file = os.path.join(args.save_path, "data_quality.pdf")
    plot_bar_chart(results, save_file)

if __name__ == "__main__":
    main()