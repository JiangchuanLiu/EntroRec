import pandas as pd
import json


def generate_head_items_by_title(train_file_path, output_json_path, head_ratio=0.8):
    print(f"Loading training set: {train_file_path} ...")
    df_train = pd.read_csv(train_file_path)

    print("Counting interaction frequency by item title...")
    # Core change: this was changed from 'item_id' to 'item_title'
    item_counts = df_train['item_title'].value_counts().reset_index()
    item_counts.columns = ['item_title', 'count']

    total_interactions = item_counts['count'].sum()
    item_counts['cumulative_percentage'] = item_counts['count'].cumsum() / total_interactions

    # Set the cutoff for 80% of interactions
    head_items_df = item_counts[item_counts['cumulative_percentage'] <= head_ratio]

    # Extract head item titles and save them as a list of strings
    head_items_list = head_items_df['item_title'].tolist()

    total_items = len(item_counts)
    head_items_count = len(head_items_list)
    print("\n--- Statistics ---")
    print(f"Total training interactions: {total_interactions}")
    print(f"Total training items (Unique Titles): {total_items}")
    print(f"Number of head items covering {head_ratio * 100}% of interactions: {head_items_count}")
    print(f"Percentage of head items in the full item pool: {(head_items_count / total_items) * 100:.2f}%")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(head_items_list, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {head_items_count} head item titles to {output_json_path}")


if __name__ == "__main__":
    # Replace with your actual path
    TRAIN_FILE = "./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv"
    OUTPUT_FILE = "./data/Amazon/train/Office_Products_head_titles.json"

    generate_head_items_by_title(TRAIN_FILE, OUTPUT_FILE)
