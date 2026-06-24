import os
import fire
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def gao(path, item_path, head_titles_path=""):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC = 0

    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    item_names = [_[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:
            item_dict[item_names[i]].append(item_ids[i])

    head_titles_set = set()
    if head_titles_path and os.path.exists(head_titles_path):
        with open(head_titles_path, 'r', encoding='utf-8') as f:
            head_titles_set = set(json.load(f))
        print(f"\nSuccessfully loaded the head item set, with {len(head_titles_set)} popular item titles.")
    else:
        print("\nhead_titles_path was not provided or the file does not exist; Matthew effect metrics cannot be calculated.")

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1

    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
            "MCD": [],
            "Tail_NDCG": [],
            "Tail_HR": [],
            "Diversity": []
        }

        f = open(p, 'r')
        test_data = json.load(f)
        f.close()

        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]

        ALL_MCD = None
        TAIL_NDCG = None
        TAIL_HR = None
        tail_sample_count = 0
        ALL_DIVERSITY = None
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))

                ALL_MCD = np.zeros(len(valid_topk))
                TAIL_NDCG = np.zeros(len(valid_topk))
                TAIL_HR = np.zeros(len(valid_topk))

                ALL_DIVERSITY = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")

            is_tail_target = (target_item not in head_titles_set)
            if is_tail_target:
                tail_sample_count += 1

            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                if sample[i] == target_item:
                    minID = i
                    break

            for idx_k, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue


                current_topk_items = sample[:topk]
                unique_count = len(set(current_topk_items))
                ALL_DIVERSITY[idx_k] += (unique_count / topk)
                head_count = sum(1 for x in current_topk_items if x in head_titles_set)
                ALL_MCD[idx_k] += (head_count / topk)

                if minID < topk:
                    ALLNDCG[idx_k] += (1 / math.log(minID + 2))
                    ALLHR[idx_k] += 1

                    if is_tail_target:
                        TAIL_NDCG[idx_k] += (1 / math.log(minID + 2))
                        TAIL_HR[idx_k] += 1

        valid_topk = [k for k in topk_list if k <= n_beam]
        total_samples = len(text)

        final_ndcg = (ALLNDCG / total_samples / (1.0 / math.log(2))).tolist()
        final_hr = (ALLHR / total_samples).tolist()
        final_mcd = (ALL_MCD / total_samples).tolist()
        final_diversity = (ALL_DIVERSITY / total_samples).tolist()
        if tail_sample_count > 0:
            final_tail_ndcg = (TAIL_NDCG / tail_sample_count / (1.0 / math.log(2))).tolist()
            final_tail_hr = (TAIL_HR / tail_sample_count).tolist()
        else:
            final_tail_ndcg = [0.0] * len(valid_topk)
            final_tail_hr = [0.0] * len(valid_topk)

        result_dict[p]["NDCG"] = final_ndcg
        result_dict[p]["HR"] = final_hr
        result_dict[p]["MCD"] = final_mcd
        result_dict[p]["Tail_NDCG"] = final_tail_ndcg
        result_dict[p]["Tail_HR"] = final_tail_hr
        result_dict[p]["Diversity"] = final_diversity
        out_metrics_path = p.replace(".json", "_metrics1.json")
        if out_metrics_path == p:
            out_metrics_path = p + "_metrics1.json"

        with open(out_metrics_path, 'w', encoding='utf-8') as f_out:
            save_data = {"TopK_settings": valid_topk}
            save_data.update(result_dict[p])
            json.dump(save_data, f_out, indent=4, ensure_ascii=False)

        print("\n" + "=" * 50)
        print(f"Evaluation completed | Beam Width: {n_beam} | Total samples: {total_samples}")
        print(f"Top-K settings: {valid_topk}")

        print(f"\n[Global Metrics]")
        print(f"NDCG:\t{final_ndcg}")
        print(f"HR:\t{final_hr}")
        print(f"DIVERSITY:\t{final_diversity}")
        print(f"\n[Matthew Effect X-axis Metric: Dominant Category Ratio (Generation MCD)]")
        print(f"MCD:\t{final_mcd}")

        print(f"\n[Matthew Effect Y-axis Metric: Long-tail Item Recommendation Quality (Tail-NDCG / Tail-HR)]")
        if tail_sample_count > 0:
            print(f"Long-tail sample count: {tail_sample_count} ({(tail_sample_count / total_samples) * 100:.2f}% of the test set)")
            print(f"Tail NDCG:\t{final_tail_ndcg}")
            print(f"Tail HR:\t{final_tail_hr}")
        else:
            print("No long-tail samples are available, so Tail metrics cannot be calculated.")
        print("=" * 50 + "\n")
        print(f"Number of generated items not matched in the dictionary (CC): {CC}")
        print(f"All metric data for the current model has been automatically saved to:\n {out_metrics_path}")

    return result_dict


if __name__ == '__main__':
    fire.Fire(gao)
