import sys
import json
import numpy as np
import pdb


def main(numpy_path, pred_path, result_path):
    # pred logits
    logits = np.load(numpy_path)
    # pred json
    with open(pred_path) as f:
        pred = json.load(f)
    # core function
    assert len(logits) == len(pred)
    for i in range(len(logits)):
        # pred index
        logit = logits[i]
        top2_indices = np.argpartition(logit, -2)[-2:]
        # original data
        item = pred[i]
        context = item["context"]
        # pred documents
        pred_docs = []
        for index in top2_indices:
            title = context[index][0]
            pred_docs.append(title)
        item["pred_documents"] = pred_docs
    # save results
    with open(result_path, "w") as f:
        json.dump(pred, f, indent=4)


if __name__ == "__main__":
    numpy_path = "/home/mxdong/Codes/MHQA_Retriever/Checkpoints/Retriever1/Test_Preds/test_preds.npy"
    pred_path = "/home/mxdong/Data/HotpotQA/format_data/dev_dist.json"
    result_path = "/home/mxdong/Codes/MHQA_Retriever/Checkpoints/Retriever1/Test_Preds/pred_docs.json"
    main(numpy_path, pred_path, result_path)
