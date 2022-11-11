import sys
import json
import numpy as np
import pdb


def main(numpy_path, pred_path, result_path, top_k, hotpot):
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
        top_k_indices = np.argpartition(logit, -top_k)[-top_k:]
        # original data
        item = pred[i]
        if hotpot:
            context = item["context"]
        else:
            context = item["paragraphs"]
        # pred documents
        pred_docs = []
        for index in top_k_indices:
            try:
                if hotpot:
                    title = context[index][0]
                else:
                    title = context[index]["title"]
            except IndexError:
                continue
            pred_docs.append(title)
        item["pred_documents"] = pred_docs
    # save results
    with open(result_path, "w") as f:
        json.dump(pred, f, indent=4)


if __name__ == "__main__":
    top_k = 4
    hotpot = False   # HotpotQA/2Wiki: True; MuSiQue: False
    numpy_path = "/home/mxdong/Codes/MHQA_Retriever/Checkpoints/Retriever3/Test_Preds/test_preds.npy"
    pred_path = "/home/mxdong/Data/MuSiQue/format_data/dev.json"
    result_path = "/home/mxdong/Codes/MHQA_Retriever/Checkpoints/Retriever3/Test_Preds/pred_docs.json"
    main(numpy_path, pred_path, result_path, top_k, hotpot)
