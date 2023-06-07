import numpy as np
import sys

epsilon = sys.float_info.epsilon


# Get TP, FP, FN for final evaluation
def evaluate(metric_fn, ground_truth_labels_count, print_or_not=True):
    true_positive = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]["tp"]))
    false_positive = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]["fp"]))
    false_negative = ground_truth_labels_count - true_positive
    if print_or_not is True:
        print(
            "\nTrue Positive: {}, False Posive: {}, False Negative: {}".format(
                true_positive,
                false_positive,
                false_negative,
            )
        )
    # add epsilon to avoid float division by zero
    precision = true_positive / (true_positive + false_positive) + epsilon
    recall = true_positive / (true_positive + false_negative) + epsilon
    F1_score = (2 * precision * recall) / (precision + recall) + epsilon
    if print_or_not is True:
        print(f"Precision = {precision}, Recall ={recall}, F1-Score = {F1_score}")
    return true_positive, false_positive, false_negative, precision, recall, F1_score


def final_evaluate(metric_fn, result_dict):
    true_positive = result_dict["true_positive"][-1]
    false_positive = result_dict["false_positive"][-1]
    false_negative = result_dict["false_negative"][-1]
    print(
        "True Positive: {}, False Posive: {}, False Negative: {}".format(
            true_positive,
            false_positive,
            false_negative,
        )
    )
    # print(
    #     "COCO AP@[.5:.95]:",
    #     round(
    #         metric_fn.value(
    #             iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy="soft"
    #         )["mAP"],
    #         4,
    #     ),
    # )

    final_precision = result_dict["precision"][-1]
    final_recall = result_dict["recall"][-1]
    final_F1_score = result_dict["F1_score"][-1]
    print(
        "Final Precision = {},\nFinal Recall ={},\nFinal F1-Score = {}\n".format(
            final_precision, final_recall, final_F1_score
        )
    )

    highest_precision = np.max(result_dict["precision"])
    highest_recall = np.max(result_dict["recall"])
    highest_F1_score = np.max(result_dict["F1_score"])
    print(
        "Highest Precision = {},\nHighest Recall ={},\nHighest F1-Score = {}".format(
            highest_precision, highest_recall, highest_F1_score
        )
    )
