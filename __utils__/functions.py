import _pickle
from pathlib import Path
import natsort
from collections import Counter
import numpy as np
import sys
import random
import cv2
from skimage.util import random_noise
from __utils__.mean_average_precision.mean_average_precision import (
    MeanAveragePrecision2d,
)
import threading
from queue import Queue
from __utils__ import spotting

random.seed(1)
epsilon = sys.float_info.epsilon


def load_subject_pkl_file(expression_type, dataset_dir, subject, image_size):
    subject_pkl_file_path = Path(
        dataset_dir,
        f"{expression_type}_clean_subjects_x_{image_size}",
        subject,
    ).with_suffix(".pkl")
    with open(subject_pkl_file_path, "rb") as pkl_file:
        X = _pickle.load(pkl_file)
        pkl_file.close()
    return X


def load_test_video_pkl_file(
    expression_type, test_dataset_dir, test_video_name, image_size
):
    test_dataset_dir = Path(test_dataset_dir)
    test_video_pkl_file_path = test_dataset_dir.parent / Path(
        test_dataset_dir.name.split("_")[0] + "_" + test_dataset_dir.name.split("_")[1],
        f"{expression_type}_videos_x_{image_size}",
        test_video_name,
    ).with_suffix(".pkl")
    with open(test_video_pkl_file_path, "rb") as pkl_file:
        X_test = _pickle.load(pkl_file)
        pkl_file.close()
    return X_test


def downsample(X_train, y_train):
    """
    Downsampling non expression samples the dataset by 1/2 to reduce dataset bias
    """
    print("Dataset Labels", Counter(y_train))
    # the unique_count is sorted
    unique, unique_count = np.unique(y_train, return_counts=True)

    if unique_count[0] > unique_count[1]:
        random_count = int(unique_count[0] * 1 / 2)

        # Randomly remove non expression samples (With label 0) from dataset
        random_y_train_indices = random.sample(
            [y_train_index for y_train_index, i in enumerate(y_train) if i == 0],
            random_count,
        )
        random_y_train_indices += (
            y_train_index for y_train_index, i in enumerate(y_train) if i > 0
        )
        random_y_train_indices.sort()
        X_train = [
            X_train[random_y_train_index]
            for random_y_train_index in random_y_train_indices
        ]
        y_train = [
            y_train[random_y_train_index]
            for random_y_train_index in random_y_train_indices
        ]
        print("After Downsampling Dataset Labels", Counter(y_train))
    return X_train, y_train


def augment_data(X, y, image_size=128, subject=None):
    transformations = {
        0: lambda image: np.fliplr(image),
        1: lambda image: cv2.GaussianBlur(image, (7, 7), 0),
        2: lambda image: random_noise(image),
    }
    y1 = y.copy()
    for index, label in enumerate(y1):
        # Only augment on expression samples (label=1)
        if label == 1:
            for augment_type in range(3):
                if image_size == 128:
                    img_transformed = transformations[augment_type](X[index]).reshape(
                        42, 42, 3
                    )
                elif image_size == 256:
                    img_transformed = transformations[augment_type](X[index]).reshape(
                        84, 84, 3
                    )
                X.append(np.array(img_transformed))
                y.append(1)
    if subject is not None:
        print(
            f"subject {subject} after augmentation labels:",
            Counter(y),
        )
    else:
        print("After Augmentation Dataset Labels", Counter(y))
    return X, y


def normalize(images):
    for index in range(len(images)):
        for channel in range(3):
            images[index][:, :, channel] = cv2.normalize(
                images[index][:, :, channel],
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
            )
    return images


def save_pred(split, clean_subjects, preds_path, pred):
    (preds_path.parent / preds_path.stem).mkdir(exist_ok=True)
    pred_path = (
        preds_path.parent
        / preds_path.stem
        / f"split_{split}_clean_subject_{clean_subjects[split-1]}"
    ).with_suffix(".pkl")
    with open(pred_path, "wb") as pkl_file:
        _pickle.dump(pred, pkl_file)
        pkl_file.close()
    print(f"Dumped {pred_path}")


def merge_preds(preds_path):
    preds = []
    for pred_path in natsort.natsorted((preds_path.parent / preds_path.stem).iterdir()):
        with open(pred_path, "rb") as pkl_file:
            pred = _pickle.load(pkl_file)
            pkl_file.close()
        preds.append(pred)
        print(pred_path)

    with open(preds_path, "wb") as pkl_file:
        _pickle.dump(preds, pkl_file)
        pkl_file.close()


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


def spot_and_evaluate(
    preds,
    clean_subjects_videos_ground_truth_labels,
    clean_videos_images,
    clean_subjects,
    clean_subjects_videos_code,
    k,
    p,
    show_plot_or_not=False,
    print_or_not=True,
):
    total_ground_truth_labels_count = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    result_dict = {
        "p": [],
        "true_positive": [],
        "false_positive": [],
        "false_negative": [],
        "expression_count": [],
        "precision": [],
        "recall": [],
        "F1_score": [],
    }

    for split, pred in enumerate(preds):
        print(f"Split {split+1}/{len(preds)} is in process.")
        # spotting
        metric_fn, total_ground_truth_labels_count = spotting.spot(
            pred,
            total_ground_truth_labels_count,
            clean_subjects_videos_ground_truth_labels,
            split,
            clean_videos_images,
            clean_subjects,
            clean_subjects_videos_code,
            k,
            metric_fn,
            p,
            find_peaks_distance=k,
            show_plot_or_not=show_plot_or_not,
        )

        # evaluation
        # every evaluation considers all splitted videos
        (
            true_positive,
            false_positive,
            false_negative,
            precision,
            recall,
            F1_score,
        ) = evaluate(
            metric_fn,
            total_ground_truth_labels_count,
        )

        result_dict["true_positive"].append(true_positive)
        result_dict["false_positive"].append(false_positive)
        result_dict["false_negative"].append(false_negative)
        result_dict["expression_count"].append(total_ground_truth_labels_count)
        result_dict["precision"].append(precision)
        result_dict["recall"].append(recall)
        result_dict["F1_score"].append(F1_score)

        print(f"Split {split+1}/{len(preds)} is processed.\n")

    return metric_fn, result_dict


def multithread_spot_and_evaluate(
    preds,
    clean_subjects_videos_ground_truth_labels,
    clean_videos_images,
    clean_subjects,
    clean_subjects_videos_code,
    k,
    p,
    dict_queue,
):
    # Don't feed the global parameter dict into this thread function
    # unexpected results may occur
    temp_dict = {
        "p": [],
        "true_positive": [],
        "false_positive": [],
        "false_negative": [],
        "expression_count": [],
        "precision": [],
        "recall": [],
        "F1_score": [],
    }
    total_ground_truth_labels_count = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    for split, pred in enumerate(preds):
        # Spotting
        metric_fn, total_ground_truth_labels_count = spotting.spot(
            pred,
            total_ground_truth_labels_count,
            clean_subjects_videos_ground_truth_labels,
            split,
            clean_videos_images,
            clean_subjects,
            clean_subjects_videos_code,
            k,
            metric_fn,
            p,
            find_peaks_distance=k,
            show_plot_or_not=False,
            print_or_not=False,
        )

    # Evaluation
    (
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall,
        F1_score,
    ) = evaluate(metric_fn, total_ground_truth_labels_count, print_or_not=False)

    # Put data into the queue
    temp_dict["p"].append(p)
    temp_dict["true_positive"].append(true_positive)
    temp_dict["false_positive"].append(false_positive)
    temp_dict["false_negative"].append(false_negative)
    temp_dict["expression_count"].append(total_ground_truth_labels_count)
    temp_dict["precision"].append(precision)
    temp_dict["recall"].append(recall)
    temp_dict["F1_score"].append(F1_score)
    dict_queue.put(temp_dict)


def ablation_study_p(
    preds,
    clean_subjects_videos_ground_truth_labels,
    clean_videos_images,
    clean_subjects,
    clean_subjects_videos_code,
    k,
):
    """
    10 threads
    """
    ablation_dict = {
        "p": [],
        "true_positive": [],
        "false_positive": [],
        "false_negative": [],
        "expression_count": [],
        "precision": [],
        "recall": [],
        "F1_score": [],
    }

    print(" p | TP | FP | FN | Precision | Recall | F1-Score")
    p_arange = np.arange(0.00, 0.1, 0.01)
    segments = np.arange(0.0, 1.0, 0.1)
    dict_queues = []
    threads = []
    for _ in segments:
        dict_queues.append(Queue())

    for p in p_arange:
        for segment_index, segment in enumerate(segments):
            thread = threading.Thread(
                target=multithread_spot_and_evaluate,
                args=(
                    preds,
                    clean_subjects_videos_ground_truth_labels,
                    clean_videos_images,
                    clean_subjects,
                    clean_subjects_videos_code,
                    k,
                    p + segment,
                    dict_queues[segment_index],
                ),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        print(f" p: {p}", end="\r")

    # Attention!!!
    # Different loop sequence
    for segment_index, segment in enumerate(segments):
        for p in p_arange:
            p_segment_matrix = dict_queues[segment_index].get()
            ablation_dict["p"] += p_segment_matrix["p"]
            ablation_dict["true_positive"] += p_segment_matrix["true_positive"]
            ablation_dict["false_positive"] += p_segment_matrix["false_positive"]
            ablation_dict["false_negative"] += p_segment_matrix["false_negative"]
            ablation_dict["expression_count"] += p_segment_matrix["expression_count"]
            ablation_dict["precision"] += p_segment_matrix["precision"]
            ablation_dict["recall"] += p_segment_matrix["recall"]
            ablation_dict["F1_score"] += p_segment_matrix["F1_score"]
            print(
                "{:.2f} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    ablation_dict["p"][-1],
                    ablation_dict["true_positive"][-1],
                    ablation_dict["false_positive"][-1],
                    ablation_dict["false_negative"][-1],
                    ablation_dict["precision"][-1],
                    ablation_dict["recall"][-1],
                    ablation_dict["F1_score"][-1],
                )
            )

    return ablation_dict


def ablation_study_p_dev(
    preds,
    clean_subjects_videos_ground_truth_labels,
    clean_videos_images,
    clean_subjects,
    clean_subjects_videos_code,
    k,
):
    """
    100 theads
    """
    ablation_dict = {
        "p": [],
        "true_positive": [],
        "false_positive": [],
        "false_negative": [],
        "expression_count": [],
        "precision": [],
        "recall": [],
        "F1_score": [],
    }

    print(" p | TP | FP | FN | Precision | Recall | F1-Score")
    p_arange = np.arange(0.01, 1.0, 0.01)
    dict_queues = []
    threads = []
    for _ in p_arange:
        dict_queues.append(Queue())

    for p_index, p in enumerate(p_arange):
        thread = threading.Thread(
            target=multithread_spot_and_evaluate,
            args=(
                preds,
                clean_subjects_videos_ground_truth_labels,
                clean_videos_images,
                clean_subjects,
                clean_subjects_videos_code,
                k,
                p,
                dict_queues[p_index],
            ),
        )
        thread.start()
        threads.append(thread)
        for thread in threads:
            thread.join()
        print(f" p: {p}", end="\r")

    # Attention!!!
    # Different loop sequence
    for p_index, p in enumerate(p_arange):
        p_segment_matrix = dict_queues[p_index].get()
        ablation_dict["p"] += p_segment_matrix["p"]
        ablation_dict["true_positive"] += p_segment_matrix["true_positive"]
        ablation_dict["false_positive"] += p_segment_matrix["false_positive"]
        ablation_dict["false_negative"] += p_segment_matrix["false_negative"]
        ablation_dict["expression_count"] += p_segment_matrix["expression_count"]
        ablation_dict["precision"] += p_segment_matrix["precision"]
        ablation_dict["recall"] += p_segment_matrix["recall"]
        ablation_dict["F1_score"] += p_segment_matrix["F1_score"]
        print(
            "{:.2f} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                ablation_dict["p"][-1],
                ablation_dict["true_positive"][-1],
                ablation_dict["false_positive"][-1],
                ablation_dict["false_negative"][-1],
                ablation_dict["precision"][-1],
                ablation_dict["recall"][-1],
                ablation_dict["F1_score"][-1],
            )
        )

    return ablation_dict
