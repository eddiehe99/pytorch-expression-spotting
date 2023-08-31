import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from scipy.signal import find_peaks, savgol_filter
import time

plt.style.use("ggplot")
backend_inline.set_matplotlib_formats("svg")


def spot(
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
    find_peaks_distance,
    show_plot_or_not,
    print_or_not=True,
):
    previous_index = 0
    splitted_clean_subjects_videos_len = 0

    # every subject
    for (
        splitted_clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels[:split]:
        splitted_clean_subjects_videos_len += len(
            splitted_clean_subject_videos_ground_truth_labels
        )
    if print_or_not is True:
        print(
            splitted_clean_subjects_videos_len,
            "video(s) have been processed.",
        )

    # every videos of the current subject
    for (
        current_clean_subject_video_index,
        current_clean_subject_video_ground_truth_labels,
    ) in enumerate(clean_subjects_videos_ground_truth_labels[split]):
        predictions = []

        current_clean_subject_video_images_len = len(
            clean_videos_images[
                splitted_clean_subjects_videos_len + current_clean_subject_video_index
            ]
        )
        current_clean_subject_video_features_end_index = (
            previous_index + current_clean_subject_video_images_len - k
        )

        # Get related frames to each video
        scores = np.array(
            pred[previous_index:current_clean_subject_video_features_end_index]
        ).flatten()

        previous_index = current_clean_subject_video_features_end_index

        """
        Divided by (1 / 2k).
        Weird! Hard to understand!

        My interpretation:

        One:
        It is designed to be divided by (1 / (2k + 1)),
        but the offset score is not considered in the actual calculation.

        Two:
        Or, actually, it is really originally designed to be divided by (1 / 2k).
        Neither it forget to count the current frame score
        in the calculation (see the formulation in the paper),
        nor the offset score is intended to be dropped.

        If the interpretation two is right, then it is the code that misleads readers.
        I suppose it does not want to conisder the offset score but the code has bug.
        Apex = Onset + k
        Offset = Onset + 2k
        [Onset, (k - 1) frames, Apex, (k - 1) frames, Offset]
        Hence, [x : x + 2 * k] means [Onset, (k - 1) frames, Apex, (k - 1) frames]
        and the legacy smoothed scores omit the last element.
        As there are (k - 1) frames on the right of the Apex be considered, not k frames.
        But it does not matter, omitting the last element does not affect the results.
        """

        # # Score smoothing
        # legacy_smoothed_scores = scores.copy()
        # for x in range(len(scores) - 2 * k):
        #     legacy_smoothed_scores[x + k] = scores[x : x + 2 * k].mean()
        # legacy_smoothed_scores = legacy_smoothed_scores[k:-k]

        # Fast score smoothing
        smoothed_scores = np.lib.stride_tricks.sliding_window_view(scores, 2 * k)
        # smoothed_scores = smoothed_scores[:-1]
        smoothed_scores = np.mean(smoothed_scores, axis=-1)

        """
        Divided by 1 / (2k + 1) 
        """
        # # Score smoothing
        # legacy_smoothed_scores = scores.copy()
        # for x in range(len(scores) - 2 * k):
        #     legacy_smoothed_scores[x + k] = scores[x : x + 2 * k + 1].mean()
        # legacy_smoothed_scores = legacy_smoothed_scores[k:-k]

        # # Fast score smoothing
        # smoothed_scores = np.lib.stride_tricks.sliding_window_view(scores, 2 * k + 1)
        # smoothed_scores = np.mean(smoothed_scores, axis=-1)

        # # More filtering
        # smoothed_scores = savgol_filter(smoothed_scores, 6, 2, mode="nearest")
        # smoothed_scores = np.convolve(
        #     smoothed_scores,
        #     (np.ones(2) / 2).flatten(),
        #     mode="valid",
        # )

        # Moilanen threshold technique
        threshold = smoothed_scores.mean() + p * (
            max(smoothed_scores) - smoothed_scores.mean()
        )
        peaks, _ = find_peaks(
            smoothed_scores, height=threshold, distance=find_peaks_distance
        )

        # parameters of mean_average_precision
        """Add sample to evaluation.

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.

        Input format:
            preds: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        """
        # Occurs when no peak is detected,
        # simply give a value to pass the exception in mean_average_precision
        if len(peaks) == 0:
            predictions.append([0, 0, 0, 0, 0, 0])

        for peak in peaks:
            # Align the peak
            peak += k
            # Extend left and right side of peak by k frames
            predictions.append([peak - k, 0, peak + k, 0, 0, 0])

        ground_truth_labels = []
        for (
            current_clean_subject_video_ground_truth_label
        ) in current_clean_subject_video_ground_truth_labels:
            # `- 1` is processed in label_processing
            ground_truth_labels.append(
                [
                    current_clean_subject_video_ground_truth_label[0],
                    0,
                    current_clean_subject_video_ground_truth_label[1],
                    0,
                    0,
                    0,
                    0,
                ]
            )
            total_ground_truth_labels_count += 1

        if show_plot_or_not is True:
            # Note for some video the ground truth samples is below frame index 0
            # due to the effect of smoothing, but no impact to the evaluation
            plt.figure(figsize=(11, 4))
            frames = range(k, len(smoothed_scores) + k)
            # plt.plot(scores, color="tab:gray", label="scores")
            plt.plot(
                frames,
                smoothed_scores,
                color="tab:blue",
                label="smoothed scores",
            )
            # plt.plot(
            #     smoothed_scores_copy, color="tab:blue", label="smoothed scores"
            # )
            plt.axhline(y=threshold, color="tab:green", label="threshold (T)")
            plt.xlabel("Frame")
            plt.ylabel("Score")

            for peak in peaks:
                # Align the peak
                peak += k
                plt.axvline(
                    x=peak,
                    color="tab:olive",
                    label="spotted apex",
                )
                # plt.axvline(
                #     x=peak - k,
                #     color="tab:brown",
                #     label="predicted onset",
                # )
                # plt.axvline(
                #     x=peak + k,
                #     color="tab:gray",
                #     label="predicted offset",
                # )

            for (
                current_clean_subject_video_ground_truth_label
            ) in current_clean_subject_video_ground_truth_labels:
                # `- 1` is processed in label_processing
                plt.axvline(
                    x=current_clean_subject_video_ground_truth_label[0],
                    color="tab:orange",
                    label="onset",
                    linestyle="dashed",
                )
                plt.axvline(
                    x=current_clean_subject_video_ground_truth_label[1],
                    color="tab:red",
                    label="offset",
                    linestyle="dashed",
                )

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()

        if print_or_not is True:
            print(
                "The current video be processed: subject {}, video {}".format(
                    clean_subjects[split],
                    clean_subjects_videos_code[split][
                        current_clean_subject_video_index
                    ],
                )
            )

        # IoU = 0.5 according to MEGC2020 metrics
        metric_fn.add(
            np.array(predictions),
            np.array(ground_truth_labels),
        )

    return metric_fn, total_ground_truth_labels_count


def spot_test(preds, k, p, test_videos_name, show_plot_or_not=True):
    test_videos_preds = []
    for index, pred in enumerate(preds):
        print(f"test video {index+1}/{len(preds)} is in process.")
        test_video_preds = []
        scores = np.array(pred).flatten()
        smoothed_scores = scores.copy()

        # Score smoothing
        for x in range(len(scores[k:-k])):
            smoothed_scores[x + k] = scores[x : x + 2 * k].mean()
        smoothed_scores = smoothed_scores[k:-k]
        # smoothed_scores = savgol_filter(smoothed_scores, 2 * k, k, mode="nearest")
        # smoothed_scores = np.convolve(
        #     smoothed_scores.flatten(),
        #     (np.ones(2 * k) / 2 * k).flatten(),
        #     mode="valid",
        # )

        # Moilanen threshold technique
        threshold = smoothed_scores.mean() + p * (
            max(smoothed_scores) - smoothed_scores.mean()
        )
        peaks, _ = find_peaks(smoothed_scores, height=threshold, distance=k)

        if len(peaks) != 0:
            for peak in peaks:
                # align the peak
                peak += k
                # Extend left and right side of peak by k frames
                test_video_preds.append([peak - k, peak + k])
        else:
            print(f"test video {test_videos_name[index]} has no preds!")
        test_videos_preds.append(test_video_preds)

        # Plot the result to see the peaks
        # Note for some video the ground truth samples is below frame index 0
        # due to the effect of smoothing, but no impact to the evaluation
        if show_plot_or_not is True:
            # frames = range(len(smoothed_scores))
            plt.figure(figsize=(11, 4))
            # plt.plot(frames, scores[k:-k], label="scores")
            # plt.plot(frames, smoothed_scores, label="smoothed scores")
            plt.plot(smoothed_scores, color="tab:blue", label="smoothed scores")
            for peak in peaks:
                plt.axvline(
                    # the peak is already aligned above
                    x=peak,
                    color="tab:olive",
                    label="predicted apex",
                )
            plt.axhline(y=threshold, color="tab:green", label="threshold (T)")
            plt.xlabel("Frame")
            plt.ylabel("Score")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            # plt.legend()
            plt.show()
        print(f"The current test video be processed: {test_videos_name[index]}")
        print(f"test video {index+1}/{len(preds)} is processed.\n")
    return test_videos_preds
