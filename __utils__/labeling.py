import numpy as np


def get_original_labels(
    clean_videos_images, clean_subjects_videos_ground_truth_labels, k
):
    list_labels = []
    labeled_video_count = 0

    for (
        clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels:
        for (
            clean_subject_video_ground_truth_labels
        ) in clean_subject_videos_ground_truth_labels:
            clean_subject_video_ground_truth_durations = np.array([])
            if len(clean_subject_video_ground_truth_labels) == 0:
                # Integrate all videos into one list
                # The last k frames are ignored
                list_labels += [0] * (len(clean_videos_images[labeled_video_count]) - k)
            else:
                for (
                    clean_subject_video_ground_truth_label
                ) in clean_subject_video_ground_truth_labels:
                    clean_subject_video_ground_truth_durations = np.append(
                        clean_subject_video_ground_truth_durations,
                        np.arange(
                            clean_subject_video_ground_truth_label[0],
                            clean_subject_video_ground_truth_label[1] + 1,
                        ),
                    )
                clean_subject_video_ground_truth_durations = np.unique(
                    clean_subject_video_ground_truth_durations
                )

                video_labels = np.zeros(
                    len(clean_videos_images[labeled_video_count]) - k
                )
                labels_for_calculation = np.arange(
                    len(clean_videos_images[labeled_video_count]) - k
                )

                video_labels = (
                    np.isin(
                        labels_for_calculation,
                        clean_subject_video_ground_truth_durations,
                    )
                    .astype(int)
                    .tolist()
                )

                # Integrate all videos into one list
                list_labels += video_labels

            labeled_video_count += 1

    print("Total frames:", len(list_labels))
    return list_labels


def get_pseudo_labels(
    clean_videos_images, clean_subjects_videos_ground_truth_labels, k
):
    list_pseudo_labels = []
    pseudo_labeled_video_count = 0

    for (
        clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels:
        for (
            clean_subject_video_ground_truth_labels
        ) in clean_subject_videos_ground_truth_labels:
            clean_subject_video_ground_truth_durations = []
            if len(clean_subject_video_ground_truth_labels) == 0:
                # Integrate all videos into one dataset
                # The last k frames are ignored
                list_pseudo_labels += [0] * (
                    len(clean_videos_images[pseudo_labeled_video_count]) - k
                )
            else:
                for (
                    clean_subject_video_ground_truth_label
                ) in clean_subject_video_ground_truth_labels:
                    clean_subject_video_ground_truth_durations = np.append(
                        clean_subject_video_ground_truth_durations,
                        np.arange(
                            clean_subject_video_ground_truth_label[0],
                            clean_subject_video_ground_truth_label[1] + 1,
                        ),
                    )
                clean_subject_video_ground_truth_durations = np.unique(
                    clean_subject_video_ground_truth_durations
                )

                video_pseudo_labels = np.zeros(
                    len(clean_videos_images[pseudo_labeled_video_count]) - k
                )
                # - 1 for shape consistency
                labels_for_sliding_window = np.arange(
                    len(clean_videos_images[pseudo_labeled_video_count]) - 1
                )
                view = np.lib.stride_tricks.sliding_window_view(
                    labels_for_sliding_window, k
                )

                # each video may contains more than one micro-expression
                # so there may be more than one duration
                # for (
                #     clean_subject_video_ground_truth_duration
                # ) in clean_subject_video_ground_truth_durations:
                # Numerator
                numerator = (
                    np.isin(view, clean_subject_video_ground_truth_durations)
                    .astype(int)
                    .sum(axis=1)
                )

                # Denumerator
                view_not_in_duration_count = (
                    np.isin(
                        view, clean_subject_video_ground_truth_durations, invert=True
                    )
                    .astype(int)
                    .sum(axis=1)
                )
                denumerator = (
                    clean_subject_video_ground_truth_durations.size
                    + view_not_in_duration_count
                )

                iou = numerator / denumerator
                # Function g (Heaviside step function)
                video_pseudo_labels = np.where(iou > 0, 1, 0).tolist()

                # Integrate all videos into one list
                list_pseudo_labels += video_pseudo_labels

            pseudo_labeled_video_count += 1

    print("Total frames:", len(list_pseudo_labels))
    return list_pseudo_labels


def legacy_get_original_labels(
    clean_videos_images, clean_subjects_videos_ground_truth_labels, k
):
    list_labels = []
    labeled_video_count = 0

    for (
        clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels:
        for (
            clean_subject_video_ground_truth_labels
        ) in clean_subject_videos_ground_truth_labels:
            clean_subject_video_ground_truth_durations = []
            if len(clean_subject_video_ground_truth_labels) == 0:
                # Integrate all videos into one dataset
                # The last k frames are ignored
                list_labels += [0] * (len(clean_videos_images[labeled_video_count]) - k)
            else:
                video_labels = [0] * (len(clean_videos_images[labeled_video_count]) - k)

                for (
                    clean_subject_video_ground_truth_label
                ) in clean_subject_video_ground_truth_labels:
                    clean_subject_video_ground_truth_durations.append(
                        np.arange(
                            clean_subject_video_ground_truth_label[0],
                            clean_subject_video_ground_truth_label[1] + 1,
                        )
                    )

                # each video may contains more than one micro-expression
                # so there may be more than one duration
                for (
                    clean_subject_video_ground_truth_duration
                ) in clean_subject_video_ground_truth_durations:
                    for video_label_index in range(len(video_labels)):
                        if (
                            len(
                                np.intersect1d(
                                    video_label_index,
                                    clean_subject_video_ground_truth_duration,
                                )
                            )
                            > 0
                        ):
                            video_labels[video_label_index] = 1
                # Integrate all videos into one dataset
                list_labels += video_labels
            labeled_video_count += 1

    print("Total frames:", len(list_labels))
    return list_labels


def legacy_get_pseudo_labels(
    clean_videos_images, clean_subjects_videos_ground_truth_labels, k
):
    list_pseudo_labels = []
    pseudo_labeled_video_count = 0

    for (
        clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels:
        for (
            clean_subject_video_ground_truth_labels
        ) in clean_subject_videos_ground_truth_labels:
            clean_subject_video_ground_truth_durations = []
            if len(clean_subject_video_ground_truth_labels) == 0:
                # Integrate all videos into one dataset
                # The last k frames are ignored
                list_pseudo_labels += [0] * (
                    len(clean_videos_images[pseudo_labeled_video_count]) - k
                )

            else:
                video_pseudo_labels = [0] * (
                    len(clean_videos_images[pseudo_labeled_video_count]) - k
                )

                for (
                    clean_subject_video_ground_truth_label
                ) in clean_subject_video_ground_truth_labels:
                    # the first `[0] + 1` is weird
                    clean_subject_video_ground_truth_durations.append(
                        np.arange(
                            clean_subject_video_ground_truth_label[0],
                            clean_subject_video_ground_truth_label[1] + 1,
                        )
                    )

                # each video may contains more than one micro-expression
                # so there may be more than one duration
                for (
                    clean_subject_video_ground_truth_duration
                ) in clean_subject_video_ground_truth_durations:
                    for video_pseudo_label_index in range(len(video_pseudo_labels)):
                        sliding_window = np.arange(
                            video_pseudo_label_index, video_pseudo_label_index + k
                        )
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if video_pseudo_labels[video_pseudo_label_index] < len(
                            np.intersect1d(
                                sliding_window,
                                clean_subject_video_ground_truth_duration,
                            )
                        ) / len(
                            np.union1d(
                                sliding_window,
                                clean_subject_video_ground_truth_duration,
                            )
                        ):
                            video_pseudo_labels[video_pseudo_label_index] = 1

                # Integrate all videos into one list
                list_pseudo_labels += video_pseudo_labels

            pseudo_labeled_video_count += 1

    print("Total frames:", len(list_pseudo_labels))
    return list_pseudo_labels
