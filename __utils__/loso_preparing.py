import numpy as np


def prepare_for_loso(
    labels,
    clean_subjects,
    clean_videos_images,
    clean_subjects_videos_ground_truth_labels,
    k,
):
    labels = np.array(labels)
    y = []
    clean_videos_images_len = []
    groups = labels.copy()
    previous_sum_clean_video_images_len = 0
    sum_clean_subject_videos_ground_truth_labels_len = 0

    # Get total frames of each video
    for clean_video_images_index in range(len(clean_videos_images)):
        clean_videos_images_len.append(
            len(clean_videos_images[clean_video_images_index]) - k
        )

    # Integrate all frames into one dataset
    print("Frame Index for each subject:-")
    for clean_subject_videos_ground_truth_labels_index in range(
        len(clean_subjects_videos_ground_truth_labels)
    ):
        sum_clean_subject_videos_ground_truth_labels_len += len(
            clean_subjects_videos_ground_truth_labels[
                clean_subject_videos_ground_truth_labels_index
            ]
        )
        sum_clean_video_images_len = sum(
            clean_videos_images_len[:sum_clean_subject_videos_ground_truth_labels_len]
        )
        groups[
            previous_sum_clean_video_images_len:sum_clean_video_images_len
        ] = clean_subject_videos_ground_truth_labels_index
        print(
            "\nsubject {} ( group = {}): {} -> {}".format(
                clean_subjects[clean_subject_videos_ground_truth_labels_index],
                clean_subject_videos_ground_truth_labels_index,
                previous_sum_clean_video_images_len,
                sum_clean_video_images_len,
            )
        )
        print(
            "subject",
            clean_subjects[clean_subject_videos_ground_truth_labels_index],
            "has",
            len(
                clean_subjects_videos_ground_truth_labels[
                    clean_subject_videos_ground_truth_labels_index
                ]
            ),
            "clean video(s)",
        )
        print(
            "sum clean_subject_videos_ground_truth_labels_len: ",
            sum_clean_subject_videos_ground_truth_labels_len,
        )

        y.append(labels[previous_sum_clean_video_images_len:sum_clean_video_images_len])
        previous_sum_clean_video_images_len = sum_clean_video_images_len

    return y, groups


def legacy_prepare_for_loso(
    resampled_clean_videos_images_features,
    labels,
    clean_subjects,
    clean_videos_images,
    clean_subjects_videos_ground_truth_labels,
    k,
):
    y = np.array(labels)
    clean_videos_images_len = []
    groups = y.copy()
    previous_sum_clean_video_images_len = 0
    sum_clean_subject_videos_ground_truth_labels_len = 0

    # Get total frames of each video
    # for clean_video_images_index in range(len(clean_videos_images)):
    #     clean_videos_images_len.append(
    #         clean_videos_images[clean_video_images_index].shape[0] - k
    #     )
    for clean_video_images_index in range(len(clean_videos_images)):
        clean_videos_images_len.append(
            len(clean_videos_images[clean_video_images_index]) - k
        )

    # Integrate all frames into one dataset
    print("Frame Index for each subject:-")
    for clean_subject_videos_ground_truth_labels_index in range(
        len(clean_subjects_videos_ground_truth_labels)
    ):
        sum_clean_subject_videos_ground_truth_labels_len += len(
            clean_subjects_videos_ground_truth_labels[
                clean_subject_videos_ground_truth_labels_index
            ]
        )
        sum_clean_video_images_len = sum(
            clean_videos_images_len[:sum_clean_subject_videos_ground_truth_labels_len]
        )
        groups[
            previous_sum_clean_video_images_len:sum_clean_video_images_len
        ] = clean_subject_videos_ground_truth_labels_index
        print(
            "\nsubject",
            clean_subjects[clean_subject_videos_ground_truth_labels_index],
            "( group =",
            clean_subject_videos_ground_truth_labels_index,
            ") :",
            previous_sum_clean_video_images_len,
            "->",
            sum_clean_video_images_len,
        )
        print(
            "subject",
            clean_subjects[clean_subject_videos_ground_truth_labels_index],
            "has",
            len(
                clean_subjects_videos_ground_truth_labels[
                    clean_subject_videos_ground_truth_labels_index
                ]
            ),
            "clean video(s)",
        )
        print(
            "sum clean_subject_videos_ground_truth_labels_len: ",
            sum_clean_subject_videos_ground_truth_labels_len,
        )
        previous_sum_clean_video_images_len = sum_clean_video_images_len

    # X = [frame for video in dataset for frame in video]
    X = []
    for resampled_clean_video_images_features in resampled_clean_videos_images_features:
        for (
            resampled_clean_video_image_features
        ) in resampled_clean_video_images_features:
            X.append(resampled_clean_video_image_features)
    print("\nlen(X): ", len(X), ", len(y): ", len(y))
    return X, y, groups
