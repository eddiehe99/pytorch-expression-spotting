import numpy as np
import pandas as pd
from pathlib import Path
import time


def load_excel(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if dataset_dir.name == "CAS(ME)^2":
        Excel_file = pd.ExcelFile(dataset_dir / "CAS(ME)^2code_final(Updated).xlsx")

        column_names = [
            "participant",
            "video_name_&_expression_number",
            "onset",
            "apex",
            "offset",
            "AUs",
            "extimated_emotion",
            "expression_type",
            "self-reported_emotion",
        ]
        Excel_data = Excel_file.parse(
            Excel_file.sheet_names[0], header=None, names=column_names
        )

        video_names = []
        for video_name in Excel_data.iloc[:, 1]:
            video_names.append(video_name.split("_")[0])
        Excel_data["video_name"] = video_names

        Excel_sheet_data_3 = Excel_file.parse(
            Excel_file.sheet_names[2], header=None, converters={0: str}
        )
        video_codes = dict(
            zip(Excel_sheet_data_3.iloc[:, 1], Excel_sheet_data_3.iloc[:, 0])
        )
        Excel_data["video_code"] = [video_codes[i] for i in Excel_data["video_name"]]

        Excel_sheet_data_2 = Excel_file.parse(Excel_file.sheet_names[1], header=None)
        subjects_dict = dict(
            zip(Excel_sheet_data_2.iloc[:, 2], Excel_sheet_data_2.iloc[:, 1])
        )
        Excel_data["subject"] = [subjects_dict[i] for i in Excel_data["participant"]]

    elif dataset_dir.name == "SAMM_longvideos":
        column_names = [
            "Subject",
            "Filename",
            "Inducement Code",
            "Onset",
            "Apex",
            "Offset",
            "Duration",
            "Type",
            "Action Units",
            "Notes",
        ]
        Excel_data = pd.read_excel(
            dataset_dir / "SAMM_LongVideos_V3_Release.xlsx",
            dtype=object,
            header=None,
            names=column_names,
            skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

        # Excel_data = Excel_file.parse(
        #     Excel_file.sheet_names[0],
        #     header=None,
        #     names=column_names,
        #     skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # )

        video_codes = []
        subject_codes = []
        for subject_video_expression in Excel_data.iloc[:, 1]:
            video_codes.append(
                str(subject_video_expression).split("_")[0]
                + "_"
                + str(subject_video_expression).split("_")[1]
            )
            subject_codes.append(str(subject_video_expression).split("_")[0])
        Excel_data["subject_video_code"] = video_codes
        Excel_data["subject_code"] = subject_codes
        # Synchronize the columns name with CAS(ME)^2
        Excel_data.rename(
            columns={
                "Subject": "subject",
                "Inducement Code": "video_code",
                "Onset": "onset",
                "Apex": "apex",
                "Offset": "offset",
                "Type": "expression_type",
            },
            inplace=True,
        )
        Excel_data["video_code"] = Excel_data["video_code"].apply(str)
        print("Data Columns:", Excel_data.columns)  # Final data column
    return Excel_data


def load_ground_truth_labels(
    dataset_dir,
    expression_type,
    videos_images,
    subjects_videos_code,
    subjects,
    Excel_data,
):
    dataset_dir = Path(dataset_dir)
    dataset_expression_type = expression_type
    if dataset_dir.name == "CAS(ME)^2" and expression_type == "me":
        dataset_expression_type = "micro-expression"
    elif dataset_dir.name == "CAS(ME)^2" and expression_type == "mae":
        dataset_expression_type = "macro-expression"
    elif dataset_dir.name == "SAMM_longvideos" and expression_type == "me":
        dataset_expression_type = "Micro - 1/2"
    elif dataset_dir.name == "SAMM_longvideos" and expression_type == "mae":
        dataset_expression_type = "Macro"

    required_videos_index = []
    video_count = 0
    subjects_videos_ground_truth_lables = []
    for subject_videos_code_index, subject_videos_code in enumerate(
        subjects_videos_code
    ):
        subjects_videos_ground_truth_lables.append([])
        for subject_video_code in subject_videos_code:
            subject_video_onsets_offsets = []
            for i, row in Excel_data.iterrows():
                # S15, S16... for CAS(ME)^2
                # 001, 002... for SAMM_longvideos
                condition = (
                    (row["subject"] == subjects[subject_videos_code_index])
                    and (row["video_code"] == subject_video_code)
                    and (row["expression_type"] == dataset_expression_type)
                )
                if condition is True:
                    if row["offset"] == 0:  # Take apex if offset is 0
                        subject_video_onsets_offsets.append(
                            [int(row["onset"] - 1), int(row["apex"] - 1)]
                        )
                    else:
                        # Ignore the samples that is extremely long in SAMMLV
                        if dataset_expression_type != "Macro" or int(row["onset"]) != 1:
                            subject_video_onsets_offsets.append(
                                [int(row["onset"] - 1), int(row["offset"] - 1)]
                            )
            if len(subject_video_onsets_offsets) > 0:
                # To get the video that is needed
                required_videos_index.append(video_count)
            subjects_videos_ground_truth_lables[-1].append(subject_video_onsets_offsets)
            video_count += 1

    # Remove unused video
    clean_subjects_videos_ground_truth_labels = []
    clean_subjects_videos_code = []
    clean_subjects = []
    count = 0
    for subject_videos_onsets_offsets_index, subject_videos_onsets_offsets in enumerate(
        subjects_videos_ground_truth_lables
    ):
        clean_subjects_videos_ground_truth_labels.append([])
        clean_subjects_videos_code.append([])
        for (
            subject_video_onsets_offsets_index,
            subject_video_onsets_offsets,
        ) in enumerate(subject_videos_onsets_offsets):
            if count in required_videos_index:
                clean_subjects_videos_ground_truth_labels[-1].append(
                    subject_video_onsets_offsets
                )
                clean_subjects_videos_code[-1].append(
                    subjects_videos_code[subject_videos_onsets_offsets_index][
                        subject_video_onsets_offsets_index
                    ]
                )
                clean_subjects.append(subjects[subject_videos_onsets_offsets_index])
            count += 1

    # Remove the empty data in array
    clean_subjects = np.unique(clean_subjects)
    clean_subjects_videos_code = [
        element for element in clean_subjects_videos_code if element != []
    ]
    clean_subjects_videos_ground_truth_labels = [
        element
        for element in clean_subjects_videos_ground_truth_labels
        if element != []
    ]
    clean_videos_images = [videos_images[i] for i in required_videos_index]
    print("required_videos_index: ", required_videos_index)
    print("len(clean_videos_images) =", len(clean_videos_images))
    return (
        clean_videos_images,
        clean_subjects_videos_code,
        clean_subjects,
        clean_subjects_videos_ground_truth_labels,
    )


def calculate_k(clean_subjects_videos_ground_truth_labels):
    clean_subject_video_ground_truth_label_list = []
    for (
        clean_subject_videos_ground_truth_labels
    ) in clean_subjects_videos_ground_truth_labels:
        for (
            clean_subject_video_ground_truth_labels
        ) in clean_subject_videos_ground_truth_labels:
            for (
                clean_subject_video_ground_truth_label
            ) in clean_subject_video_ground_truth_labels:
                clean_subject_video_ground_truth_label_list.append(
                    clean_subject_video_ground_truth_label
                )

    total_duration = 0
    for (
        clean_subject_video_ground_truth_list_label_element
    ) in clean_subject_video_ground_truth_label_list:
        total_duration += (
            clean_subject_video_ground_truth_list_label_element[1]
            - clean_subject_video_ground_truth_list_label_element[0]
        )
    n = total_duration / len(clean_subject_video_ground_truth_label_list)
    k = int((n + 1) / 2)
    print("k (Half of average length of expression) = ", k)
    return k
