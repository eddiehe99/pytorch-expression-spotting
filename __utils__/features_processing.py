import dlib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import natsort
import gc
import _pickle


def pol2cart(rho, phi):
    # Convert polar coordinates to cartesian coordinates
    # for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def compute_optical_strain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis="columns")
    v_y = v - pd.DataFrame(v).shift(-1, axis="index")
    u_y = u - pd.DataFrame(u).shift(-1, axis="index")
    v_x = v - pd.DataFrame(v).shift(-1, axis="columns")
    optical_strain = np.array(
        np.sqrt(u_x**2 + v_y**2 + 1 / 2 * (u_y + v_x) ** 2)
        .ffill(axis="columns")
        .ffill(axis="index")
    )
    return optical_strain


def extract_features_video_images(video_images, image_size, k, **kwargs):
    video_images_features = []
    video_name = kwargs["video_name"]
    for clean_video_image_index in range(len(video_images) - k):
        print(
            f"extracting features from video: {video_name},",
            f"image: {clean_video_image_index}",
            end="\r",
        )

        image1 = video_images[clean_video_image_index]
        image2 = video_images[clean_video_image_index + k]

        if type(image1).__name__ != "ndarray":
            # cv2 needs a str
            image1 = cv2.imread(str(image1), 0)
            image2 = cv2.imread(str(image2), 0)
            if image_size == 128:
                image1 = cv2.resize(image1, (image_size, image_size))
                image2 = cv2.resize(image2, (image_size, image_size))
            elif image_size == 256:
                pass

        """
        https://docs.opencv.org/4.7.0/df/dde/classcv_1_1DenseOpticalFlow.html
        """

        # Compute Optical Flow Features
        # optical_flow = cv2.DualTVL1OpticalFlow_create() #Depends on cv2 version
        optical_flow_create = cv2.optflow.DualTVL1OpticalFlow_create()
        optical_flow = optical_flow_create.calc(image1, image2, None)
        u, v = optical_flow[..., 0], optical_flow[..., 1]
        optical_strain = compute_optical_strain(u, v)

        # calculate opical flow features using Gunnar Farneback algorithm
        # optical_flow = cv2.calcOpticalFlowFarneback(
        #     image1, image2, None, 0.5, 3, 15, 5, 7, 1.5, 0
        # )

        # Features Concatenation into 128x128x3
        # clean_video_image_features = np.zeros((128, 128, 3))
        video_image_features = np.zeros((image_size, image_size, 3))
        video_image_features[:, :, 0] = u
        video_image_features[:, :, 1] = v
        video_image_features[:, :, 2] = optical_strain

        video_images_features.append(video_image_features)

    return video_images_features


def pre_process_video_images(
    video_images, image_size, k, video_images_features, **kwargs
):
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    resampled_video_images_features = []
    video_name = kwargs["video_name"]
    for video_image_index in range(len(video_images) - k):
        # Loop through the frames until all the landmark is detected
        # if clean_video_image_index == 0:
        reference_image = video_images[video_image_index]
        if type(reference_image).__name__ != "ndarray":
            reference_image = cv2.imread(str(reference_image), 0)
        if reference_image.shape[0] != image_size:
            reference_image = cv2.resize(reference_image, (image_size, image_size))
        detect = face_detector(reference_image, 1)
        index_shift = 0
        print(
            f"pre-processing video: {video_name},",
            f"image: {video_image_index},",
            end="\r",
        )
        while len(detect) == 0:
            # when a video is processed, get the backup image first
            if video_image_index == 0:
                if video_image_index + index_shift < len(video_images):
                    reference_image = video_images[video_image_index + index_shift]
                else:
                    # when index exceed the len
                    # extremely less likely to happen
                    # if happens, it means the detect fail on all images of this video
                    # and the backup_image is from the last video
                    reference_image = backup_image
            else:
                if (
                    video_image_index + index_shift < len(video_images)
                    and index_shift < (k * 2) + 1
                ):
                    reference_image = video_images[video_image_index + index_shift]
                else:
                    # when index exceed the len
                    reference_image = backup_image
            if type(reference_image).__name__ != "ndarray":
                reference_image = cv2.imread(str(reference_image), 0)
            if reference_image.shape[0] != image_size:
                reference_image = cv2.resize(reference_image, (image_size, image_size))
            detect = face_detector(reference_image, 1)
            index_shift += 1
            print(
                f"pre-processing clean_video: {video_name},",
                f"image: {video_image_index},",
                f"index_shift: {index_shift}",
                end="\r",
            )
        backup_image = reference_image
        if type(reference_image).__name__ != "ndarray":
            reference_image = cv2.imread(str(reference_image), 0)
        if reference_image.shape[0] != image_size:
            reference_image = cv2.resize(reference_image, (image_size, image_size))
        shape = face_pose_predictor(reference_image, detect[0])

        # Left Eye
        x11 = max(shape.part(36).x - 15, 0)
        y11 = shape.part(36).y
        x12 = shape.part(37).x
        y12 = max(shape.part(37).y - 15, 0)
        x13 = shape.part(38).x
        y13 = max(shape.part(38).y - 15, 0)
        # x14 = min(shape.part(39).x + 15, 128)
        x14 = min(shape.part(39).x + 15, image_size)
        y14 = shape.part(39).y
        x15 = shape.part(40).x
        # y15 = min(shape.part(40).y + 15, 128)
        y15 = min(shape.part(40).y + 15, image_size)
        x16 = shape.part(41).x
        # y16 = min(shape.part(41).y + 15, 128)
        y16 = min(shape.part(41).y + 15, image_size)

        # Right Eye
        x21 = max(shape.part(42).x - 15, 0)
        y21 = shape.part(42).y
        x22 = shape.part(43).x
        y22 = max(shape.part(43).y - 15, 0)
        x23 = shape.part(44).x
        y23 = max(shape.part(44).y - 15, 0)
        # x24 = min(shape.part(45).x + 15, 128)
        x24 = min(shape.part(45).x + 15, image_size)
        y24 = shape.part(45).y
        x25 = shape.part(46).x
        # y25 = min(shape.part(46).y + 15, 128)
        y25 = min(shape.part(46).y + 15, image_size)
        x26 = shape.part(47).x
        # y26 = min(shape.part(47).y + 15, 128)
        y26 = min(shape.part(47).y + 15, image_size)

        # ROI 1 (Left Eyebrow)
        x31 = max(shape.part(17).x - 12, 0)
        y32 = max(shape.part(19).y - 12, 0)
        # x33 = min(shape.part(21).x + 12, 128)
        x33 = min(shape.part(21).x + 12, image_size)
        # y34 = min(shape.part(41).y + 12, 128)
        y34 = min(shape.part(41).y + 12, image_size)

        # ROI 2 (Right Eyebrow)
        x41 = max(shape.part(22).x - 12, 0)
        y42 = max(shape.part(24).y - 12, 0)
        # x43 = min(shape.part(26).x + 12, 128)
        x43 = min(shape.part(26).x + 12, image_size)
        # y44 = min(shape.part(46).y + 12, 128)
        y44 = min(shape.part(46).y + 12, image_size)

        # ROI 3 #Mouth
        x51 = max(shape.part(60).x - 12, 0)
        y52 = max(shape.part(50).y - 12, 0)
        # x53 = min(shape.part(64).x + 12, 128)
        x53 = min(shape.part(64).x + 12, image_size)
        # y54 = min(shape.part(57).y + 12, 128)
        y54 = min(shape.part(57).y + 12, image_size)

        # Nose landmark
        x61 = shape.part(28).x
        y61 = shape.part(28).y

        # Load video image features
        video_image_features = video_images_features[video_image_index]

        # Remove global head movement by minus nose region
        video_image_features[:, :, 0] = abs(
            video_image_features[:, :, 0]
            - video_image_features[y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 0].mean()
        )
        video_image_features[:, :, 1] = abs(
            video_image_features[:, :, 1]
            - video_image_features[y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 1].mean()
        )
        video_image_features[:, :, 2] = (
            video_image_features[:, :, 2]
            - video_image_features[y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 2].mean()
        )

        # Eye masking
        left_eye = [
            (x11, y11),
            (x12, y12),
            (x13, y13),
            (x14, y14),
            (x15, y15),
            (x16, y16),
        ]
        right_eye = [
            (x21, y21),
            (x22, y22),
            (x23, y23),
            (x24, y24),
            (x25, y25),
            (x26, y26),
        ]
        cv2.fillPoly(video_image_features, [np.array(left_eye)], 0)
        cv2.fillPoly(video_image_features, [np.array(right_eye)], 0)

        if image_size == 128:
            # ROI Selection -> Image resampling into 42x22x3
            # cv2 resize to (weight, height)
            resampled_video_image_features = np.zeros((42, 42, 3))
            resampled_video_image_features[:21, :, :] = cv2.resize(
                video_image_features[min(y32, y42) : max(y34, y44), x31:x43, :],
                (42, 21),
            )
            resampled_video_image_features[21:42, :, :] = cv2.resize(
                video_image_features[y52:y54, x51:x53, :], (42, 21)
            )
        elif image_size == 256:
            # ROI Selection -> Image resampling into [84, 84, 3]
            # cv2 resize to (weight, height)
            resampled_video_image_features = np.zeros((84, 84, 3))
            resampled_video_image_features[:42, :, :] = cv2.resize(
                video_image_features[min(y32, y42) : max(y34, y44), x31:x43, :],
                (84, 42),
            )
            resampled_video_image_features[42:84, :, :] = cv2.resize(
                video_image_features[y52:y54, x51:x53, :], (84, 42)
            )
        resampled_video_images_features.append(resampled_video_image_features)

    return resampled_video_images_features


def extract_features_and_pre_process(
    clean_videos_images,
    k,
    expression_type,
    clean_subjects,
    clean_subjects_videos_code,
    dataset_dir,
    image_size=128,
):
    finished_videos_len = 0
    # Every subjects
    for clean_subject_videos_code_index, clean_subject_videos_code in enumerate(
        clean_subjects_videos_code
    ):
        clean_videos_images_start_index = finished_videos_len
        clean_videos_images_end_index = finished_videos_len + len(
            clean_subject_videos_code
        )
        resampled_clean_subject_videos_images_features = []
        # Every videos of the subject
        for clean_video_images_index, clean_video_images in enumerate(
            clean_videos_images[
                clean_videos_images_start_index:clean_videos_images_end_index
            ]
        ):
            video_name = "{}_{}".format(
                clean_subjects[clean_subject_videos_code_index],
                clean_subject_videos_code[clean_video_images_index],
            )
            kwargs = {"video_name": video_name}

            """extract features"""
            clean_video_images_features = extract_features_video_images(
                video_images=clean_video_images,
                image_size=image_size,
                k=k,
                **kwargs,
            )
            # print(
            #     "\nFinish extract features from video:",
            #     video_name,
            # )

            """pre-process"""
            resampled_clean_video_images_features = pre_process_video_images(
                video_images=clean_video_images,
                image_size=image_size,
                k=k,
                video_images_features=clean_video_images_features,
                **kwargs,
            )
            # print(
            #     "\nFinish pre-process video:",
            #     video_name,
            # )

            # Merge every videos of one subject into one list
            resampled_clean_subject_videos_images_features += (
                resampled_clean_video_images_features
            )

            del clean_video_images_features, resampled_clean_video_images_features
            gc.collect()

        # Update the finished_videos_len
        finished_videos_len += len(clean_subject_videos_code)

        """save pkl files"""
        Path(dataset_dir, f"{expression_type}_clean_subjects_x_{image_size}").mkdir(
            parents=True, exist_ok=True
        )
        subject_pkl_file_path = Path(
            dataset_dir,
            f"{expression_type}_clean_subjects_x_{image_size}",
            clean_subjects[clean_subject_videos_code_index],
        ).with_suffix(".pkl")
        print(f"subject_pkl_file_path: {subject_pkl_file_path}")
        with open(subject_pkl_file_path, "wb") as pkl_file:
            _pickle.dump(resampled_clean_subject_videos_images_features, pkl_file)
            pkl_file.close()
        del resampled_clean_subject_videos_images_features
        gc.collect()

    print("All features extracted and pre-processed.")


def extract_features_and_pre_process_test(
    clean_videos_images,
    k,
    expression_type,
    test_dataset_dir,
    image_size=128,
):
    test_videos_name = []
    test_dataset_dir = Path(test_dataset_dir)
    for test_video in natsort.natsorted(test_dataset_dir.iterdir()):
        test_videos_name.append(test_video.name)

    for clean_video_images_index, clean_video_images in enumerate(clean_videos_images):
        video_name = "{}".format(
            test_videos_name[clean_video_images_index],
        )
        kwargs = {"video_name": video_name}
        """extract features"""
        clean_video_images_features = extract_features_video_images(
            video_images=clean_video_images,
            image_size=image_size,
            k=k,
            **kwargs,
        )
        print(
            "\nFinish extract features from required_video_index ",
            clean_video_images_index,
        )

        """pre-process"""
        resampled_clean_video_images_features = pre_process_video_images(
            video_images=clean_video_images,
            image_size=image_size,
            k=k,
            video_images_features=clean_video_images_features,
            **kwargs,
        )
        print(
            "\nFinish pre-process clean_video_images_index ", clean_video_images_index
        )

        """save pkl files"""
        (
            test_dataset_dir.parent
            / Path(
                test_dataset_dir.name.split("_")[0]
                + "_"
                + test_dataset_dir.name.split("_")[1],
                f"{expression_type}_videos_x_{image_size}",
            )
        ).mkdir(parents=True, exist_ok=True)
        test_video_pkl_file_path = test_dataset_dir.parent / Path(
            test_dataset_dir.name.split("_")[0]
            + "_"
            + test_dataset_dir.name.split("_")[1],
            f"{expression_type}_videos_x_{image_size}",
            test_videos_name[clean_video_images_index],
        ).with_suffix(".pkl")
        print(f"test_video_pkl_file_path: {test_video_pkl_file_path}")
        with open(test_video_pkl_file_path, "wb") as pkl_file:
            _pickle.dump(resampled_clean_video_images_features, pkl_file)
            pkl_file.close()

    print("All features extracted and pre-processed.")


def extract_features(clean_videos_images, k, image_size=128):
    clean_videos_images_features = []
    for clean_video_images_index in range(len(clean_videos_images)):
        clean_video_images_features = []
        for clean_video_image_index in range(
            len(clean_videos_images[clean_video_images_index]) - k
        ):
            print(
                f"extracting features from clean_video {clean_video_images_index},",
                f"image {clean_video_image_index}",
                end="\r",
            )

            image1 = clean_videos_images[clean_video_images_index][
                clean_video_image_index
            ]
            image2 = clean_videos_images[clean_video_images_index][
                clean_video_image_index + k
            ]

            if type(image1).__name__ != "ndarray":
                # cv2 needs a str
                image1 = cv2.imread(str(image1), 0)
                image2 = cv2.imread(str(image2), 0)
                if image_size == 128:
                    image1 = cv2.resize(image1, (image_size, image_size))
                    image2 = cv2.resize(image2, (image_size, image_size))
                elif image_size == 256:
                    pass

            """
            https://docs.opencv.org/4.7.0/df/dde/classcv_1_1DenseOpticalFlow.html
            """

            # Compute Optical Flow Features
            # optical_flow = cv2.DualTVL1OpticalFlow_create() #Depends on cv2 version
            optical_flow_create = cv2.optflow.DualTVL1OpticalFlow_create()
            optical_flow = optical_flow_create.calc(image1, image2, None)
            u, v = optical_flow[..., 0], optical_flow[..., 1]
            optical_strain = compute_optical_strain(u, v)

            # calculate opical flow features using Gunnar Farneback algorithm
            # optical_flow = cv2.calcOpticalFlowFarneback(
            #     image1, image2, None, 0.5, 3, 15, 5, 7, 1.5, 0
            # )

            # Features Concatenation into 128x128x3
            # clean_video_image_features = np.zeros((128, 128, 3))
            clean_video_image_features = np.zeros((image_size, image_size, 3))
            clean_video_image_features[:, :, 0] = u
            clean_video_image_features[:, :, 1] = v
            clean_video_image_features[:, :, 2] = optical_strain

            clean_video_images_features.append(clean_video_image_features)

        clean_videos_images_features.append(clean_video_images_features)
        print(
            "\nFinish extract features form required_video_index ",
            clean_video_images_index,
        )
    print("All features extracted.")
    return clean_videos_images_features


def pre_process(clean_videos_images, clean_videos_images_features, k, image_size=128):
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    resampled_clean_videos_images_features = []
    for clean_video_images_index, clean_video_images in enumerate(clean_videos_images):
        resampled_clean_video_images_features = []
        for clean_video_image_index in range(len(clean_video_images) - k):
            # Loop through the frames until all the landmark is detected
            # if clean_video_image_index == 0:
            reference_image = clean_video_images[clean_video_image_index]
            if type(reference_image).__name__ != "ndarray":
                reference_image = cv2.imread(str(reference_image), 0)
            if reference_image.shape[0] != image_size:
                reference_image = cv2.resize(reference_image, (image_size, image_size))
            detect = face_detector(reference_image, 1)
            index_shift = 0
            print(
                f"pre-processing clean_video {clean_video_images_index},",
                f"image {clean_video_image_index}",
                end="\r",
            )
            while len(detect) == 0:
                # when a video is processed, get the backup image first
                if clean_video_image_index == 0:
                    if clean_video_image_index + index_shift < len(clean_video_images):
                        reference_image = clean_video_images[
                            clean_video_image_index + index_shift
                        ]
                    else:
                        # when index exceed the len
                        # extremely less likely to happen
                        # if happens, it means the detect fail on all images of this video
                        # and the backup_image is from the last video
                        reference_image = backup_image
                else:
                    if (
                        clean_video_image_index + index_shift < len(clean_video_images)
                        and index_shift < (k * 2) + 1
                    ):
                        reference_image = clean_video_images[
                            clean_video_image_index + index_shift
                        ]
                    else:
                        # when index exceed the len
                        reference_image = backup_image
                if type(reference_image).__name__ != "ndarray":
                    reference_image = cv2.imread(str(reference_image), 0)
                if reference_image.shape[0] != image_size:
                    reference_image = cv2.resize(
                        reference_image, (image_size, image_size)
                    )
                detect = face_detector(reference_image, 1)
                index_shift += 1
                print(
                    f"pre-processing clean_video {clean_video_images_index},",
                    f"image {clean_video_image_index},",
                    f"index_shift {index_shift}",
                    end="\r",
                )
            backup_image = reference_image
            if type(reference_image).__name__ != "ndarray":
                reference_image = cv2.imread(str(reference_image), 0)
            if reference_image.shape[0] != image_size:
                reference_image = cv2.resize(reference_image, (image_size, image_size))
            shape = face_pose_predictor(reference_image, detect[0])

            # Left Eye
            x11 = max(shape.part(36).x - 15, 0)
            y11 = shape.part(36).y
            x12 = shape.part(37).x
            y12 = max(shape.part(37).y - 15, 0)
            x13 = shape.part(38).x
            y13 = max(shape.part(38).y - 15, 0)
            # x14 = min(shape.part(39).x + 15, 128)
            x14 = min(shape.part(39).x + 15, image_size)
            y14 = shape.part(39).y
            x15 = shape.part(40).x
            # y15 = min(shape.part(40).y + 15, 128)
            y15 = min(shape.part(40).y + 15, image_size)
            x16 = shape.part(41).x
            # y16 = min(shape.part(41).y + 15, 128)
            y16 = min(shape.part(41).y + 15, image_size)

            # Right Eye
            x21 = max(shape.part(42).x - 15, 0)
            y21 = shape.part(42).y
            x22 = shape.part(43).x
            y22 = max(shape.part(43).y - 15, 0)
            x23 = shape.part(44).x
            y23 = max(shape.part(44).y - 15, 0)
            # x24 = min(shape.part(45).x + 15, 128)
            x24 = min(shape.part(45).x + 15, image_size)
            y24 = shape.part(45).y
            x25 = shape.part(46).x
            # y25 = min(shape.part(46).y + 15, 128)
            y25 = min(shape.part(46).y + 15, image_size)
            x26 = shape.part(47).x
            # y26 = min(shape.part(47).y + 15, 128)
            y26 = min(shape.part(47).y + 15, image_size)

            # ROI 1 (Left Eyebrow)
            x31 = max(shape.part(17).x - 12, 0)
            y32 = max(shape.part(19).y - 12, 0)
            # x33 = min(shape.part(21).x + 12, 128)
            x33 = min(shape.part(21).x + 12, image_size)
            # y34 = min(shape.part(41).y + 12, 128)
            y34 = min(shape.part(41).y + 12, image_size)

            # ROI 2 (Right Eyebrow)
            x41 = max(shape.part(22).x - 12, 0)
            y42 = max(shape.part(24).y - 12, 0)
            # x43 = min(shape.part(26).x + 12, 128)
            x43 = min(shape.part(26).x + 12, image_size)
            # y44 = min(shape.part(46).y + 12, 128)
            y44 = min(shape.part(46).y + 12, image_size)

            # ROI 3 #Mouth
            x51 = max(shape.part(60).x - 12, 0)
            y52 = max(shape.part(50).y - 12, 0)
            # x53 = min(shape.part(64).x + 12, 128)
            x53 = min(shape.part(64).x + 12, image_size)
            # y54 = min(shape.part(57).y + 12, 128)
            y54 = min(shape.part(57).y + 12, image_size)

            # Nose landmark
            x61 = shape.part(28).x
            y61 = shape.part(28).y

            # Load video image features
            clean_video_image_features = clean_videos_images_features[
                clean_video_images_index
            ][clean_video_image_index]

            # Remove global head movement by minus nose region
            clean_video_image_features[:, :, 0] = abs(
                clean_video_image_features[:, :, 0]
                - clean_video_image_features[
                    y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 0
                ].mean()
            )
            clean_video_image_features[:, :, 1] = abs(
                clean_video_image_features[:, :, 1]
                - clean_video_image_features[
                    y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 1
                ].mean()
            )
            clean_video_image_features[:, :, 2] = (
                clean_video_image_features[:, :, 2]
                - clean_video_image_features[
                    y61 - 5 : y61 + 6, x61 - 5 : x61 + 6, 2
                ].mean()
            )

            # Eye masking
            left_eye = [
                (x11, y11),
                (x12, y12),
                (x13, y13),
                (x14, y14),
                (x15, y15),
                (x16, y16),
            ]
            right_eye = [
                (x21, y21),
                (x22, y22),
                (x23, y23),
                (x24, y24),
                (x25, y25),
                (x26, y26),
            ]
            cv2.fillPoly(clean_video_image_features, [np.array(left_eye)], 0)
            cv2.fillPoly(clean_video_image_features, [np.array(right_eye)], 0)

            if image_size == 128:
                # ROI Selection -> Image resampling into 42x22x3
                # cv2 resize to (weight, height)
                resampled_clean_video_image_features = np.zeros((42, 42, 3))
                resampled_clean_video_image_features[:21, :, :] = cv2.resize(
                    clean_video_image_features[
                        min(y32, y42) : max(y34, y44), x31:x43, :
                    ],
                    (42, 21),
                )
                resampled_clean_video_image_features[21:42, :, :] = cv2.resize(
                    clean_video_image_features[y52:y54, x51:x53, :], (42, 21)
                )
                resampled_clean_video_images_features.append(
                    resampled_clean_video_image_features
                )
            elif image_size == 256:
                # ROI Selection -> Image resampling into [112, 112, 3]
                # cv2 resize to (weight, height)
                resampled_clean_video_image_features = np.zeros((112, 112, 3))
                resampled_clean_video_image_features[:56, :, :] = cv2.resize(
                    clean_video_image_features[
                        min(y32, y42) : max(y34, y44), x31:x43, :
                    ],
                    (112, 56),
                )
                resampled_clean_video_image_features[56:112, :, :] = cv2.resize(
                    clean_video_image_features[y52:y54, x51:x53, :], (112, 56)
                )
                resampled_clean_video_images_features.append(
                    resampled_clean_video_image_features
                )

        resampled_clean_videos_images_features.append(
            resampled_clean_video_images_features
        )
        print(
            "\nFinish pre-process clean_video_images_index ", clean_video_images_index
        )
    print("All pre-process done.")
    return resampled_clean_videos_images_features
