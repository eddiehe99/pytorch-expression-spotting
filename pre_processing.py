import dlib
import cv2
import numpy as np


def pre_process(clean_videos_images, clean_videos_images_features, k, image_size=128):
    predictor_model = "__utils__/shape_predictor_68_face_landmarks.dat"
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
