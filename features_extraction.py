import numpy as np
import pandas as pd
import cv2


def pol2cart(rho, phi):
    # Convert polar coordinates to cartesian coordinates for computation of optical strain
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
