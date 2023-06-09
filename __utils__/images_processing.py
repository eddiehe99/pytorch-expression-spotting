import os
import shutil
import glob
import natsort
import dlib
import numpy as np
import cv2
from pathlib import Path


def load_images(
    dataset_dir, images_loading=False, image_size=128, load_cropped_images=True
):
    videos_images = []
    subjects = []
    subjects_videos_code = []
    dataset_dir = Path(dataset_dir)

    if dataset_dir.name == "CAS(ME)^2":
        pattern = "cropped_rawpic/*" if load_cropped_images is True else "rawpic/*"
        for subject_dir in natsort.natsorted(dataset_dir.glob(pattern)):
            print("subject: ", subject_dir.name)
            subjects.append(subject_dir.name)
            subjects_videos_code.append([])
            for subject_video_dir in natsort.natsorted(subject_dir.glob("*")):
                # Ex:'CASME_sq/rawpic_aligned/s15/15_0101disgustingteeth' -> '0101'
                subjects_videos_code[-1].append(subject_video_dir.name[3:7])
                video_images = []
                for subject_video_image_path in natsort.natsorted(
                    subject_video_dir.glob("img*.jpg")
                ):
                    if images_loading is True:
                        # load the image
                        # cv2 needs a str
                        video_image = cv2.imread(str(subject_video_image_path), 0)
                        video_image = cv2.resize(video_image, (image_size, image_size))
                        video_images.append(video_image)
                    else:
                        # load the image path
                        video_images.append(subject_video_image_path)
                videos_images.append(video_images)

    elif dataset_dir.name == "SAMM_longvideos":
        pattern = (
            "cropped_SAMM_longvideos/*"
            if load_cropped_images is True
            else "SAMM_longvideos/*"
        )
        for subject_video_dir in natsort.natsorted(dataset_dir.glob(pattern)):
            print("subject_video: " + subject_video_dir.name)
            subject = str(subject_video_dir.name).split("_")[0]
            if subject not in subjects:  # Only append unique subject_dir name
                subjects.append(subject)
                subjects_videos_code.append([])
            subjects_videos_code[-1].append(str(subject_video_dir.name).split("_")[-1])

            video_images = []
            for subject_video_image_path in natsort.natsorted(
                subject_video_dir.glob("*.jpg")
            ):
                if images_loading is True:
                    # load the image
                    # cv2 needs a str
                    video_image = cv2.imread(str(subject_video_image_path), 0)
                    video_image = cv2.resize(video_image, (image_size, image_size))
                    video_images.append(video_image)
                else:
                    # load the image path
                    video_images.append(subject_video_image_path)
            videos_images.append(video_images)

    elif dataset_dir.name == "CAS_Test_cropped":
        if load_cropped_images is True:
            dataset_dir = Path(dataset_dir.parent) / "CAS_Test" / "cropped_CAS_Test"
        else:
            dataset_dir = Path(dataset_dir)
        for subject_dir in natsort.natsorted(dataset_dir.glob("*")):
            subjects.append(subject_dir.name)
            subjects_videos_code.append(subject_dir.name)
            print("test_subject_video_code: ", subject_dir.name)
            video_images = []
            # Read videos_images
            for subject_image_path in natsort.natsorted(subject_dir.glob("*.jpg")):
                if images_loading is True:
                    # load the image
                    # cv2 needs a str
                    video_image = cv2.imread(str(subject_image_path), 0)
                    video_image = cv2.resize(video_image, (image_size, image_size))
                    video_images.append(video_image)
                else:
                    # load the image path
                    video_images.append(subject_image_path)
            videos_images.append(video_images)

    elif dataset_dir.name == "SAMM_Test_cropped":
        if load_cropped_images is True:
            dataset_dir = Path(dataset_dir.parent) / "SAMM_Test" / "cropped_SAMM_Test"
        else:
            dataset_dir = Path(dataset_dir)
        for subject_dir in natsort.natsorted(dataset_dir.glob("*")):
            subjects.append(subject_dir.name.split("_")[0])
            subjects_videos_code.append(subject_dir.name)
            print("test_subject_video_code: ", subject_dir.name)
            video_images = []
            # Read videos_images
            for subject_image_path in natsort.natsorted(subject_dir.glob("*.jpg")):
                if images_loading is True:
                    # load the image
                    # cv2 needs a str
                    video_image = cv2.imread(str(subject_image_path), 0)
                    video_image = cv2.resize(video_image, (image_size, image_size))
                    video_images.append(video_image)
                else:
                    # load the image path
                    video_images.append(subject_image_path)
            videos_images.append(video_images)

    return videos_images, subjects, subjects_videos_code


def get_square_face(
    image,
    face_left,
    face_top,
    face_right,
    face_bottom,
):
    frame_height, frame_width, _ = image.shape
    face_height = face_bottom - face_top
    face_weight = face_right - face_left
    if face_weight < face_height:
        half_difference = (face_height - face_weight) // 2
        crop_left = (
            0 if face_left - half_difference < 0 else face_left - half_difference
        )
        crop_right = (
            frame_width
            if face_right + half_difference > frame_width
            else face_right + half_difference
        )
        face = image[face_top:face_bottom, crop_left:crop_right]
    else:
        # face_height < face_weight
        half_difference = (face_weight - face_height) // 2
        crop_top = 0 if face_top - half_difference < 0 else face_top - half_difference
        crop_bottom = (
            frame_height
            if face_bottom + half_difference > frame_height
            else face_bottom + half_difference
        )
        face = image[crop_top:crop_bottom, face_left:face_right]
    return face


def cv2_haar_cascade_face_detect(image):
    faceCascade = cv2.CascadeClassifier(
        "./__utils__/haarcascade_frontalface_default.xml"
    )
    faces = faceCascade.detectMultiScale(image)
    for face in faces:
        x1, y1, w, h = face
        x2 = x1 + w
        y2 = y1 + h
    face = get_square_face(
        image=image,
        face_left=x1,
        face_top=y1,
        face_right=x2,
        face_bottom=y2,
    )
    return face


def cv2_dnn_face_detect(image):
    # use DNN Face Detector in OpenCV
    model_file = "./__utils__/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "./__utils__/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    frame_height, frame_width, _ = image.shape
    conf_threshold = 0.5
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
    # print(f"cv2 dnn height = {y2-y1}, weight = {x2-x1}")
    face = get_square_face(
        image=image,
        face_left=x1,
        face_top=y1,
        face_right=x2,
        face_bottom=y2,
    )
    return face


def dlib_hog_face_detect(image):
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    for face_rect in detected_faces:
        x1 = face_rect.left()
        y1 = face_rect.top()
        x2 = face_rect.right()
        y2 = face_rect.bottom()
    face = get_square_face(
        image=image,
        face_left=x1,
        face_top=y1,
        face_right=x2,
        face_bottom=y2,
    )
    return face


def dlib_mmod_face_detect(image):
    # Convolutional Neural Network – CNN Face Detector in Dlib
    # the mmod_human_face_detector is rubbish!
    face_detector = dlib.cnn_face_detection_model_v1(
        "./__utils__/mmod_human_face_detector.dat"
    )
    detected_faces = face_detector(image, 1)
    for detected_face in detected_faces:
        x1 = detected_face.rect.left()
        y1 = detected_face.rect.top()
        x2 = detected_face.rect.right()
        y2 = detected_face.rect.bottom()
    face = get_square_face(
        image=image,
        face_left=x1,
        face_top=y1,
        face_right=x2,
        face_bottom=y2,
    )
    return face


# Save the videos_images into folder 'cropped_xxx'
def crop_images(dataset_dir):
    """
    https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    https://www.cnblogs.com/ssyfj/p/9286643.html
    """
    dataset_dir = Path(dataset_dir)

    if dataset_dir.name == "CAS(ME)^2":
        for subject_dir in natsort.natsorted(dataset_dir.glob("rawpic/*")):
            # Create new directory for 'cropped_rawpic'
            cropped_dir = dataset_dir / "cropped_rawpic"
            if cropped_dir.exists() == False:
                cropped_dir.mkdir()

            # Create new directory for each subject_dir
            cropped_subject_dir = dataset_dir / "cropped_rawpic" / subject_dir.name
            if cropped_subject_dir.exists() == False:
                cropped_subject_dir.mkdir()
            print(f"\nsubject: {subject_dir.name}")

            for subject_video_dir in natsort.natsorted(subject_dir.glob("*")):
                # Get dir of subject_video_dir
                cropped_subject_video_dir = cropped_subject_dir / subject_video_dir.name

                if cropped_subject_video_dir.exists() == False:
                    cropped_subject_video_dir.mkdir()

                # Read videos_images
                for subject_video_image_path in natsort.natsorted(
                    subject_video_dir.glob("img*.jpg")
                ):
                    print(
                        "subject: {}, video: {}, image: {}".format(
                            subject_dir.name,
                            subject_video_dir.name,
                            subject_video_image_path.name,
                        ),
                        end="\r",
                    )
                    # Get img num Ex 001,002,...,2021
                    image_filename = subject_video_image_path.name
                    image = cv2.imread(str(subject_video_image_path))

                    # use DNN Face Detector in OpenCV
                    face = cv2_dnn_face_detect(image)

                    # resize
                    # face = cv2.resize(face, (128, 128))
                    face = cv2.resize(face, (256, 256))

                    cv2.imwrite(str(cropped_subject_video_dir / image_filename), face)

    elif dataset_dir.name == "SAMM_longvideos":
        # create new dir
        if (dataset_dir / "cropped_SAMM_longvideos").exists() == False:
            (dataset_dir / "cropped_SAMM_longvideos").mkdir()

        for subject_video_dir in natsort.natsorted(
            dataset_dir.glob("SAMM_longvideos/*")
        ):
            cropped_subject_video_dir = (
                dataset_dir / "cropped_SAMM_longvideos" / subject_video_dir.name
            )

            # create new dir
            if cropped_subject_video_dir.exists() == False:
                cropped_subject_video_dir.mkdir()
            print(f"\nsubject_video: {subject_video_dir.name}", end="\r")

            for subject_video_image_path in natsort.natsorted(
                subject_video_dir.glob("*.jpg")
            ):
                print(
                    "subject_video: {}, image: {}".format(
                        subject_video_dir.name, subject_video_image_path.name
                    ),
                    end="\r",
                )
                # Get img num Ex 0001,0002,...,2021
                image_filename = subject_video_image_path.name
                image = cv2.imread(str(subject_video_image_path))

                # use DNN Face Detector in OpenCV
                face = cv2_dnn_face_detect(image)

                # resize
                # face = cv2.resize(face, (128, 128))
                face = cv2.resize(face, (256, 256))

                cv2.imwrite(str(cropped_subject_video_dir / image_filename), face)

    elif dataset_dir.name == "CAS_Test_cropped":
        # Create new directory for 'cropped_'
        cropped_dir = dataset_dir.parent / "CAS_Test" / "cropped_CAS_Test"
        if cropped_dir.exists() == False:
            cropped_dir.mkdir(parents=True)
        for subject_dir in natsort.natsorted(dataset_dir.glob("*")):
            # Create new directory for each subject_dir
            cropped_subject_dir = cropped_dir / subject_dir.name
            if cropped_subject_dir.exists() == False:
                cropped_subject_dir.mkdir()
            print(f"\nsubject: {subject_dir.name}", end="\r")

            # Read videos_images
            for subject_image_path in natsort.natsorted(subject_dir.glob("*.jpg")):
                print(
                    "subject: {}, image: {}".format(
                        subject_dir.name, subject_image_path.name
                    ),
                    end="\r",
                )
                # Get img num Ex 001,002,...,2021
                image_filename = subject_image_path.name
                image = cv2.imread(str(subject_image_path))

                # use DNN Face Detector in OpenCV
                face = cv2_dnn_face_detect(image)

                # resize
                # face = cv2.resize(face, (128, 128))
                face = cv2.resize(face, (256, 256))

                cv2.imwrite(str(cropped_subject_dir / image_filename), face)

    elif dataset_dir.name == "SAMM_Test_cropped":
        # Create new directory for 'cropped_'
        cropped_dir = dataset_dir.parent / "SAMM_Test" / "cropped_SAMM_Test"
        if cropped_dir.exists() == False:
            cropped_dir.mkdir(parents=True)
        for subject_video_dir in natsort.natsorted(dataset_dir.glob("*")):
            # Create new directory for each subject_dir
            cropped_subject_video_dir = cropped_dir / subject_video_dir.name
            if cropped_subject_video_dir.exists() == False:
                cropped_subject_video_dir.mkdir()
            print(f"\nsubject_video: {subject_video_dir.name}", end="\r")

            # Read videos_images
            for subject_video_image_path in natsort.natsorted(
                subject_video_dir.glob("*.jpg")
            ):
                print(
                    "subject_video: {}, image: {}".format(
                        subject_video_dir.name, subject_video_image_path.name
                    ),
                    end="\r",
                )
                # Get img num Ex 001,002,...,2021
                image_filename = subject_video_image_path.name
                image = cv2.imread(str(subject_video_image_path))

                # use DNN Face Detector in OpenCV
                face = cv2_dnn_face_detect(image)

                # resize
                # face = cv2.resize(face, (128, 128))
                face = cv2.resize(face, (256, 256))

                cv2.imwrite(str(cropped_subject_video_dir / image_filename), face)


def legacy_load_images(dataset_name):
    videos_images = []
    subjects = []
    subjects_videos_code = []

    if dataset_name == "D:/Databases/CAS(ME)^2":
        for i, subject_dir in enumerate(
            natsort.natsorted(glob.glob(dataset_name + "/cropped_rawpic/*"))
        ):
            print("Subject: " + subject_dir.split("/")[-1])
            subjects.append(subject_dir.split("/")[-1])
            subjects_videos_code.append([])
            for subject_video_dir in natsort.natsorted(glob.glob(subject_dir + "/*")):
                # Ex:'CASME_sq/rawpic_aligned/s15/15_0101disgustingteeth' -> '0101'
                subjects_videos_code[-1].append(
                    subject_video_dir.split("/")[-1].split("_")[1][:4]
                )
                video_images = []
                for subject_video_image_path in natsort.natsorted(
                    glob.glob(subject_video_dir + "/img*.jpg")
                ):
                    video_images.append(cv2.imread(subject_video_image_path, 0))
                videos_images.append(np.array(video_images))

    elif dataset_name == "SAMMLV":
        for i, subject_video_dir in enumerate(
            natsort.natsorted(glob.glob(dataset_name + "\\SAMM_longvideos_crop\\*"))
        ):
            print("Subject: " + subject_video_dir.split("\\")[-1].split("_")[0])
            subject = subject_video_dir.split("\\")[-1].split("_")[0]
            if subject not in subjects:  # Only append unique subject_dir name
                subjects.append(subject_dir)
                subjects_videos_code.append([])
            subjects_videos_code[-1].append(subject_video_dir.split("\\")[-1])

            video_images = []
            for subject_video_image_path in natsort.natsorted(
                glob.glob(subject_video_dir + "\\*.jpg")
            ):
                video_images.append(cv2.imread(subject_video_image_path, 0))
            videos_images.append(np.array(video_images))

    return videos_images, subjects, subjects_videos_code


# Save the videos_images into folder 'cropped_xxx'
def legacy_crop_images(dataset_name):
    face_detector = dlib.get_frontal_face_detector()
    # the mmod_human_face_detector is rubbish!
    # face_detector = dlib.cnn_face_detection_model_v1(
    #     "mmod_human_face_detector.dat"
    # )
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    if dataset_name == "D:/Databases/CAS(ME)^2":
        for subject_dir in glob.glob(dataset_name + "/rawpic/*"):
            subject_videos_dir = (
                dataset_name + "/rawpic/" + str(subject_dir.split("/")[-1]) + "/*"
            )

            # Create new directory for 'cropped_rawpic'
            cropped_dir = dataset_name + "/cropped_rawpic/"
            if os.path.exists(cropped_dir) == False:
                os.mkdir(cropped_dir)

            # Create new directory for each subject_dir
            cropped_subject_dir = (
                dataset_name
                + "/cropped_rawpic/"
                + str(subject_dir.split("/")[-1])
                + "/"
            )
            if os.path.exists(cropped_subject_dir):
                shutil.rmtree(cropped_subject_dir)
            os.mkdir(cropped_subject_dir)
            print("Subject: ", subject_dir.split("/")[-1])

            for subject_video_dir in glob.glob(subject_videos_dir):
                # Get dir of subject_video_dir
                cropped_subject_video_dir = (
                    cropped_subject_dir + subject_video_dir.split("/")[-1]
                )
                if os.path.exists(cropped_subject_video_dir):
                    shutil.rmtree(cropped_subject_video_dir)
                os.mkdir(cropped_subject_video_dir)

                # Read videos_images
                for subject_video_image_path in natsort.natsorted(
                    glob.glob(subject_video_dir + "/img*.jpg")
                ):
                    # Get img num Ex 001,002,...,2021
                    image_filename = subject_video_image_path.split("/")[-1]
                    frame = image_filename[3:-4]

                    image = cv2.imread(subject_video_image_path)
                    detected_faces = face_detector(image, 1)

                    # Use the first frame as the reference frame
                    if frame == "001":
                        for face_rect in detected_faces:
                            face_top = face_rect.top()
                            face_bottom = face_rect.bottom()
                            face_left = face_rect.left()
                            face_right = face_rect.right()
                    face = image[face_top:face_bottom, face_left:face_right]
                    face = cv2.resize(face, (128, 128))
                    # face = cv2.resize(face, (256, 256))

                    cv2.imwrite(
                        cropped_subject_video_dir + "/img{}.jpg".format(frame), face
                    )

    elif dataset_name == "I:/HEH/Databases/SAMM_longvideos":
        # Delete dir if exist and create new dir
        if os.path.exists(dataset_name + "\\SAMM_longvideos_crop"):
            shutil.rmtree(dataset_name + "\\SAMM_longvideos_crop")
        os.mkdir(dataset_name + "\\SAMM_longvideos_crop")

        for subject_video_dir in glob.glob(dataset_name + "\\SAMM_longvideos\\*"):
            frame = 0
            cropped_subject_video_dir = (
                dataset_name
                + "\\cropped_SAMM_longvideos\\"
                + subject_video_dir.split("\\")[-1]
            )

            # Delete dir if exist and create new dir
            if os.path.exists(cropped_subject_video_dir):
                shutil.rmtree(cropped_subject_video_dir)
            os.mkdir(cropped_subject_video_dir)
            print("subject_video: ", subject_video_dir.split("\\")[-1])

            for subject_video_image_path in natsort.natsorted(
                glob.glob(subject_video_dir + "\\*.jpg")
            ):
                # Get img num Ex 0001,0002,...,2021
                image_filename = subject_video_image_path.split("\\")[-1].split(".")[0]
                frame = image_filename[-4:]

                # Run the HOG face detector on the image data
                image = cv2.imread(subject_video_image_path)
                detected_faces = face_detector(image, 1)

                # Loop through each face we found in the image
                if frame == "0001":  # Use first frame as reference frame
                    for i, face_rect in enumerate(detected_faces):
                        face_top = face_rect.top()
                        face_bottom = face_rect.bottom()
                        face_left = face_rect.left()
                        face_right = face_rect.right()
                face = image[face_top:face_bottom, face_left:face_right]
                face = cv2.resize(face, (128, 128))

                cv2.imwrite(cropped_subject_video_dir + "\\{}.jpg".format(frame), face)
