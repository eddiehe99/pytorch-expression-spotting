import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gc
import functions
import torch_functions
import models


print("torch Version: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(device))


def train_and_test(
    X,
    image_size,
    y,
    test_dataset_dir,
    test_videos_name,
    expression_type,
    model_name,
    train_or_not,
    batch_size,
    epochs,
):
    preds = []
    print("------Initializing Model-------")
    model = models.load_model(model_name)
    # model = torch.compile(model)
    model = model.to(device)

    print("------Training-------")
    X_train = X
    y_train = y
    if train_or_not is True:
        # downsampling
        X_train, y_train = functions.downsample(X_train, y_train)

        # Data augmentation to the micro-expression samples only
        if expression_type == "me":
            X_train, y_train = functions.augment_data(X_train, y_train)

        # normalization
        # cv2.normalize works better than tf.image
        X_train = functions.normalize(X_train)

        transform = transforms.Compose([transforms.ToTensor()])
        training_ds = torch_functions.DS(X_train, y_train, transform)
        training_dataloader = DataLoader(
            dataset=training_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005)
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01, weight_decay=0.0001)

        loss_fn = nn.MSELoss()
        loss_fn = loss_fn.to(device)

        history = torch_functions.History(device)
        history.fit_model(
            model=model,
            training_dataloader=training_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
        )
        del X_train
        gc.collect()
    else:
        # Load Pretrained Weights
        # model.load_weights(weights_path)
        pass

    print("------Test-------")
    for test_video_name in test_videos_name:
        print(f"Testing {test_video_name}")
        # Get test set
        X_test = functions.load_test_video_pkl_file(
            expression_type,
            test_dataset_dir,
            test_video_name,
            image_size=image_size,
        )

        # Normalization
        # cv2.normalize works better than tf.image
        X_test = functions.normalize(X_test)

        test_ds = torch_functions.Test_DS(X_test, transform)
        test_dataloader = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        pred = history.predict_test(test_dataloader, model)
        preds.append(pred)
        del X_test
        gc.collect()
        print(f"Finish testing {test_video_name}")

    print("All training and test are done.")
    return preds
