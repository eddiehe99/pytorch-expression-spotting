import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import LeaveOneGroupOut
import gc
from __utils__ import functions
from __utils__ import torch_functions
import models


print("torch Version: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(device))


def train(
    X,
    y,
    groups,
    image_size,
    expression_type,
    model_name,
    train_or_not,
    batch_size,
    epochs,
):
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    preds = []

    # Leave One Subject Out
    for split, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
        print(f"Split {split+1}/{n_splits} is in process.")

        # Get training set
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        # Get testing set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # To reset the model at every LOSO testing
        print("------Initializing Model-------")
        model = models.load_model(model_name)
        # model = torch.compile(model)
        model = model.to(device)

        if train_or_not is True:
            # downsampling
            X_train, y_train = functions.downsample(X_train, y_train)

            # Data augmentation to the micro-expression samples only
            if expression_type == "me":
                X_train, y_train = functions.augment_data(X_train, y_train, image_size)

            # normalization
            # cv2.normalize works better than tf.image
            X_train = functions.normalize(X_train)
            X_test = functions.normalize(X_test)

            transform = transforms.Compose([transforms.ToTensor()])
            training_ds = torch_functions.DS(X_train, y_train, transform)
            validation_ds = torch_functions.DS(X_test, y_test, transform)
            training_dataloader = DataLoader(
                dataset=training_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
            validation_dataloader = DataLoader(
                dataset=validation_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005)
            # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
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
                verbose=0,
                validation_dataloader=validation_dataloader,
            )
            del X_train
            gc.collect()
        else:
            # Load Pretrained Weights
            # model.load_weights(weights_path)
            pass

        pred = history.predict(validation_dataloader, model)
        preds.append(pred)
        del X_test
        gc.collect()
        print(f"Split {split+1}/{n_splits} is processed.\n")

    return preds
