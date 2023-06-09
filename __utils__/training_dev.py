import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
    dataset_dir,
    clean_subjects,
    image_size,
    y,
    expression_type,
    model_name,
    train_or_not,
    batch_size,
    epochs,
    preds_path,
    split_beginning=0,
):
    preds = []

    # Leave One Subject Out
    for split_index, split_clean_subject in enumerate(clean_subjects[split_beginning:]):
        split_index += split_beginning
        split = split_index + 1
        print(
            "Split {}/{} subject {} is in process.".format(
                split, len(clean_subjects), split_clean_subject
            )
        )
        # To reset the model at every LOSO testing
        print("------Initializing Model-------")
        model = models.load_model(model_name)
        # model = torch.compile(model)
        model = model.to(device)

        # get test dataset
        X_test = functions.load_subject_pkl_file(
            expression_type,
            dataset_dir,
            subject=split_clean_subject,
            image_size=image_size,
        )
        y_test = y[split_index]

        # test normalization
        X_test = functions.normalize(X_test)

        transform = transforms.Compose([transforms.ToTensor()])
        validation_ds = torch_functions.DS(X_test, y_test, transform)
        validation_dataloader = DataLoader(
            dataset=validation_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        for clean_subject_index, clean_subject in enumerate(clean_subjects):
            if clean_subject_index != split_index:
                # Get training set
                X_train = functions.load_subject_pkl_file(
                    expression_type,
                    dataset_dir,
                    subject=clean_subject,
                    image_size=image_size,
                )
                y_train = y[clean_subject_index]

                if train_or_not is True:
                    # downsampling
                    X_train, y_train = functions.downsample(X_train, y_train)

                    # Data augmentation to the micro-expression samples only
                    if expression_type == "me":
                        X_train, y_train = functions.augment_data(
                            X_train, y_train, image_size
                        )

                    # training normalization
                    X_train = functions.normalize(X_train)

                    training_ds = torch_functions.DS(X_train, y_train, transform)
                    training_dataloader = DataLoader(
                        dataset=training_ds,
                        batch_size=batch_size,
                        shuffle=True,
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
        functions.save_pred(split, clean_subjects, preds_path, pred)
        preds.append(pred)
        del X_test
        gc.collect()
        print(
            "Split {}/{} subject {} is processed.\n".format(
                split, len(clean_subjects), split_clean_subject
            )
        )

    return preds
