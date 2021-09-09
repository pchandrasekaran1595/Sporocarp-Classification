import os
import re
import cv2
import torch
import imgaug
import numpy as np
import matplotlib.pyplot as plt

from termcolor import colored
from imgaug import augmenters
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import train_test_split
os.system("color")

#####################################################################################################

class DS(Dataset):
    def __init__(self, X=None, y=None, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform

        assert(re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE) or re.match(r"test", self.mode, re.IGNORECASE))

        self.X = X
        if re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE):
            self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE):
            return self.transform(self.X[idx]), torch.LongTensor(self.y[idx])
        else:
            return self.transform(self.X[idx])

#####################################################################################################

def myprint(text: str, color: str) -> None:
    print(colored(text=text, color=color))


def breaker(num=50, char="*") -> None:
    myprint("\n" + num*char + "\n", "magenta")


def debug(text: str):
    myprint(text, "red")

#####################################################################################################

def get_augment(seed: int):
    imgaug.seed(seed)
    augment = augmenters.SomeOf(None, [
        augmenters.HorizontalFlip(p=0.5),
        augmenters.VerticalFlip(p=0.5),
        augmenters.Affine(scale=(0.75, 1.25), translate_percent=(-0.1, 0.1), rotate=(-45, 45), seed=seed),
    ], seed=seed)

    return augment


def read_image(name: str) -> np.ndarray:
    image = cv2.imread(os.path.join(TEST_DATA_PATH, name), cv2.IMREAD_COLOR)
    assert(image is not None)
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)


def downscale(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)


def show(image: np.ndarray, title=None) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def get_images(path: str, size: int) -> np.ndarray:
    images = np.zeros((len(os.listdir(path)), size, size, 3)).astype("uint8")
    i = 0
    for name in os.listdir(path):
        image = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
        images[i] = image
        i += 1
    return images


def get_data(base_path: str):
    folders = os.listdir(base_path)
    images_1 = get_images(os.path.join(base_path, folders[0]), size=SIZE)
    images_2 = get_images(os.path.join(base_path, folders[1]), size=SIZE)
    images_3 = get_images(os.path.join(base_path, folders[2]), size=SIZE)
    images_4 = get_images(os.path.join(base_path, folders[3]), size=SIZE)
    labels_1 = np.zeros((images_1.shape[0], 1))
    labels_2 = np.ones((images_2.shape[0], 1))
    labels_3 = np.ones((images_3.shape[0], 1)) * 2
    labels_4 = np.ones((images_4.shape[0], 1)) * 3

    images = np.concatenate((images_1, images_2, images_3, images_4), axis=0)
    labels = np.concatenate((labels_1, labels_2, labels_3, labels_4), axis=0)

    return images, labels


def build_dataloaders(path: str, batch_size: int, pretrained=False, do_augment=False):
    breaker()
    myprint("Fetching images and labels ...", "yellow")
    images, labels = get_data(path)

    breaker()
    myprint("Splitting into train and validation sets ...", "yellow")
    tr_images, va_images, tr_labels, va_labels = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=SEED, stratify=labels)

    if do_augment:
        breaker()
        myprint("Augmenting training set ...", "yellow")
        augment = get_augment(SEED)
        tr_images = augment(images=tr_images)

    breaker()
    myprint("Building Dataloaders ...", "yellow")
    if pretrained:
        tr_data_setup = DS(X=tr_images, y=tr_labels, transform=TRANSFORM_1, mode="train")
        va_data_setup = DS(X=va_images, y=va_labels, transform=TRANSFORM_1, mode="valid")
    else:
        tr_data_setup = DS(X=tr_images, y=tr_labels, transform=TRANSFORM_2, mode="train")
        va_data_setup = DS(X=va_images, y=va_labels, transform=TRANSFORM_2, mode="valid")

    dataloaders = {
        "train" : DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(SEED)),
        "valid" : DL(va_data_setup, batch_size=batch_size, shuffle=False)
    }

    return dataloaders


def save_graphs(L: list, A: list) -> None:
    TL, VL, TA, VA = [], [], [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graph")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Accuracy Graph")
    plt.savefig("./Graphs.jpg")
    plt.close("Plots")

#####################################################################################################

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH_1 = "./Data"
DATA_PATH_2 = "./Data_Reduced"
DATA_PATH_3 = "../input/edible-and-poisonous-fungi"
data_path_4 = None
TEST_DATA_PATH = "./Test Images"
CHECKPOINT_PATH = "./Checkpoints"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# To avoid clipping issue during tensor display, do not normalize
TRANSFORM_1 = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std=[0.229, 0.224, 0.225])])
TRANSFORM_2 = transforms.Compose([transforms.ToTensor(), ])
SIZE = 224
LABELS = [
    "Edible Mushroom Sporocarp", 
    "Edible Sporocarp", 
    "Poisonous Mushroom Sporocarp", 
    "Poisonous Sporocarp"
]

#####################################################################################################
