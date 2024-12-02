import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics
from configs.model_configs import build_RMAMamba_T, build_RMAMamba_S


def load_CirrMRI_data(path):
    def get_data(path, name):
        images = sorted(glob(os.path.join(path, name, "images", "*.png")))
        labels = sorted(glob(os.path.join(path, name, "masks", "*.png")))
        return images, labels

    """ Names """
    train_path = f"{path}/train"
    valid_path = f"{path}/valid"
    test_path = f"{path}/test"

    train_dirs = sorted(os.listdir(train_path))
    valid_dirs = sorted(os.listdir(valid_path))
    test_dirs = sorted(os.listdir(test_path))

    train_names = [item for item in train_dirs]
    valid_names = [item for item in valid_dirs]
    test_names = [item for item in test_dirs]

    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        x, y = get_data(train_path, name)
        train_x += x
        train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        x, y = get_data(valid_path, name)
        valid_x += x
        valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        x, y = get_data(test_path, name)
        test_x += x
        test_y += y

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def print_score(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)
    hd = metrics_score[6]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD: {hd:1.4f}")

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")
        name = f"{name[-3]}_{name[-1]}"

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image/255.0
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()

            sample = {'images': image, 'masks': mask}
            out = model(sample)
            y_pred = out['prediction']

            y_pred = torch.sigmoid(y_pred)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask """
            y_pred = process_mask(y_pred)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}", cat_images)
        cv2.imwrite(f"{save_path}/pred/{name}", y_pred)

    print_score(metrics_score)
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='RMAMamba_S')
    parser.add_argument('--data_path', type=str,
                        default='data/CirrMRI600+/T2-cirrhosis-2D')
    opt = parser.parse_args()
    file_path = f"files/{opt.model}"
    checkpoint_path = f"{file_path}/checkpoint.pth"
    save_path = f"{file_path}/result_map/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = eval(f'build_{opt.model}')().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('load best model success')
    model.eval()
    print(f"model: {opt.model}")

    """ Test dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_CirrMRI_data(opt.data_path)
    data_str = f"Dataset Size:\nTest: {len(test_x)}\n"
    print(data_str)

    for item in ["pred", "joint"]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)
    evaluate(model, save_path, test_x, test_y, size)
