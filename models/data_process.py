import albumentations as A
from torchvision.transforms import transforms
def get_transforms():
    image_trfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((352, 352)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
    mask_trfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((352, 352))
    ])
    trfm_dict = {'image': image_trfm, 'mask': mask_trfm}
    return trfm_dict

def get_data_augmentation():
    trfm = A.Compose([
        # A.RandomResizedCrop(352, 352, scale=(0.75, 1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])
    return trfm

if __name__ == '__main__':
    trfm_dict = {}
    trfm_dict['1'], trfm_dict['2'] = 1, 3
    print(trfm_dict['1'])

