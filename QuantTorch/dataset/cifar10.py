import torch
import torch.utils.data
import torchvision
from torchvision import transforms

def get_cifar10(batch_size=200, valid_ratio=0.2, directory="/tmp/CIFAR10", transform_both=None, transform_train=None, transform_valid=None, num_workers=2, seed=None):    
    if transform_train is None and transform_both is None:
        transform_train = transforms.Compose([
                        transforms.RandomCrop(size=(32, 32), padding=4),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                    ])
    elif not transform_both is None:
        transform_train = transform_both
    
    if transform_valid is None  and transform_both is None:
        transform_valid = transforms.Compose([
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    elif not transform_both is None:
        transform_valid = transform_both


    main_dataset = torchvision.datasets.CIFAR10(directory, train=True, download=True, transform=transform_train)
    train_dataset, valid_dataset = torch.utils.data.random_split(main_dataset,[int((1-valid_ratio)*main_dataset.data.shape[0]),int(valid_ratio*main_dataset.data.shape[0])] )
    train_set = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)    
    valid_set = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=True,num_workers=num_workers)    

    test_set = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(directory, train=False, download=True, transform=transform_valid),
                                batch_size=batch_size, shuffle=True)

    return train_set, valid_set, test_set

if __name__ == "__main__":
    _,_,_ = get_cifar10()