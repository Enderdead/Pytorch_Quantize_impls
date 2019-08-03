import torch
import torch.utils.data
import torchvision
def get_mnist(batch_size=200, valid_ratio=0.2, directory="/tmp/MNIST", transform_both=None, transform_train=None, transform_valid=None, num_workers=2):
    if transform_train is None and transform_both is None:
        if transform_both is None:
            transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
        else:
            transform_train = transform_both
    
    if transform_valid is None  and transform_both is None:
        if transform_both is None:
            transform_valid = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
        else:
            transform_valid = transform_both


    main_dataset = torchvision.datasets.MNIST(directory, train=True, download=True, transform=transform_train)
    train_dataset, valid_dataset = torch.utils.data.random_split(main_dataset,[int((1-valid_ratio)*main_dataset.data.size()[0]),int(valid_ratio*main_dataset.data.size()[0])] )
    
    train_set = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)    
    valid_set = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=True,num_workers=num_workers)    

    test_set = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(directory, train=False, download=True, transform=transform_valid),
                                batch_size=batch_size, shuffle=True)

    return train_set, valid_set, test_set