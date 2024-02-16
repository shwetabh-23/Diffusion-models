import torchvision

def get_data(dataset_path):
    
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(80),
            torchvision.transforms.RandomResizedCrop(64, scale = (0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataset = torchvision.datasets.ImageFolder(dataset_path, transform= transform)

    return dataset

if __name__ == '__main__':
    dataset_path = r'data/'
    dataset = get_data(dataset_path= dataset_path)
    breakpoint()