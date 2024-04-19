import tonic
import tonic.transforms as transforms
import torch



sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)
denoise_transform = tonic.transforms.Denoise(filter_time=10000)

transform = transforms.Compose([denoise_transform, frame_transform])
dataset = tonic.datasets.NMNIST(save_to="../data_new", train=False, transform=transform)


torch.manual_seed(1234)

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
frames, target = next(iter(dataloader))

print("finished")