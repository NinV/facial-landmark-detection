import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from libs.models.networks.hourglass import Hourglass, StackedHourglass

# dims = [256, 256, 384]
# model = Hourglass(dims)

dims = [[256, 256, 384], [384, 384, 512]]
model = StackedHourglass(3, dims, 15)

summary(model, input_size=(3, 512, 512), device="cuda") # device="cpu"
writer = SummaryWriter()    # default folder "./runs"
image = torch.rand(1, 3, 512, 512)
writer.add_graph(model, image)
writer.close()









