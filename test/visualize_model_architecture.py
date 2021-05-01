import sys
from pathlib import Path
project_path = str(Path(__file__).absolute().parents[1])
sys.path.insert(0, project_path)

import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from libs.models.networks.hourglass import Hourglass, StackedHourglass


dims = [[256, 256, 384], [384, 384, 512]]
image_size = 512, 512

device = "cuda"     # or device="cpu"
model = StackedHourglass(3, dims).to(device=device)

# reduce memory usage
with torch.no_grad():
    # summary(model, input_size=(3, 512, 512), device=device)
    writer = SummaryWriter()    # default folder "./runs"
    image = torch.rand(1, 3, *image_size).to(device=device)
    writer.add_graph(model, image)
    writer.close()









