import os
from PIL import Image
from torch.utils import data
from torchvision import transforms

patch_size_x = 160
patch_size_y = 160
to_tensor = transforms.Compose([transforms.Resize((patch_size_y, patch_size_x)),
                                transforms.ToTensor()])


class edge_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor, patch_size_y=160, patch_size_x=160):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'high':
                self.normal_path = temp_path
            if sub_dir == 'low':
                self.low_path = temp_path
        self.name_list = os.listdir(self.low_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]
        normal_edge = Image.open(os.path.join(self.normal_path, name)).convert('L')
        low = Image.open(os.path.join(self.low_path, name)).convert('L')
        normal_edge = self.transform(normal_edge)
        low = self.transform(low)
        #vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        # return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name
        return normal_edge,low, name

    def __len__(self):
        return len(self.name_list)
