import numpy as np

class imgPatch():
    def __init__(self, img, patch_size, edge_overlay):
        self.patch_size = patch_size
        self.edge_overlay = edge_overlay
        self.img = img[:,:,np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]

    def toPatch(self):
        patch_list = []
        start_list = []
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step),
                                          (self.edge_overlay, patch_step), (0,0)), 'constant')
        img_patch_row = (img_expand.shape[0]-self.edge_overlay)//patch_step
        img_patch_col = (img_expand.shape[1]-self.edge_overlay)//patch_step
        for i in range(img_patch_row):
            for j in range(img_patch_col):
                patch_list.append(img_expand[i*patch_step:i*patch_step+self.patch_size,
                                                j*patch_step:j*patch_step+self.patch_size, :])
                start_list.append([i*patch_step-self.edge_overlay, j*patch_step-self.edge_overlay])
        return patch_list, start_list, img_patch_row, img_patch_col

    def higher_patch_crop(self, higher_patch_size, start_list):
        higher_patch_list = []
        radius_bias = higher_patch_size//2-self.patch_size//2
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step), (self.edge_overlay, patch_step), (0,0)), 'constant')
        img_expand_higher = np.pad(img_expand, ((radius_bias, radius_bias), (radius_bias, radius_bias), (0,0)), 'constant')
        start_list_new = list(np.array(start_list)+self.edge_overlay+radius_bias)
        for start_i in start_list_new:
            higher_row_start, higher_col_start = start_i[0]-radius_bias, start_i[1]-radius_bias
            higher_patch = img_expand_higher[higher_row_start:higher_row_start+higher_patch_size,higher_col_start:higher_col_start+higher_patch_size,:]
            higher_patch_list.append(higher_patch)
        return higher_patch_list

    def toImage(self, patch_list, img_patch_row, img_patch_col):
        patch_list = [patch[self.edge_overlay//2:-self.edge_overlay//2, self.edge_overlay//2:-self.edge_overlay//2,:]
                                                        for patch in patch_list]
        patch_list = [np.hstack((patch_list[i*img_patch_col:i*img_patch_col+img_patch_col]))
                                                        for i in range(img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[self.edge_overlay//2:self.img_row+self.edge_overlay//2, \
            self.edge_overlay//2:self.img_col+self.edge_overlay//2,:]
        
        return img_array