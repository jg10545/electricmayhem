import numpy as np
import torch


DEFAULT_PRINTABILITY = [[0.10588, 0.054902, 0.1098],
 [0.48235, 0.094118, 0.16863],
 [0.50196, 0.52549, 0.17647],
 [0.082353, 0.31765, 0.18431],
 [0.47843, 0.61176, 0.51765],
 [0.07451, 0.3098, 0.45882],
 [0.67843, 0.14902, 0.18039],
 [0.086275, 0.14118, 0.26275],
 [0.26667, 0.36863, 0.47843],
 [0.76078, 0.54118, 0.5451],
 [0.73333, 0.49412, 0.27451],
 [0.25882, 0.35294, 0.18039],
 [0.47843, 0.22353, 0.36471],
 [0.27059, 0.086275, 0.11765],
 [0.7098, 0.32157, 0.2],
 [0.27451, 0.13725, 0.29412],
 [0.75294, 0.75686, 0.63137],
 [0.28627, 0.54902, 0.41176],
 [0.47451, 0.2902, 0.15294],
 [0.74902, 0.70196, 0.28627],
 [0.098039, 0.42745, 0.44314],
 [0.50588, 0.65098, 0.65882],
 [0.12549, 0.42745, 0.23529],
 [0.4902, 0.58431, 0.33725],
 [0.26275, 0.49412, 0.26275],
 [0.07451, 0.14902, 0.12549],
 [0.090196, 0.20392, 0.36078],
 [0.68627, 0.15686, 0.30196],
 [0.30196, 0.5451, 0.57647],
 [0.71765, 0.32941, 0.40784]]



class NPSCalculator(torch.nn.Module):
    """
    calculates the non-printability score of a patch
    """

    def __init__(self, patch_size, printability_file=None):
        """
        :patch_size: tuple giving patch dimensions (H,W)
        :printability_file: string; path to comma-delimited text file containing
            RGB tuples of printable colors. if None, use default printability from paper.
        """
        super(NPSCalculator, self).__init__()
        self.printability_array = torch.nn.Parameter(self.get_printability_array(printability_file, patch_size),requires_grad=False)

    def forward(self, adv_patch):
        """
        calculate euclidian distance between colors in patch and colors in printability_array. square root of sum of squared difference.
        
        adv_patch should have shape (batch, C, H, W)
        """
        output = []
        for i in range(adv_patch.shape[0]):
            color_dist = (adv_patch[i] - self.printability_array+0.000001) # (num_prints, C, H, W)
            color_dist = color_dist ** 2  # squared difference # (num_prints, C, H, W)
            color_dist = torch.sum(color_dist, 1)+0.000001  # (num_prints, H, W)
            color_dist = torch.sqrt(color_dist) # (num_prints, H, W)

            # only work with the min distance
            color_dist_prod = torch.min(color_dist, 0)[0] # (num_prints, H, W)

            # calculate the nps by summing over all pixels
            nps_score = torch.mean(color_dist_prod)
            output.append(nps_score)
        return torch.stack(output)

    def get_printability_array(self, printability_file, size):
        if printability_file is None:
            printability_list = DEFAULT_PRINTABILITY
        else:
            printability_list = []
            # read in printability triplets and put them in a list
            with open(printability_file) as f:
                for line in f:
                    printability_list.append(line.split(","))

        # see notes for a better graphical representation
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((size[0], size[1]), red))
            printability_imgs.append(np.full((size[0], size[1]), green))
            printability_imgs.append(np.full((size[0], size[1]), blue))

            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array) 
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array) 
        return pa


def compute_total_variation_loss(img):      
    """
    
    """
    tv_h = torch.sum((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2))
    tv_w = torch.sum((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2))
    return tv_h + tv_w