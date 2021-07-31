import os 
from PIL import Image 
import numpy as np
from k_medoids import K_Mediods

class mosaic():
    
    def __init__(self,src_image_dir,target_images_dir,alpha=0.5,grid_size=None,num_clusters=3):
        '''
        Mosaic Class to create mosaic images from source
        and target images

        Args:
            src_image_dir : str
                Path to source directory
            target_images_dir : str
                Path to target directory
            alpha : float
                Coefficient for linear combination
            grid_size : [int int]
                Size of grid 
            num_clusters : int
                Number of clusters for k mediods
        '''

        # list of image paths in the directory
        source_path = os.listdir(src_image_dir)
        target_path = os.listdir(target_images_dir)

        # Read the source image
        self.src_imgs = self.read_image(source_path,src_image_dir)

        # Read the target images
        self.target_imgs = self.read_image(target_path,target_images_dir,newsize=grid_size)

        # Coefficient for linear combination
        self.alpha = alpha

        self.num_clusters = num_clusters 


    def read_image(self,image_paths,parent_path,newsize=None):
        '''
        Read images

        Args:
            image_paths : str
                Path to images
            parent_path : str
                Path to parent directory
            newsize : [int int]
                Reshape the image
        
        Returns:
            numpy array [N x H x W x C]
                Numpy array of all the images
        '''

        images_list = []
        for i in range(len(image_paths)):

            # Read the image
            image = Image.open(parent_path + '/' + image_paths[i])

            # Resize the image
            if newsize != None:
                image = image.resize(newsize)

            # Normalize the image
            image = np.array(image) / 255.

            images_list.append(image)

        return np.array(images_list) 

    
    def apply_padding(self,src_img,target_shape):
        '''
        Apply padding to the top and right of the images

        Args:
            src_img : numpy array [H x W x C]
                Source image
            target_shape : (height, width)
                Path to parent directory
        
        Returns:
            src_img : numpy array [H x W x C]
                Numpy array of all the images
            padding_top : int
                size of padding on top
            padding_right : int
                size of padding on right
        '''

        h_src, w_src, _ = src_img.shape
        h_target, w_target = target_shape

        # Calculate padding
        padding_top = h_target - (h_src % h_target)
        padding_right = w_target - (w_src % w_target)

        # Pad the image on top and right
        src_img = np.pad(src_img,((padding_top,0),(0,padding_right),(0,0)))

        return src_img ,padding_top, padding_right

    
    def apply_mosaic(self):
        '''
        Create the mosaic

        Returns:
            mosaic_imgs : numpy array [B x H x W x C]
        '''

        mosaic_imgs = []
        for src_img in self.src_imgs:

            # Shape of target images
            target_height, target_width, _ = self.target_imgs[0].shape

            # Apply padding to the source image
            src_img, padding_top, padding_right = self.apply_padding(src_img, (target_height, target_width))

            # Shape of padded source image
            src_height, src_width, src_channels = src_img.shape
            mosaic_img = np.zeros((src_height, src_width, src_channels))

            # Linearize the image
            X = np.reshape(self.target_imgs,(self.target_imgs.shape[0],-1))

            # Create an instance of k_mediods class
            k_mediods = K_Mediods(X, self.num_clusters)

            # Loop over the image and apply linear combination
            n = src_height // target_height
            m = src_width // target_width

            for i in range(n):
                for j in range(m):

                    # Patch of the source image 
                    src_kernel = src_img[i*target_height:(i+1)*target_height,j*target_width:(j+1)*target_width,:]
                    img_kernel = np.reshape(src_kernel,(1,-1))

                    # index of the image using k_mediods
                    index = k_mediods.pred(img_kernel)

                    # Target image
                    target_kernel = self.target_imgs[index]

                    # Apply linear combination
                    mosaic_img[i*target_height:(i+1)*target_height,j*target_width:(j+1)*target_width,:] = src_kernel * self.alpha + target_kernel  * (1-self.alpha)

            # Remove the padding
            mosaic_img = mosaic_img[padding_top:,:-padding_right]

            # Append the final image
            mosaic_imgs.append(mosaic_img)
        
        return  mosaic_imgs