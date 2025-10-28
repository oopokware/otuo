# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:32:56 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import torch


x_train= np.load('IMG_dataset.pkl',allow_pickle=True) #this line reads/loads the training data file 
print(x_train.shape)

def preprocess(img, min_accl, max_accl):
    transformed_images = np.zeros(img.shape)
    for i in range(img.shape[0]):
        transformed_images[i]=(img[i]-min_accl)/(max_accl - min_accl)
    return transformed_images

#image = x_train[0]
transformed_image = preprocess(x_train, -1.5, 1.5) #transforming all the training data instead of just one

n_rows, n_cols = 2, 6
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(17,6)) #creating an interface of plots usng subplots of 2 rows and 6 columns to display the earthquake data


for i in range(n_rows):
    for j in range(n_cols):
        idx= random.randint(0, transformed_image.shape[0])
        img = transformed_image[idx]
       # signal = img.flatten() #trying to unpack the img with is now in the form of a picture and now put in a form of a typical earthquake signal using the "flatten" function
        axs[i,j].imshow(img)
       # axs[i,j].plot(signal)

        
training_images = np.repeat(transformed_image,64,axis=0) #multiplying the number of data 382 by 64=24448
training_images = torch.as_tensor(training_images)   #concerting training data to torch tensor
training_images = training_images[:,:,:,None] #an alternative for this procedure in on the following line. used to add a 4th dimension at the index where none was specified
#training_images = training_images.unsqueeze(3) #we're adding another dimension to make the data 4-dimensional like a picture data which is always a 4-D tensor. the index we want to put the 4th dimension is typed into the parenthesis 

training_images = training_images.float() #by default the datatype we have is float64 which is a waste of computer memory. float() reduces it to float32 to save space
training_images = training_images.permute((0,3,1,2))

#print(training_images.max())  #printing the maximum and minimum values from the normalozation to see if our data is really within the specified min and max range. you can keep changing the max and min in the preprocess function
#print(training_images.min())


class DiffusionUtils():
    def __init__(self, n_timesteps,beta_min,beta_max,device="cpu" ): #we've created a class to hold the attributes for the beta we will use for the diffusion model forward process. the class will hold the number of steps from a minimum beta to a maximum beta.
        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.betas, self.alphas, self.alpha_bars, self.signal_rates, self.noise_rates  = self.diffusion_schedule() #calling the beatas here so that we can use them for calculations in this init function but we need to call the diffusion_schedule in which the betas was defined
    def diffusion_schedule(self):
        betas = torch.linspace(start= self.beta_min, end= self.beta_max, steps= self.n_timesteps).to(self.device)  #this function is created to create a linear spacing for the betas or noising to be used for the diffusion modeling process.
        alphas = 1-betas
        alpha_bars = torch.cumprod(alphas, 0)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1-alpha_bars)
        return betas, alphas, alpha_bars, signal_rates, noise_rates  #unlike the init function which is special and is able to store the betas otherwise they will be calculated and lost - nothing will be returned as output.
        #alphas, alpha_bars and the others are all formulas used in the diffusion modeling forward process.
    def noised_image(self,x0,t):
        alpha_bar_t = self.alpha_bars[t][:,None,None,None] #trying convert alpha_bars to have the same dimension of 4-dimension so that it can multiply a tensor dimensional object training_data with is in 4-dimension
        noise = torch.randn_like(x0) #random normal distribution of with a size like x0
        x_t = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1-alpha_bar_t)*noise
        return x_t
    
sample_images = training_images[0:4,:,:,:]
#sample_times = torch.randint(0,999,(4,1))
sample_times = torch.as_tensor(np.array([20,100,509,788]))       
akua = DiffusionUtils(1000, 0.0001, 0.02) #created an object to hold the attributes of a specific diffusion beta and retrieved the attributes to test the code.
abena = DiffusionUtils(800, 0.08, 0.8)

noised_images = akua.noised_image(sample_images, sample_times)

fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(33,3.5))

x0=training_images[0,:,:,:].squeeze(0).squeeze(0).cpu().numpy()

#axs[0].imshow(x0)
axs[0].plot(x0.flatten())
#plt.show()

for i in range(9):
    t = 100*(i+1) + 99
    t = torch.Tensor([t]).type(torch.int64)
    image_x0 = training_images[0,:,:,:]
    image_x_t =akua.noised_image(image_x0, t).squeeze(0).squeeze(0).cpu().numpy()
   # axs[i+1].imshow(image_x_t,cmap='gray')
    axs[i+1].plot(image_x_t.flatten())
   
plt.show()

