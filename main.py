import os
import torch
import random
import argparse
from EncDec import *
import importlib
mod = importlib.import_module("EncDec.2021485A3")
# from DLA3.EncDec.2021485A3 import *
from torch.utils.data import DataLoader

P = argparse.ArgumentParser()
P.add_argument("gpu", type=str)
P.add_argument("bonus", type=str)
A = P.parse_args()
    

if __name__ == "__main__":

    Data = DataLoader(dataset=mod.AlteredMNIST(),
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=2,
                      drop_last=True,
                      pin_memory=True)
    E = mod.Encoder()
    D = mod.Decoder()
    
    L = [mod.AELossFn(),
         mod.VAELossFn()]
    
    O = torch.optim.Adam(mod.ParameterSelector(E, D), lr=LEARNING_RATE)
    
    print("Training Encoder: {}, Decoder: {} on Modified MNIST dataset in AE training paradigm".format(
        E.__class__.__name__,
        D.__class__.__name__,
    ))
    mod.AETrainer(Data,
              E,
              D,
              L[0],
              O,
              A.gpu)
    
    print("Training Encoder: {}, Decoder: {} on Modified MNIST dataset in VAE training paradigm".format(
        E.__class__.__name__,
        D.__class__.__name__,
    ))
    mod.VAETrainer(Data,
               E,
               D,
               L[1],
               O,
               A.gpu)
    
    print("AE, VAE Training Complete")
        
    AE_Pipeline = mod.AE_TRAINED(gpu=False)
    VAE_Pipeline = mod.VAE_TRAINED(gpu=False)
    
    
    
    # TestData = TestMNIST()
    # AESSIM, VAESSIM = [], []
    # AEPSNR, VAEPSNR = [], []
    # for sample, original in TestData:
    #     AESSIM.append(AE_Pipeline.from_path(sample, original, type="SSIM"))
    #     VAESSIM.append(VAE_Pipeline.from_path(sample, original, type="SSIM"))
    #     AEPSNR.append(AE_Pipeline.from_path(sample, original, type="PSNR"))
    #     VAEPSNR.append(VAE_Pipeline.from_path(sample, original, type="PSNR"))
    
    # print("SSIM Score of AutoEncoder Training paradigm: {}".format(sum(AESSIM)/len(AESSIM)))
    # print("SSIM Score of Variational AutoEncoder Training paradigm: {}".format(sum(VAESSIM)/len(VAESSIM)))
    # print("PSNR Score of AutoEncoder Training paradigm: {}".format(sum(AEPSNR)/len(AEPSNR)))
    # print("PSNR Score of Variational AutoEncoder Training paradigm: {}".format(sum(VAEPSNR)/len(VAEPSNR)))
