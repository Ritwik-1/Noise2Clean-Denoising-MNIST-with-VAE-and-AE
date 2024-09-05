from EncDec import *
import os
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from sklearn.manifold import TSNE

# MAPPING ALL CLEAN IMAGES TO AUG IMAGES
class AlteredMNIST:
    def __init__(self):
        self.aug_path = os.path.join("Data/aug")
        self.clean_path = os.path.join("Data/clean")

        self.aug_fnames = os.listdir(self.aug_path)
        self.clean_fnames = os.listdir(self.clean_path)

        self.aug_map = {}
        self.clean_map = {}

        for path in self.aug_fnames:
            number = int(path.split("_")[-1].split(".")[0])
            if number not in self.aug_map:
                self.aug_map[number] = []
                self.aug_map[number].append(path)
            else:
                self.aug_map[number].append(path)

        for path in self.clean_fnames:
            number = int(path.split("_")[-1].split(".")[0])
            if number not in self.clean_map:
                self.clean_map[number] = []
                self.clean_map[number].append(path)
            else:
                self.clean_map[number].append(path)

        self.aug_images_final = []
        self.clean_images_final = []

        # 1 clean image is mapped to 3 aug images , 23,199 pairs 
        for i in range(10):
            # aug_map[i] = [aug_index_i] and clean_map[i] = [clean_index_i]
            j = 0
            k = 0
            while(k+2 < len(self.aug_map[i]) and j < len(self.clean_map[i])):
                self.clean_images_final.append(self.clean_map[i][j])
                self.clean_images_final.append(self.clean_map[i][j])
                self.clean_images_final.append(self.clean_map[i][j])

                self.aug_images_final.append(self.aug_map[i][k])
                self.aug_images_final.append(self.aug_map[i][k+1])
                self.aug_images_final.append(self.aug_map[i][k+2])

                j+=1
                k+=3
            
    def __len__(self):
        return len(self.clean_images_final)
    
    def __getitem__(self,index):
        aug_image_fname = self.aug_images_final[index]
        # mnist_digit = int(aug_image_fname.split("_")[-1].split(".")[0])
        clean_image_fname = self.clean_images_final[index]

        aug_image = torchvision.io.read_image(os.path.join(self.aug_path,aug_image_fname))
        clean_image = torchvision.io.read_image(os.path.join(self.clean_path,clean_image_fname))

        if aug_image.shape[0] == 3:
            aug_image = torchvision.transforms.functional.rgb_to_grayscale(aug_image)

        if clean_image.shape[0] == 3:
            clean_image = torchvision.transforms.functional.rgb_to_grayscale(clean_image)
 
        aug_image = aug_image.float() 
        clean_image = clean_image.float() 
        
        return aug_image,clean_image
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,type,stride=1, dim=1):
        super().__init__()
        if type == "D":
            conv_class = torch.nn.ConvTranspose2d
        else:
            conv_class = torch.nn.Conv2d

        self.conv1 = conv_class(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv_class(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                conv_class(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.residual = ResBlock(64, 64,"E",stride=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # for AE 
        self.fc_ae = torch.nn.Linear(64*28*28,20)

        # For VAE
        self.fc_mu = torch.nn.Linear(64 * 28 * 28, 20)
        self.fc_logvar = torch.nn.Linear(64 * 28 * 28, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_ae(x)
        # x = self.pool(x)
        return x
    
    def forward2(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # input here is 64xHxW
        self.residual = ResBlock(64, 64,"D",stride=1)

        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # For VAE
        self.fc = torch.nn.Linear(20, 64 * 28 * 28)
        # self.unpool = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 28, 28)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)
        return x
    
    def forward2(self,x):
        x = self.fc(x)
        x = x.view(-1, 64, 28, 28)
        # x = self.unpool(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)
        return x

# L = [mod.AELossFn(),
#      mod.VAELossFn()]
    
class AELossFn:
    def forward(self,img1,img2):
        mse_loss = F.mse_loss(img1,img2)
        return mse_loss 


class VAELossFn:
    def forward(self,output,clean,mu,logvar):
        mse = F.mse_loss(output,clean)
        kl_d = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + kl_d
        return loss

# O = torch.optim.Adam(mod.ParameterSelector(E, D), lr=LEARNING_RATE)
def ParameterSelector(E, D):
    combined_parameters = list(E.parameters()) + list(D.parameters())

    return combined_parameters

# mod.AETrainer(Data,E,D,L[0],O,A.gpu)

def calculateSimilarity(tens1,tens2):
    # assuming tens1 and tens2 = Bx1x28x28 
    
    ssim_values = [ssim(tens1[i].squeeze(0).detach().cpu().numpy(),
                                    tens2[i].squeeze(0).detach().cpu().numpy()
                                    ,data_range = tens2[i].squeeze(0).detach().cpu().numpy().max() 
                                    - tens2[i].squeeze(0).detach().cpu().numpy().min()) 
                                    for i in range(tens1.shape[0])]
    average_ssim = sum(ssim_values)/len(ssim_values)
    # average ssim is a floating point scalar 
    return average_ssim


class AETrainer:
    def __init__(self,data,enc,dec,los,opti,gpu):
        # torch.autograd.set_detect_anomaly(True)
        self.gpu = gpu 
        self.device = torch.device("cuda" if self.gpu == "T" else "cpu")

        self.train_dataloader = data
        self.encoder = enc.to(self.device)
        self.decoder = dec.to(self.device)
        self.loss_fn = los
        self.optimizer = opti 
    
        embeddings = []  
        best_similarity = 0
        for epoch in range(EPOCH):
            epoch_losses = 0
            epoch_similarities = 0

            minibatch = 0

            for aug,clean in self.train_dataloader:
                # print("aug shape : ",aug.shape)
                # print("clean shape : ",clean.shape)

                # print("aug shape : ",aug.dtype)
                # print("clean shape : ",clean.dtype)

                aug = aug.to(self.device)
                clean = clean.to(self.device)

                latent = self.encoder(aug)
                output = self.decoder(latent)

                loss = self.loss_fn.forward(output,clean)

                epoch_losses += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                similarity = calculateSimilarity(output,clean)
                epoch_similarities += similarity

                if minibatch == 9:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
                minibatch += 1
            epoch_loss = epoch_losses/len(self.train_dataloader)
            epoch_similarity = epoch_similarities/len(self.train_dataloader)

            # SAVING CHECKPOINT AFTER EACH EPOCH IF SSIM IS BETTER 
            if epoch_similarity > best_similarity:
                torch.save(self.encoder,'AE_enc.pth')
                torch.save(self.decoder,'AE_dec.pth')
                best_similarity = epoch_similarity

            print(f"----- Epoch:{epoch}, Loss:{epoch_loss}, Similarity:{epoch_similarity}")
            
            # Plotting tsne for every 10th epoch
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    embeddings = []
                    for aug, clean in self.train_dataloader:
                        aug = aug.to(self.device)
                        clean = clean.to(self.device)
                        outputs = self.encoder(aug)
                        embeddings.append(outputs.view(outputs.size(0), -1))

                    embeddings = torch.cat(embeddings, dim=0)

                    embeddings_cpu = embeddings.detach().cpu().numpy()
                    
                    tsne = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
                    embeddings_tsne = tsne.fit_transform(embeddings_cpu)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1)  
                    plt.title(f'2D t-SNE after Epoch {epoch}')
                    plt.savefig(f'AE_epoch_{epoch}.png')
                    plt.close()

class VAETrainer:
    def __init__(self,data,enc,dec,los,opti,gpu):
        # torch.autograd.set_detect_anomaly(True)
        self.gpu = gpu 
        self.device = torch.device("cuda" if self.gpu == "T" else "cpu")

        self.train_dataloader = data
        self.encoder = enc.to(self.device)
        self.decoder = dec.to(self.device)
        self.loss_fn = los
        self.optimizer = opti 
    
        embeddings = []
        best_similarity = 0  
        for epoch in range(EPOCH):
            epoch_losses = 0
            epoch_similarities = 0

            minibatch = 0

            for aug,clean in self.train_dataloader:
                aug = aug.to(self.device)
                clean = clean.to(self.device)

                mu, logvar = self.encoder.forward2(aug)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                outputs = self.decoder.forward2(z)

                loss = self.loss_fn.forward(outputs,clean,mu,logvar)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_losses += loss.item()

                similarity = calculateSimilarity(outputs,clean)
                epoch_similarities += similarity

                if minibatch == 9:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
                minibatch += 1
            epoch_loss = epoch_losses/len(self.train_dataloader)
            epoch_similarity = epoch_similarities/len(self.train_dataloader)

            # SAVING CHECKPOINT AFTER EACH EPOCH IF SSIM IS BETTER 
            if epoch_similarity > best_similarity:
                torch.save(self.encoder,'AE_enc.pth')
                torch.save(self.decoder,'AE_dec.pth')
                best_similarity = epoch_similarity

            print(f"----- Epoch:{epoch}, Loss:{epoch_loss}, Similarity:{epoch_similarity}")

            # Plotting tsne for every 10th epoch
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    embeddings = []
                    for aug, clean in self.train_dataloader:  
                        aug = aug.to(self.device)
                        clean = clean.to(self.device)

                        mu, logvar = self.encoder.forward2(aug)
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mu + eps * std
                        outputs = self.decoder.forward2(z)

                        embeddings.append(outputs.view(outputs.size(0), -1))

                    embeddings = torch.cat(embeddings, dim=0)

                    embeddings_cpu = embeddings.detach().cpu().numpy()
                    
                    tsne = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
                    embeddings_tsne = tsne.fit_transform(embeddings_cpu)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1)  
                    plt.title(f'2D t-SNE after Epoch {epoch}')
                    plt.savefig(f'VAE_epoch_{epoch}.png')
                    plt.close()


class AE_TRAINED:
    def __init__(self,gpu):
        self.gpu = gpu

        self.enc = torch.load('AE_enc.pth')
        self.enc.eval()

        self.dec = torch.load('AE_dec.pth')
        self.dec.eval()

    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        with torch.no_grad():
            enc_out = self.enc(sample)
            output = self.dec(enc_out)

            # assuming sample is a tensor of 1xHxW
            
            if type == "SSIM":
                return ssim(output[0].detach().cpu().numpy(),original[0].detach().cpu().numpy())
                # return structure_similarity_index(output,original)
            else:
                return peak_signal_noise_ratio(output[0].detach().cpu().numpy(),original[0].detach().cpu().numpy())
                # return peak_signal_to_noise_ratio(output,original)

class VAE_TRAINED:
    def __init__(self,gpu):
        self.gpu = gpu

        self.enc = torch.load('VAE_enc.pth')
        self.enc.eval()

        self.dec = torch.load('VAE_dec.pth')
        self.dec.eval()

    def from_path(self,sample, original, type):
        with torch.no_grad():
            mu, logvar = self.encoder.forward2(sample)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            output = self.decoder.forward2(z)

            if type == "SSIM":
                return ssim(output[0].detach().cpu().numpy(),original[0].detach().cpu().numpy())
                # return structure_similarity_index(output,original)
            else:
                return peak_signal_noise_ratio(output[0].detach().cpu().numpy(),original[0].detach().cpu().numpy())
                # return peak_signal_to_noise_ratio(output,original)

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()