import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloaders import get_numpy_dataloader2D
from models_sagan import Generator, Discriminator
from training import Trainer
import argparse

def main(latent_dim):
    t1 = time.time()
    # Parameters
    data_dir = "./data/img_size32/linear_rect_train.npy"
    img_size = (32, 32, 1)
    batch_size = 64
    n_critic = 5
    dim = 16
    lr = 2e-4
    betas = (0.0, .99)
    epochs = 500
    np_save_freq = 10
    ckpt_save_freq = 10

    save_dir = f"./exps/sagan_Ncritic{n_critic}_Zdim{latent_dim}_BS{batch_size}_Nepoch{epochs}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + '/samples'):
        os.makedirs(save_dir + '/samples')
    if not os.path.exists(save_dir + '/checkpoints'):    
        os.makedirs(save_dir + '/checkpoints')
    if not os.path.exists(save_dir + '/optimizers'):
        os.makedirs(save_dir + '/optimizers')

    print('\n --- Saving parameters to file \n')
    param_file = save_dir + '/hyperparameters.txt'
    with open(param_file,"w") as fid:
        fid.write(f"Data directory = {data_dir}\n")
        fid.write(f"Image size = {img_size}\n")
        fid.write(f"Batch size = {batch_size}\n")
        fid.write(f"Number of critic iterations = {n_critic}\n")
        fid.write(f"Learning rate = {lr}\n")
        fid.write(f"Betas = {betas}\n")
        fid.write(f"Number of epochs = {epochs}\n")
        fid.write(f"Frequency of saving numpy files = {np_save_freq}\n")
        fid.write(f"Frequency of saving checkpoints = {ckpt_save_freq}\n")


    data_loader = get_numpy_dataloader2D(numpy_file_path=data_dir, batch_size=batch_size)
    generator = Generator(image_size=img_size[0], z_dim=latent_dim, batch_size=batch_size)
    discriminator = Discriminator(image_size=img_size[0], batch_size=batch_size)

    print(generator)
    print(discriminator)

    # Initialize optimizers
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


    # Train models
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,critic_iterations=n_critic,use_cuda=use_cuda)
    trainer.train(data_loader, epochs, np_save_freq=np_save_freq, ckpt_save_freq=ckpt_save_freq, save_dir=save_dir,save_training_np=True, save_training_gif=True)


    # Save models
    name = f"RectData"
    torch.save(trainer.G.state_dict(), f"./{save_dir}/gen_{name}.pt")
    torch.save(trainer.D.state_dict(), f"./{save_dir}/dis_{name}.pt")
    torch.save(G_optimizer.state_dict(), f"./{save_dir}/Goptimizer_{name}.pt")
    torch.save(D_optimizer.state_dict(), f"./{save_dir}/Doptimizer_{name}.pt")
    torch.save({"G_state_dict": trainer.G.state_dict(), "D_state_dict": trainer.D.state_dict(),
                "G_optimizer": G_optimizer.state_dict(), "D_optimizer": D_optimizer.state_dict()}, 
            f"./{save_dir}/model_{name}.pth")
    print(f"Total ellapsed time for {epochs} epoch is {time.time()-t1}")

    # Plot samples
    n_rows = 5
    n_cols = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        stat_z = torch.randn(n_rows*n_cols, latent_dim).to(device)
        pred_ = 0.5*(generator(stat_z).cpu().detach().numpy().squeeze()+1.0) 

    idx = 0
    fig, axs = plt.subplots(n_rows, n_cols)
    for ii in range(n_rows):
        for jj in range(n_cols):
            im = axs[ii,jj].imshow(pred_[idx], aspect='equal', vmin=0, vmax=1)
            axs[ii,jj].axis("off")
            idx += 1
        
    fig.tight_layout()      
    fig.savefig(f"{save_dir}/samples_at_ep_{epochs}.png")
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', type=int, default=5)
    args = parser.parse_args()
    main(args.z)
    print('\n ============== DONE =================\n')

