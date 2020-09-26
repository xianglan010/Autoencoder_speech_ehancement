import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

from scipy.io import wavfile
from tqdm import tqdm 
import os
from tensorboardX import SummaryWriter
from model import *
from dataset import AudioSampleGenerator,split_pair_to_vars
from pa.pase.models.frontend import wf_builder

def model_weights_norm(model):
    for k, v in model.named_parameters():
        if 'weight' in k:
            W = v.data
            W_norm = torch.norm(W)

if __name__ == '__main__':
	# set parameters
    batch_size = 32
    num_epochs = 200 
    lr = 0.00005
    sample_rate = 16000

    # parameter for autoencoder
    enc_layer = [64,128,256,512,1024]
    kwidth = [31,31,31,31,31]
    enc_pool = [4,4,4,4,4]
    dec_layer = [512,256,128,64,1]
    dec_pool = [4,4,4,4,4]
    dec_kwidth = [31,31,31,31,31]
    z_dim = 1024
    bias = True

    data_root_dir = '/home/lan3/data'
    checkpoint_path = 'checkpoints_AE_SE'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_devices = [0]

    # set model
    model = AE_SE(1,enc_layer,kwidth,enc_pool,dec_layer,dec_kwidth,dec_pool,bias)
    model = torch.nn.DataParallel(model.to(device), device_ids = [0])
    print(model)


    # load PASE
    pase = wf_builder('pa/cfg/frontend/PASE+.cfg').eval().cuda()
    pase.load_pretrained('FE_e199.ckpt',load_last=True,verbose=True)

    # load data
    print('loading data...')
    sample_generator = AudioSampleGenerator(os.path.join(data_root_dir,'ser_data_ae_se'))
    random_data_loader = DataLoader(dataset= sample_generator,batch_size=batch_size,shuffle=True,
    	drop_last = True,pin_memory=True)
    val_generator = AudioSampleGenerator(os.path.join(data_root_dir,'val_ser_se'))
    val_data_loader = DataLoader(dataset = val_generator, batch_size=batch_size,shuffle=False,drop_last=True)

    print('Dataloader created')

    # opt and loss
    optimizer = optim.RMSprop(model.parameters(),lr=lr)
    #MSE = nn.MSEloss()

    log_dir = os.path.join(os.getcwd(),'logs')
    tbwriter = SummaryWriter(log_dir=log_dir)
    print('tensorboard summary writer created')

    # start training 
    print('start training')
    total_steps = 1

    for epoch in range(num_epochs):
        #tbwriter.add_scalar('epoch',epoch,total_steps)

        train_total_loss = 0
        train_pase_loss = 0
        train_ae_loss = 0
        cnt_train = 0
        model.train()

        for i,batch_pairs in enumerate(random_data_loader):
            batch_pairs_var, clean_batch_var, noisy_batch_var = split_pair_to_vars(batch_pairs)

            batch_pairs_var = batch_pairs_var.to(device)
            clean_batch_var = clean_batch_var.to(device)
            noisy_batch_var = noisy_batch_var.to(device)

            enh_output = model(noisy_batch_var)

            pase_outputs = pase(enh_output)
            pase_clean = pase(clean_batch_var)

            loss1 = torch.mean(torch.abs(torch.add(enh_output, torch.neg(clean_batch_var))))
            loss2 = torch.mean(torch.abs(torch.add(pase_outputs, torch.neg(pase_clean))))
            loss = loss1 + loss2 * 0.1

            #loss = torch.mean(torch.abs(torch.add(outputs, torch.neg(clean_batch_var))))

            train_total_loss += loss.item()
            train_ae_loss += loss1.item()
            train_pase_loss += loss2.item()

            cnt_train += 1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            model_weights_norm(model)

            if (i+1) % 10 == 0:
                print('Epoch {}\t' 'Step{}\t' 'loss {:.5f}'.format(epoch+1,i+1,loss.item()))
        
        # validation
        model.val()
        cnt_val = 0
        val_total_loss = 0
        val_ae_loss = 0
        val_pase_loss = 0

        with torch.no_grad():
            for i,batch_pairs in enumerate(val_data_loader):
                batch_pairs_var, clean_batch_var, noisy_batch_var = split_pair_to_vars(batch_pairs)
                batch_pairs_var = batch_pairs_var.to(device)
                clean_batch_var = clean_batch_var.to(device)
                noisy_batch_var = noisy_batch_var.to(device)


                enh_output = model(noisy_batch_var)
                pase_outputs = pase(enh_output)
                pase_clean = pase(clean_batch_var)

                loss1 = torch.mean(torch.abs(torch.add(enh_output, torch.neg(clean_batch_var))))
                loss2 = torch.mean(torch.abs(torch.add(pase_outputs, torch.neg(pase_clean))))
                loss = loss1 + loss2 * 0.1

                val_total_loss += loss.item()
                val_ae_loss += loss1.item()
                val_pase_loss += loss2.item()
                cnt_val += 1

        train_epoch_loss = round(train_total_loss/cnt_train,3)
        train_epoch_pase = round(train_pase_loss/cnt_train,3)
        train_epoch_ae = round(train_ae_loss/cnt_train,3)

        val_epoch_loss = round(val_total_loss/cnt_val,3)
        val_epoch_pase = round(val_pase_loss/cnt_val,3)
        val_epoch_ae = round(val_ae_loss/cnt_val,3)

        tbwriter.add_scalar('loss/train_toal_loss',train_epoch_loss,epoch)
        tbwriter.add_scalar('loss/train_pase_loss',train_epoch_pase,epoch)
        tbwriter.add_scalar('loss/train_ae_loss',train_epoch_ae,epoch)

        tbwriter.add_scalar('loss/val_total_loss',val_epoch_loss,epoch)
        tbwriter.add_scalar('loss/val_pase_loss',val_epoch_pase,epoch)
        tbwriter.add_scalar('loss/val_ae_loss',val_epoch_ae,epoch)

        # save various states
        state_path =os.path.join(checkpoint_path,'AE_SE-{}.pkl'.format(epoch+1))
        state ={
            'AE_SE': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, state_path)

    tbwriter.close()
    print('Finished Training')
