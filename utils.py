# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def pltLossAcc(loss_plt,acc_plt,config):
    plt.plot(acc_plt,label='Train accuracy')
    plt.plot(loss_plt,label='Train loss')
    plt.title('Train loss (NLL) and accuracy curves, \n'+config.dataset+', '+config.model_type,fontsize=15)
    plt.xlabel('Training step (mini batch)',fontsize=15)
    plt.ylabel('Loss (NLL), Accuracy',fontsize=15)
    note1 = 'seq_len='+str(config.input_length)+', num_hidden='+str(config.num_hidden)+', in_dim='+str(config.input_dim)
    note2 = 'bsize=' + str(config.batch_size) + ', lr=%.1E' %config.learning_rate
    plt.text(0,0.15, note1)
    plt.text(0,0.1, note2)
    plt.legend()
    axes=plt.axes()
    axes.set_ylim(0,1.05)
    plt.show()

  