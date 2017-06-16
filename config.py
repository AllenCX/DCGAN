class Config(object):
    def __init__(self, batch_size, latent_size, lr, beta1, epoch_num, alpha):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.epoch_num = epoch_num
        self.alpha = alpha
        self.lr = lr
        self.beta1 = beta1
        self.summary_dir = "summary"
        #self.is_training = True