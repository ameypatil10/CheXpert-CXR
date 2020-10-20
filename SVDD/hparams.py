import os
import torch

class Hparams():
    def __init__(self):

        self.cuda = True if torch.cuda.is_available() else False

        """
        Data Parameters
        """

        # os.makedirs('../input', exist_ok=True)
        os.makedirs('../model', exist_ok=True)
        os.makedirs('../data/', exist_ok=True)
        os.makedirs('../results/', exist_ok=True)

        self.mnist_target_label = 0

        self.train_csv = '/home1/himanshu/CheXpert-v1.0/oc_train_cardiomegaly.csv'
        self.test_csv = '/home1/himanshu/CheXpert-v1.0/test_cardiomegaly.csv'
        self.valid_csv = '/home1/himanshu/CheXpert-v1.0/valid_cardiomegaly.csv'

        self.train_dir = '/home1/himanshu/'
        self.test_dir = '/home1/himanshu/'
        self.valid_dir = '/home1/himanshu/'

        """
        Model Parameters
        """


        os.makedirs('../model/', exist_ok=True)

        self.image_shape = (512, 512)
        self.num_channel = 1
        self.repeat_infer = 1


        """
        Training parameters
        """

        self.epsilon = 0.1

        self.thresh = 0
        self.nu = 0.1
        self.rec_thresh = 0.06
        self.ae_weight_decay = 0.0005
        self.weight_decay = 0.5e-6
        self.pretrain = False
        self.load_model = False
        self.optimizer = 'amsgrad'

        self.latent_dim = 128*4*4

        self.objective = 'soft-boundary'

        self.pretrain_epoch = 25
        self.num_epochs = 50 + self.pretrain_epoch
        self.lr_milestones = [20, 40]
        self.warmup_epochs = 2
        self.batch_size = 64

        self.train_lr = 0.0001
        self.pretrain_lr = 0.0001

        self.print_interval = 20

        self.gpu_device = 'cuda:1'

        ################################################################################################################################################
        self.exp_name = 'svdd-chest-{}/'.format(self.objective)
        ################################################################################################################################################

        self.result_dir = '../results/'+self.exp_name
        os.makedirs(self.result_dir, exist_ok=True)

        self.model_dir = '../model/' + self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self.model_dir + 'model'


hparams = Hparams()
