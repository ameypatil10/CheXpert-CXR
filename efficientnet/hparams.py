import os
import torch

class Hparams():
    def __init__(self):

        self.cuda = True if torch.cuda.is_available() else False

        """
        Data Parameters
        """

        os.makedirs('../model', exist_ok=True)
        os.makedirs('../data/', exist_ok=True)
        os.makedirs('../results/', exist_ok=True)

        self.train_csv = '../data/train-u1.csv'
        self.valid_csv = '../data/valid.csv'
        
        self.train_dir = '/home1/amey/CheXpert-v1.0-downsampled/'
        self.valid_dir = '/home1/amey/CheXpert-v1.0-downsampled/'

        """
        Model Parameters
        """

        os.makedirs('../model/', exist_ok=True)

        self.image_shape = (320, 320)
        self.num_channel = 3
        self.num_classes = 14

        self.id_to_class = {
            0: 'No Finding',
            1: 'Cardiomegaly',
            2: 'Edema',
            3: 'Consolidation',
            4: 'Atelectasis',
            5: 'Pleural Effusion',
            6: 'Enlarged Cardiomediastinum',
            7: 'Lung Opacity',
            8: 'Lung Lesion',
            9: 'Pneumonia',
            10: 'Pneumothorax',
            11: 'Pleural Other',
            12: 'Fracture',
            13: 'Support Devices'
        }
#         self.id_to_class = {
#             0: 'No Finding',
#             1: 'Cardiomegaly',
#             2: 'Edema',
#             3: 'Consolidation',
#             4: 'Atelectasis',
#             5: 'Pleural Effusion',
#             6: 'Lung Opacity',
#             7: 'Pneumonia',
#             8: 'Pneumothorax',
#             9: 'Pleural Other',
#             10: 'Support Devices'
#         }
    
        self.eval_labels = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        self.eval_id_to_class = {
#             0: 'No Finding',
            0: 'Cardiomegaly',
            1: 'Edema',
            2: 'Consolidation',
            3: 'Atelectasis',
            4: 'Pleural Effusion',
        }

        """
        Training parameters
        """

        self.gpu_device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        self.device_ids = [6]

        self.pretrained = True

        self.thresh = 0.5
        self.repeat_infer = 1
        self.eval_dp_on = False

        self.num_epochs = 15
        self.batch_size = 32

        self.learning_rate = 1e-5

        self.momentum1 = 0.5
        self.momentum2 = 0.999
        
        self.drop_rate = 0.5

        self.avg_mode = 'micro'

        self.print_interval = 1000
        
        self.TTA = 0
        self.augment = 1
        
#         self.lsr = 1

        ################################################################################################################################################
        self.exp_name = 'efficientnet-B1-sched-u1/'
        ################################################################################################################################################

        self.result_dir = '../results/'+self.exp_name
        os.makedirs(self.result_dir, exist_ok=True)

        self.model_dir = '../model/' + self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self.model_dir + 'model'


hparams = Hparams()
