import torch
import logging
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import IDCM_NN
from train_model_knn_v2 import train_model_synchronous

from load_data import get_loader
from evaluate import  fx_calc_map_multilabel
import scipy.io as sio
from to_seed import to_seed
import numpy as np
######################################################################
# Start running
import argparse
import os
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')
def str2bool(str):
    return True if str.lower() == 'true' else False
#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, default="wiki") # wiki xmedia INRIA-Websearch nuswide

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.9) # CL rank_loss 0.9 0.5 0.5 for three datasets
parser.add_argument("--lamda", type=float, default=1) # label loss
parser.add_argument("--margin", type=float, default=0.9) # rank_loss
parser.add_argument('--lambd', default=5, type=float)
parser.add_argument('--barycenter_number', default=1, type=int)
parser.add_argument('--top_k', default=10, type=int)
parser.add_argument("--warm_up_epoch", type=int, default=0) # turning point
parser.add_argument("--MAX_EPOCH", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--output_dim", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4) # learning rate

parser.add_argument("--noisy_ratio", type=float, default=0.4) # 0.2 0.4 0.6 0.8
parser.add_argument("--noise_mode", type=str, default='asym') # sym asym
parser.add_argument("--GPU", type=int, default=1)
parser.add_argument("--hard_weight", type=str2bool, default=True) #
parser.add_argument("--hard_train", type=str2bool, default=True) #
parser.add_argument("--noisy_train", type=str2bool, default=True) #
parser.add_argument('--logging', type=str, default='test')  #


args = parser.parse_args()
print(args)
# 配置日志记录器

logger_name = args.logging if args.logging else args.dataset
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logging/' + logger_name + '.log'),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
if __name__ == '__main__':
    logger.info('Noise ratio: ' + str(args.noisy_ratio))
    logger.info(args)
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = args.dataset # IAPR MIRFlickr nuswide mscoco
    seed = args.seed
    to_seed(seed)
    # data parameters
    batch_size = args.batch_size
    output_dim = args.output_dim
    lr = args.lr
    weight_decay = 0
    noisy_ratio = args.noisy_ratio 
    noise_mode = args.noise_mode # sym asym
    print('...Data loading is beginning...')
    print('The noise_radio is: ', noisy_ratio)

    input_data_par = get_loader(dataset, batch_size, noisy_ratio, noise_mode)

    print('...Data loading is completed...')
    args.data_class = input_data_par['num_class']
    warm_up_epoch = args.warm_up_epoch
    
    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],output_dim=output_dim, num_class=input_data_par['num_class']).cuda()
    params_to_update = list(model_ft.parameters())
    # Observe that all parameters are being optimized
    optimizer = optim.Adam([{'params': model_ft.parameters(), 'lr': args.lr}]
                           )
    print('...Training is beginning...')
    # Train and evaluate
    model_ft, MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list, Clean_num_all_list = train_model_synchronous(model_ft, input_data_par, optimizer, args)
    print('...Training is completed...')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature = model_ft(torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    sio.savemat('Avg_MAP_data/' + dataset + '_Avg_MAP_%.1f_%d_'%(noisy_ratio, warm_up_epoch) +'.mat',{'MAPI2T_list':MAPI2T_list,'MAPT2I_list':MAPT2I_list})
    sio.savemat('Clean_num/' + dataset + '_Clean_Num_%.1f_%d_'%(noisy_ratio, warm_up_epoch) +'.mat',{'Clean_num_selected_list':Clean_num_selected_list,
                                                                                                    'num_selected_list':num_selected_list,
                                                                                                    'Clean_num_all_list':Clean_num_all_list})

    sio.savemat('features/' + dataset + '_' + str(0) + '_' + str(noisy_ratio) + '.mat', {'test_fea':view1_feature,
                                                                'test_lab':label})
    sio.savemat('features/' + dataset + '_' + str(1) + '_' + str(noisy_ratio) + '.mat', {'test_fea':view2_feature,
                                                                'test_lab':label})
    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='cosine')
    print('...Image to Text MAP = {}'.format(img_to_txt))
    logger.info('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='cosine')
    print('...Text to Image MAP = {}'.format(txt_to_img))
    logger.info('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
    logger.info('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))

    view1_feature, view2_feature = input_data_par['img_train'], input_data_par['text_train']
    label = np.argmax(input_data_par['label_train_ori'], axis=1)
    sio.savemat('features/' + dataset + '_' + str(0) + '_' + str(noisy_ratio) + '_train_ori.mat', {'train_fea':view1_feature,
                                                                'train_lab':label})
    sio.savemat('features/' + dataset + '_' + str(1) + '_' + str(noisy_ratio) + '_train_ori.mat', {'train_fea':view2_feature,
                                                                'train_lab':label})
    
    view1_feature, view2_feature = model_ft(torch.tensor(input_data_par['img_train']).cuda(), torch.tensor(input_data_par['text_train']).cuda())
    label = np.argmax(input_data_par['label_train_ori'], axis=1)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    sio.savemat('features/' + dataset + '_' + str(0) + '_' + str(noisy_ratio) + '_train.mat', {'train_fea':view1_feature,
                                                                'train_lab':label})
    sio.savemat('features/' + dataset + '_' + str(1) + '_' + str(noisy_ratio) + '_train.mat', {'train_fea':view2_feature,
                                                                'train_lab':label})
