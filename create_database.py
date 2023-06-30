# import sys
# import os

# if __name__ == '__main__':
#     root_path = sys.argv[1]
#     cv_annotation_path = os.path.join(root_path, 'PSI2.0_TrainVal/annotations/cv_annotation')
#     splits_path = os.path.join(root_path, "splits/PSI2_split.json")
#     database_path = os.path.join(root_path, "database")
#     args = 
    
#     if not os.path.exists(database_path):
#         os.makedirs(database_path)
#         print(f"Created '{database_path}' folder.")
from opts import get_opts
import os
from database.create_database import create_database


def main(args):
    ''' 1. Load database '''
    if not os.path.exists(args.database_path):
        os.makedirs(args.database_path)
        print(f"Created '{args.database_path}' folder.")
        
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        
        create_database(args)
    else:
        print("Database exists!")
    

if __name__ == '__main__':
    args = get_opts()
    # args.dataset_root_path = '/home/scott/Work/Toyota/PSI_Competition/Dataset'
    # Dataset
    args.dataset = 'PSI2.0'
    if args.dataset == 'PSI2.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI2.0_TrainVal/splits/PSI2_split.json')
    elif args.dataset == 'PSI1.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI1.0/splits/PSI1_split.json')
    else:
        raise Exception("Unknown dataset name!")

    # Task
    args.task_name = 'ped_traj'
    # intent prediction
    if args.task_name == 'ped_intent':
        args.database_file = 'intent_database_train.pkl'
        args.intent_model = True
        # intent prediction
        args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
        args.intent_type = 'mean' # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
        args.intent_loss = ['bce']
        args.intent_disagreement = 1  # -1: not use disagreement 1: use disagreement to reweigh samples
        args.intent_positive_weight = 0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)
    # trajectory prediction
    elif args.task_name == 'ped_traj':
        args.database_file = 'traj_database_train.pkl'
        args.intent_model = False # if use intent prediction module to support trajectory prediction
        args.traj_model = True
        args.traj_loss = ['bbox_l1']
        # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
        # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

    args.seq_overlap_rate = 0.9 # overlap rate for trian/val set
    args.test_seq_overlap_rate = 1 # overlap for test set. if == 1, means overlap is one frame, following PIE
    args.observe_length = 15
    if args.task_name == 'ped_intent':
        args.predict_length = 1 # only make one intent prediction
    elif args.task_name == 'ped_traj':
        args.predict_length = 45

    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = 'enlarge'
    args.normalize_bbox = 'subtract_first_frame' #None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]


    # # Model
    # args.model_name = 'lstmed_traj_bbox'  # LSTM module, with bboxes sequence as input, to predict intent
    # args.load_image = False # only bbox sequence as input

    # if args.load_image:
    #     args.backbone = 'resnet'
    #     args.freeze_backbone = False
    # else:
    #     args.backbone = None
    #     args.freeze_backbone = False

    # # Train
    # args.epochs = 100
    # args.batch_size = 128
    # if args.task_name == 'ped_traj':
    #     args.lr = 1e-2
    # elif args.task_name == 'ped_intent':
    #     args.lr = 1e-3

    # args.loss_weights = {
    #     'loss_intent': 0.0,
    #     'loss_traj': 1.0,
    #     'loss_driving': 0.0
    # }
    # args.val_freq = 1
    # args.test_freq = 1
    # args.print_freq = 10

    # # Record
    # now = datetime.now()
    # time_folder = now.strftime('%Y%m%d%H%M%S')
    # args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, args.dataset, args.model_name, time_folder)
    # if not os.path.exists(args.checkpoint_path):
    #     os.makedirs(args.checkpoint_path)
    # with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=4)

    # result_path = os.path.join(args.checkpoint_path, 'results')
    # if not os.path.isdir(result_path):
    #     os.makedirs(result_path)

    main(args)