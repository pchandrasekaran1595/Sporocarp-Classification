import sys
import torch

import utils as u
from api import fit, predict
from Models import ConvNet

#####################################################################################################

def app():
    args_0 = "--kaggle"
    args_1 = "--bs"
    args_2 = "--lr"
    args_3 = "--wd"
    args_4 = "--scheduler"
    args_5 = "--epochs"
    args_6 = "--early"
    args_7 = "--model-name"
    args_8 = "--path"
    args_9 = "--augment"
    args_10 = "--reduce"
    args_11 = "--pretrained"
    args_12 = "--test"
    args_13 = "--name"

    in_kaggle = False
    batch_size, lr, wd = 64, 1e-3, 0
    do_scheduler, scheduler = None, None
    epochs, early_stopping = 10, 5
    model_name = None
    do_augment = None
    do_reduce = None
    pretrained = False
    train_mode = True
    name = "Image_1.png"

    if args_0 in sys.argv: in_kaggle = True
    if args_1 in sys.argv: batch_size = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv: lr = float(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv: wd = float(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        do_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_4) + 1])
        eps = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv: epochs = int(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv: early_stopping = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv: model_name = sys.argv[sys.argv.index(args_7) + 1]
    if args_8 in sys.argv: u.data_path_4 = sys.argv[sys.argv.index(args_8) + 1]
    if args_9 in sys.argv: do_augment = True
    if args_10 in sys.argv: do_reduce = True
    if args_11 in sys.argv: pretrained = True
    if args_12 in sys.argv: train_mode = False
    if args_13 in sys.argv: name = sys.argv[sys.argv.index(args_13) + 1]

    assert(isinstance(model_name, str))
    torch.manual_seed(u.SEED)
    model = ConvNet(model_name=model_name, pretrained=pretrained)

    if train_mode:
        if isinstance(u.data_path_4, str):
            dataloaders = u.build_dataloaders(u.data_path_4, 
                                              batch_size=batch_size, 
                                              pretrained=pretrained,
                                              do_augment=do_augment)

        elif do_reduce is None:
            dataloaders = u.build_dataloaders(u.DATA_PATH_1, 
                                              batch_size=batch_size, 
                                              pretrained=pretrained,
                                              do_augment=do_augment)
        
        elif do_reduce:
            dataloaders = u.build_dataloaders(u.DATA_PATH_2, 
                                              batch_size=batch_size, 
                                              pretrained=pretrained,
                                              do_augment=do_augment)

        elif in_kaggle:
            dataloaders = u.build_dataloaders(u.DATA_PATH_3, 
                                              batch_size=batch_size, 
                                              pretrained=pretrained,
                                              do_augment=do_augment)

        
    
        optimizer = model.get_optimizer(lr=lr, wd=wd)
        if do_scheduler:
            scheduler = model.get_plateau_scheduler(patience=patience, eps=eps)
        
        L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                        early_stopping_patience=early_stopping, dataloaders=dataloaders, verbose=True)
        
        u.save_graphs(L, A)
        
        u.myprint("Execution Completed. Terminating ...", "yellow")
        u.breaker()
    
    else:
        image = u.read_image(name)
        if pretrained:
            label = predict(model=model, image=image, size=u.SIZE, transform=u.TRANSFORM_1)
        else:
            label = predict(model=model, image=image, size=u.SIZE, transform=u.TRANSFORM_2)
        
        u.breaker()
        u.myprint(label, "green")
        u.breaker()

        u.myprint("Execution Completed. Terminating ...", "yellow")
        u.breaker()

#####################################################################################################
