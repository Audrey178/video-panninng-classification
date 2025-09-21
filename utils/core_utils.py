import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os
import pandas as pd
import torch.optim as optim 
import torch.nn as nn
import numpy as np



def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
	else:
		raise NotImplementedError
	return optimizer


def performance_metric(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }



def saveresult(epoch, savefolder, strategy, test_predict, Y_test):
    save_folder_path = os.path.join(savefolder, strategy)
    result_path = os.path.join(save_folder_path, "result")
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    test_predict = test_predict.reshape(-1)
    Y_test = Y_test.reshape(-1)
    len_result = len(test_predict)

    df_result = pd.DataFrame({
        'predict': test_predict,
        'groundtruth': Y_test
    })
    pathfile = os.path.join(result_path, f"{strategy}.csv")
    df_result.to_csv(pathfile)

    res = performance_metric(Y_test, test_predict)
    res['epoch'] = epoch
    res["strategy"] = strategy

    path_res = os.path.join(save_folder_path,  f"{strategy}_result_measure.csv")
    pd.DataFrame(res, index=[0]).to_csv(
        path_res, mode='a', header=not os.path.exists(path_res),
        index=False, columns=["epoch", "strategy", "accuracy", "f1_score", "precision", "recall"]
    )
    
    
def train(args, train_loader, val_loader, device, model = None):
    writer_dir = args.results_dir_path
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    loss_fn = nn.CrossEntropyLoss()
    
    # print('\nInit Model...', end=' ')
    # model_dict = {"dropout": args.dropout, 
    #               'n_classes': args.num_classes, 
    #               "embed_dim": args.emhidden_sizebed_dim}
    
    # model = VTrans(**model_dict)
    if model is not None:
        _ = model.to(device)
    else:
        raise ValueError

    optimizer = get_optim(model, args)

    # print('\nSetup EarlyStopping...', end=' ')
    # if args.early_stopping:
    #     early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    # else:
    #     early_stopping = None
    # print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, device, loss_fn)
        validate(epoch, model, val_loader, device, args.strategy, loss_fn, args.results_dir_path)
    # if args.early_stopping:
    #     model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))


def train_loop(epoch, model, loader, optimizer, device, loss_fn = None):   
    model.train()
    train_loss = 0.

    print('==============Training===============\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits = model(data)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss_value))
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss))


def validate(epoch, model, loader, device , strategy , loss_fn = None, results_dir=None):
    model.eval()
    # loader.dataset.update_mode(True)
    val_loss = 0.
    all_preds, all_labels = [], []
    # val_error = 0.
    
    # prob = np.zeros((len(loader), n_classes))
    # labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits = model(data)
            Y_hat = torch.argmax(logits, dim=1)

            all_preds.append(Y_hat.cpu())
            all_labels.append(label.cpu())
            val_loss += loss_fn(logits, label).item()
            
    val_loss /= len(loader)
    saveresult(epoch, results_dir, strategy, all_preds, all_labels)
