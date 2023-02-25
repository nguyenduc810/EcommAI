import os
import sys
sys.path.append(os.pardir)
sys.path.insert(1,'/kaggle/working/EcommAI')
import numpy as np
from sklearn.metrics import roc_auc_score
from dbrl.utils import init_param_dssm
import torch
from tqdm import tqdm


def pretrain_model(
        model,
        train_loader,
        eval_loader,
        n_epochs,
        criterion,
        criterion_type,
        optimizer,
        device, pretrain
):
    if pretrain == 1:
        checkpoint = torch.load('model_dssm.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        max_roc = checkpoint['eval_roc']
        print('Load pretrain successful')
    else:
      max_roc = 0
      init_param_dssm(model)
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = []
        for data in tqdm(train_loader):
            data = {k: v.to(device) for k, v in data.items()}
            if criterion_type == "cosine":
                user, item, _ = model(data)
                loss = criterion(user, item, data["label"])
            elif criterion_type == "bce":
                _, _, logits = model(data)
                loss = criterion(logits, data["label"])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_loss.append(loss.item())
        
        with torch.no_grad():
            model.eval()
            eval_loss = []
            eval_roc = []
            for data in tqdm(eval_loader):
                data = {k: v.to(device) for k, v in data.items()}
                label = data["label"]
                if criterion_type == "cosine":
                    user, item, logits = model(data)  # [~oov]
                    e_loss = criterion(user, item, label)
                elif criterion_type == "bce":
                    _, _, logits = model(data)
                    e_loss = criterion(logits, label)
                eval_loss.append(e_loss.item())
                pred = torch.sigmoid(logits)
                eval_roc.append(
                    roc_auc_score(
                        label.cpu().numpy(), pred.cpu().numpy()
                    )
                )
        print(
            f"epoch {epoch}, train_loss: {np.mean(train_loss):.4f}, "
            f"eval loss: {np.mean(eval_loss):.4f}, "
            f"eval roc: {np.mean(eval_roc):.4f}"
        )

        if np.mean(eval_roc) > max_roc:
            max_roc = np.mean(eval_roc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_roc': max_roc}
                , 'model_dssm.pt')
            print("model saved ! epoch {0}".format(epoch))

