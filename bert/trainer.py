import time
import torch
from os.path import join as jp
from tqdm import tqdm

def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc

def train(epochs, device, model, train_loader, val_loader, optimizer, path_save):  
  total_step = len(train_loader)
  best_acc_val = 0.0 # best val acc to save best model
  for epoch in range(epochs):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      # prediction = model(pair_token_ids, mask_ids, seg_ids)
      loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      # loss = criterion(prediction, labels)
      acc = multi_acc(prediction, labels)

      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        # prediction = model(pair_token_ids, mask_ids, seg_ids)
        loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
        # loss = criterion(prediction, labels)
        acc = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)
    if val_acc > best_acc_val:
        # save model and
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, jp(path_save, 'best_model.pt'))
        best_acc_val = val_acc
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def infer(device, model, test_loader, path_output, the_name='infer'):
    model.eval()
    all_ans = torch.zeros((len(test_loader.dataset), 1), dtype=torch.uint8)
    all_probs = torch.zeros((len(test_loader.dataset), 3), dtype=torch.float32)
    offset = 0
    print('Running inference ...')
    with torch.no_grad():
        for batch_idx, (pair_token_ids, mask_ids, seg_ids) in enumerate(tqdm(test_loader)):
            batch_size = mask_ids.shape[0]
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            prediction = model(pair_token_ids, 
                                token_type_ids=seg_ids, 
                                attention_mask=mask_ids)
            lbls = torch.argmax(prediction.logits, axis=-1)
            # stack answers
            all_ans[offset:offset+batch_size] = lbls.unsqueeze(-1).detach().cpu()
            all_probs[offset:offset+batch_size] = torch.softmax(prediction.logits, axis=-1).detach().cpu()
            offset = offset+batch_size

    # save answers
    torch.save(all_ans, jp(path_output, the_name + '_ans.pt'))
    torch.save(all_probs, jp(path_output, the_name + '_probs.pt'))
    return all_ans