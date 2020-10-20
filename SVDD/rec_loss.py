import torch


rec_loss = torch.load('rec_loss.pt')
labels = torch.load('labels.pt')

best_acc, best_th = 0, 0

for th in range(0, 1000, 1):
    th = 1.0*th/1000.0
    pred = (rec_loss <= th).float()
    acc = torch.mean((labels == pred).float())
    if acc >= best_acc:
        best_acc = acc
        best_th = th
    print(th, acc.item())


print('Best stats =>> threshold = ', best_th, ' accuracy = ', best_acc)
