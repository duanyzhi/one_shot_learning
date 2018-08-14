from utils import *

loss = read_csv('data/csv/loss.csv')[1:]
_axis = [int(it[0]) for it in loss]
_loss = [float(it[1]) for it in loss]

plot_scatter('data/png/loss.png', [_loss], ['5-ways-5shot'],  "One Shot Learning", ylabel="Loss")

train_acc = read_csv('data/csv/train_acc.csv')[1:]
_axis = [int(it[0]) for it in train_acc]
tr_acc = [float(it[1]) for it in train_acc]

test_acc = read_csv('data/csv/test_acc.csv')[1:]
te_acc = [float(it[1]) for it in test_acc]

plot_scatter('data/png/acc.png',  [tr_acc, te_acc], ['Train', 'Test'],  "One Shot Learning", ylabel="Accuracy")
