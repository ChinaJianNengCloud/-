import os
import argparse
import torch
from openpyxl.styles.builtins import output
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from data import Dataset
import config as cfg
from tensorboardX import SummaryWriter

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    plt.show()

def plot_acc(train_accs, val_accs):
    plt.figure()
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve.png")
    plt.show()

def plot_auc(train_aucs, val_aucs):
    plt.figure()
    plt.plot(train_aucs, label="Training AUC")
    plt.plot(val_aucs, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("AUC Curve")
    plt.savefig("auc_curve.png")
    plt.show()


from sklearn.metrics import roc_curve, roc_auc_score, auc,accuracy_score
from collections import Counter
# Determine if CUDA is available and set the appropriate device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=5000,  # 全称一般是加--
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=2048,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


#     writer.add_scalar('Val Acc',  acc,e)

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def evaluate(e, model, data, vocab_size, writer):
    model.eval()
    correct = 0
    val_step = 0
    predict = {}
    total_loss = 0
    data.val_index = 0
    data.val_finish = False
    y_true = []
    y_pred = []

    while not data.val_finish:
        val_step += 1
        file_indexs, src, trg = data.get_val_batch()

        src = torch.Tensor(src).to(device)
        trg = torch.LongTensor(trg).to(device)
        src = Variable(src)
        trg = Variable(trg)
        output = model(src, trg, teacher_forcing_ratio=0.0).to(device)
        probabilities = F.softmax(output[1:].view(-1, vocab_size), dim=1)
        max_prob_indices = torch.argmax(probabilities, dim=1)
        # positive_probabilities = probabilities[:, 1]
        y_pred.extend(max_prob_indices.cpu().detach().numpy().tolist())

        pred = output[1:].view(-1, vocab_size).data.max(1, keepdim=True)[1]
        correct += pred.eq(trg[1:].data.view_as(pred)).cpu().sum()
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1))
        total_loss += loss.item()

        y_true.extend(trg[1:].contiguous().view(-1).cpu().numpy().tolist())

        for index_1, file_index in enumerate(file_indexs):
            for index_2, name in enumerate(file_index):
                if name not in predict.keys():
                    predict[name] = []
                predict[name].append(int(pred[index_1 * cfg.Batch_size + index_2][0].cpu().numpy()))

    auc_score = roc_auc_score(y_true, y_pred)

    tmp = []
    with open('./result.txt', 'w') as f:
        for key, item in predict.items():
            label = data.labels[key]
            answer = Counter(item).most_common(1)[0][0]
            tmp.append(answer)
            f.write(key + '\t' + str(label) + '\t' + str(answer) + '\n')
    x = data.get_label()
    y = list(x)
    a, b = data.get_trainval_list()
    c = []
    for j in b:
        if j in y:
            c.append(x[j])
    acc = accuracy_score(c[:-1], tmp)
    writer.add_scalar('Val Loss', total_loss / val_step, e)
    writer.add_scalar('Val Acc', acc, e)
    writer.add_scalar('Val AUC', auc_score, e)  # Log the AUC score

    return total_loss / val_step, acc, auc_score  # Now returning the AUC score

def train(e, model, optimizer, data, vocab_size, grad_clip, writer):
    model.train()
    total_loss = 0
    train_step = 50
    total_correct = 0  # 初始化 total_correct
    total_elements = 0

    all_y_true = []  # Collect true labels for all batches in this list
    all_y_pred = []  # Collect predicted values for all batches in this list

    for _, _ in enumerate(range(train_step)):
        src, trg = data.get_train_batch()

        src = torch.Tensor(src).to(device)
        trg = torch.LongTensor(trg).to(device)

        optimizer.zero_grad()
        output = model(src, trg).to(device)

        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        probabilities = F.softmax(output[1:].view(-1, vocab_size), dim=1)
        # positive_probabilities = probabilities[:, 1]
        max_prob_indices = torch.argmax(probabilities, dim=1)
        all_y_pred.extend(max_prob_indices.cpu().detach().numpy().tolist())
        all_y_true.extend(trg[1:].contiguous().view(-1).cpu().numpy().tolist())

    # 计算整个 epoch 的 AUC
    roc_auc = roc_auc_score(all_y_true, all_y_pred)

    # 计算 train acc
    total_elements = len(all_y_true)
    total_correct = sum([1 for true, pred in zip(all_y_true, all_y_pred) if true == round(pred)])
    acc = total_correct / total_elements

    writer.add_scalar('Training Loss', total_loss / train_step, e)
    writer.add_scalar('Training Acc', acc, e)
    writer.add_scalar('Training Auc', roc_auc, e)

    return total_loss / train_step, acc, roc_auc


def main():
    folder_path = "./output"
    # 判断文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    writer = SummaryWriter(log_dir="./output")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_aucs = []  # 创建一个新的列表来存储训练AUC值
    val_aucs = []  # 创建一个新的列表来存储验证AUC值
    xFPR = []
    xTPR = []

# 路径存在，并且是一
    try:
        args = parse_arguments()
        hidden_size = 128
        embed_size = 128
        en_size = cfg.Class_num
        data = Dataset()
        print("[!] Instantiating models...")
        encoder = Encoder(cfg.Input_dim, hidden_size, n_layers=2).to(device)
        decoder = Decoder(embed_size, hidden_size, en_size, n_layers=1, dropout=0.5).to(device)
        seq2seq = Seq2Seq(encoder, decoder).to(device)
        print(seq2seq)
        best_val_loss = None
        lr = args.lr

    # optimizer = optim.Adam(seq2seq.parameters(), lr=lr)  # Moved optimizer outside the loop
        optimizer = optim.SGD(seq2seq.parameters(), lr=lr)

        for e in range(1, args.epochs + 1):
            train_loss, train_acc,train_auc = train(e, seq2seq, optimizer, data, en_size, args.grad_clip,writer)
            val_loss, val_acc, val_auc = evaluate(e,seq2seq, data, en_size,writer)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_aucs.append(train_auc)  # 在每个epoch结束时，将train_auc值添加到train_aucs列表中
            val_aucs.append(val_auc)  # 在每个epoch结束时，将val_auc值添加到val_aucs列表中

            print(f"train_acc{train_acc}")
            print(f"val_acc{val_acc.item()}")

            # print("[Epoch:{}] | train_loss:{:.3f} | val_loss:{:.3f}".format(str(e).zfill(3), train_loss, val_loss))
            print(f"[Epoch:{str(e).zfill(3)}] | "
                  f"train_loss:{train_loss:.3f} | "
                  f"val_loss:{val_loss:.3f} | "
                  f"train_auc:{train_auc:.3f} | "
                  f"val_auc:{val_auc:.3f}")
            plot_loss(train_losses, val_losses)
            plot_acc(train_accs, val_accs)
            plot_auc(train_aucs, val_aucs)  # 在每个epoch结束时，调用plot_auc函数来绘制和保存AUC图表
            # plot_roc(xFPR, xTPR)
            if not best_val_loss or val_loss < best_val_loss:
                print("[!] saving model...")
                if not os.path.isdir(".save"):
                    os.makedirs(".save")
                torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
                best_val_loss = val_loss


    except KeyboardInterrupt as e:
        print("[STOP]", e)
    finally:
        writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)