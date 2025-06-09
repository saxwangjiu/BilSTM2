#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score

# === Step 1: 数据加载 ===
# 加载 .npz 文件，x 是一个 list，y 是一个 list 或 array，表示标签(0 或 1)
data = np.load("data/data.npz", allow_pickle=True)
x_raw = data['x']  # 输入序列
y_raw = data['y']  # 对应标签

# === Step 2: 定义自定义 Dataset 类 ===
# 该类用于将原始的序列数据(x)和标签数据(y)封装成 PyTorch 可以读取的数据集格式
class SequenceDataset(Dataset):
    def __init__(self, x_list, y_list):
        """
        初始化函数：将输入的原始序列列表和标签列表转换为 PyTorch Tensor 格式。
        参数:
            x_list: 一个列表，每个元素是一个序列(长度可变的整数列表)。
            y_list: 一个列表，每个元素是对应的标签(一般为0或1)。
        """
        # 将每个序列转为 LongTensor(整型张量)，适用于嵌入层的输入
        self.x = [torch.tensor(seq, dtype=torch.long) for seq in x_list]
        # 将标签列表整体转为 FloatTensor，适用于 BCELoss(用于二分类)
        self.y = torch.tensor(y_list, dtype=torch.float)

    def __len__(self):
        """
        返回数据集的大小(即样本数量)
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        根据索引返回第 idx 个样本的数据
        返回:
            - 第 idx 个输入序列(LongTensor)
            - 第 idx 个标签(float 类型)
        """
        return self.x[idx], self.y[idx]


# collate_fn 用于对变长序列进行 padding，使其能组成 batch
def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_padded = pad_sequence(x_batch, batch_first=True, padding_value=0)  # 对齐序列,不足的用0填充
    y_tensor = torch.tensor(y_batch, dtype=torch.float)
    return x_padded, y_tensor  #f返回张量

# === Step 3: 定义 BiLSTM 模型 ===
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)    #把离散的词转化为连续的向量来表示,包含了词的语义特征和上下文信息，0是之前填充使得序列长度一样的数字，不参与训练
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向 LSTM 所以是 2 倍 hidden_dim
        self.sigmoid = nn.Sigmoid()  # 输出概率(0-1)

    def forward(self, x):                                          # Sequence length：每个样本（序列）的长度，即每个序列有多少个词或字符 
        embedded = self.embedding(x)  # 词嵌入 [B, T] -> [B, T, D] #Batch size：样本的数量，一次处理多少个序列
        output, _ = self.lstm(embedded)  # LSTM 输出 [B, T, 2H]    #Embedding dim：每个词或字符被转换成多少维的向量（词向量维度） 
        pooled = torch.max(output, dim=1).values  # 使用 max pooling 取每个样本的时间最大特征 [B, 2H]
        out = self.fc(pooled)  # 全连接层 [B, 1]
        return self.sigmoid(out).squeeze(1)  # squeeze 后变为 [B]

# === Step 4: 划分训练集和测试集 ===
# 构造 Dataset
full_dataset = SequenceDataset(x_raw, y_raw)

# 划分比例(80%训练，20%测试)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 使用 random_split 划分数据集
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 构造 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 词表大小(假设序列中最大编号就是最大词+1)
vocab_size = max(max(seq) for seq in x_raw) + 1

# 选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 创建模型、损失函数、优化器
model = BiLSTMClassifier(vocab_size).to(device)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Step 5: 模型训练 ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)  # 模型预测
        loss = criterion(preds, y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# === Step 6: 测试集评估 ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():  # 不计算梯度，节省内存，测试不需要反向传播
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch)
        preds_label = (preds > 0.5).cpu().numpy()  # 概率 > 0.5 判为正类
        all_preds.extend(preds_label)
        all_labels.extend(y_batch.numpy())

# 打印评估指标
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))

# === Step 7: 保存模型权重 ===
save_path = "project.pt"
torch.save(model.state_dict(), save_path)
print(f"模型权重已保存到 {save_path}")

# %%
