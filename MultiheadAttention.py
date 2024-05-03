from PyhtonImport import *


# tính toán điểm attention và giá trị attention thông qua 3 ma trận được tách ra từ đầu ra của lớp Linear
# input: 3 x d_model -> Q, K, V
def compute_qkv(q, k, v, mask):
    d_k = q.size()[-1]
    scaled = torch.matmul(input=q, other=k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask != None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(input=scaled, dim=-1)
    values = torch.matmul(input=attention, other=v)
    return values.to(device), attention.to(device)


# tính toán giá trị chú ý của đầu vào (dùng trong cả encoder và decoder)
class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads
        self.qkv_linear = nn.Linear(in_features=d_model, out_features=3 * d_model)
        self.out_linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x, mask):
        batch_size, max_sequence_length, input_dims = x.size()
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dims)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v, = qkv.chunk(3, dim=-1)
        values, attention = compute_qkv(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, self.d_model)
        out = self.out_linear(values)
        return out.to(device)
    

# tính toán giá trị chú ý của (một phần của decoder) nó sẽ dùng output của encoder để tính toán
# kết quả y dự đoán
class MultiheadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads
        self.q_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.kv_linear = nn.Linear(in_features=d_model, out_features=2 * d_model)
        self.out_linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x, y, mask):
        batch_size, max_sequence_length, d_model = x.size()
        q = self.q_linear(x)
        kv = self.kv_linear(y)
        q = q.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dims)
        kv = kv.reshape(batch_size, max_sequence_length, self.num_heads, 2 * self.head_dims)
        q = q.permute(0, 2, 1, 3)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = compute_qkv(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, d_model)
        out = self.out_linear(values)
        return out.to(device)

# batch_size = 30
# max_sequence_length = 200
# d_model = 512
# num_heads = 8

# x = torch.rand(batch_size, max_sequence_length, d_model)
# attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
# print(attention(x, mask=None))