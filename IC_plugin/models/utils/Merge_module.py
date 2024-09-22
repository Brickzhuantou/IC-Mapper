#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyLineEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolyLineEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        # input_seq shape: (batch_size, seq_length, input_size)
        output, (hidden_state, cell_state) = self.lstm(input_seq)
        return hidden_state, cell_state

class PolyLineDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolyLineDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, input_feature, hidden_state, cell_state):
        # input_feature shape: (batch_size, 1, input_size), for LSTM
        output, (hidden_state, cell_state) = self.lstm(input_feature, (hidden_state, cell_state))
        # output shape: (batch_size, 1, hidden_size)
        output = self.fc(output.squeeze(1))
        # output shape: (batch_size, output_size)
        return output, hidden_state, cell_state

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = PolyLineEncoder(input_size, hidden_size)
        self.decoder = PolyLineDecoder(input_size, hidden_size, output_size)

    def forward(self, input_seq, target_seq_length=20):
        batch_size = input_seq.size(0)
        # Encoding
        hidden_state, cell_state = self.encoder(input_seq)
        # Initial input for decoder (could be a learnable parameter or fixed)
        decoder_input = torch.zeros(batch_size, 1, input_seq.size(2), device=input_seq.device)
        
        outputs = []
        for _ in range(target_seq_length):
            output, hidden_state, cell_state = self.decoder(decoder_input, hidden_state, cell_state)
            outputs.append(output)
            # Use current output as input for next step
            decoder_input = output.unsqueeze(1)
        
        outputs = torch.stack(outputs, dim=1)
        # outputs shape: (batch_size, target_seq_length, output_size)
        return outputs



class PointEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(PointEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, num_points, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Global max pooling
        x, _ = torch.max(x, 1)
        return x
    
    
class PointEncoderKV(nn.Module):
    def __init__(self, input_size, output_size):
        super(PointEncoderKV, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, num_points, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PointDecoder(nn.Module):
    def __init__(self, input_size, num_points, point_dim):
        super(PointDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_points * point_dim)
        self.num_points = num_points
        self.point_dim = point_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(-1, self.num_points, self.point_dim)
        return x
    

class PointTransformerDecoder(nn.Module):
    def __init__(self, emb_dim, num_points, point_dim):
        super(PointTransformerDecoder, self).__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.emb_dim = emb_dim

        # 位置编码层
        self.pos_encoder = nn.Linear(point_dim, emb_dim)
        # 查询（query）初始化层
        self.query_init = nn.Linear(emb_dim, emb_dim)
        # Transformer解码器层
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        # 输出层，将解码器输出转换为期望形状
        self.output_layer = nn.Linear(emb_dim, 2)

    def forward(self, hist_kv, pos_init, refined_inst_q):
        batch_size = hist_kv.size(0)
        
        # 位置编码
        pos_encoding = self.pos_encoder(pos_init)  # [Batch_size, num_points, emb_dim]

        # 初始化query
        query = self.query_init(refined_inst_q)  # [Batch_size, emb_dim]
        # 为每个点复制query
        query = query.unsqueeze(1).repeat(1, self.num_points, 1)  # [Batch_size, num_points, emb_dim]
        # query = query + pos_encoding  # [Batch_size, num_points, emb_dim]
        query = pos_encoding #TODO: 验证只用初始检测点的位置会有什么影响；
        # 使用hist_kv作为key和value
        key = value = hist_kv  # [Batch_size, num_points, emb_dim]

        # Transformer解码器
        decoded = self.transformer_decoder(tgt=query, memory=hist_kv)

        # 输出层调整到期望的输出形状
        output = self.output_layer(decoded).sigmoid()  # [Batch_size, num_points, 2]

        return output



class PointNetAutoencoder(nn.Module):
    def __init__(self, num_points, point_dim):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointEncoder(input_size=point_dim, output_size=256)
        self.decoder = PointDecoder(input_size=256, num_points=num_points, point_dim=point_dim)

    def forward(self, x):
        global_features = self.encoder(x)
        reconstructed_points = self.decoder(global_features)
        return reconstructed_points

#%%

if __name__ == "__main__":
    # Example usage
    num_points = 20
    point_dim = 3  # Assuming 3D points (x, y, z)

    model = PointNetAutoencoder(num_points=num_points, point_dim=point_dim)

    # Example input: batch of sets, each with 20 points with x,y,z coordinates
    input_points = torch.rand(5, num_points, point_dim)  # (batch_size, num_points, input_size)

    # Forward pass through the model
    output_points = model(input_points)

    print(output_points.shape)  # Expected shape: (batch_size, num_points, point_dim)



    # Example usage
    input_size = 2  # x, y coordinates
    hidden_size = 128
    output_size = 2  # Predicted x, y coordinates

    seq2seq_model = Seq2Seq(input_size, hidden_size, output_size)

    # Example input: batch of sequences, each sequence has 20 points with x,y coordinates
    input_seq = torch.rand(5, 20, input_size)  # (batch_size, seq_length, input_size)

    # Forward pass through Seq2Seq model
    output_seq = seq2seq_model(input_seq)

    print(output_seq.shape)  # Expected shape: (batch_size, target_seq_length, output_size)

# # %%
# import torch
# import torch.nn as nn
# from Merge_module import PolyLineDecoder
# import unittest

# class TestPolyLineDecoder():
#     def setUp(self):
#         self.input_size = 10
#         self.hidden_size = 20
#         self.output_size = 30
#         self.batch_size = 5
#         self.decoder = PolyLineDecoder(self.input_size, self.hidden_size, self.output_size)

#     def test_init_hidden(self):
#         hidden_state, cell_state = self.decoder.init_hidden(self.batch_size)

#         # self.assertEqual(hidden_state.shape, (1, self.batch_size, self.hidden_size))
#         # self.assertEqual(cell_state.shape, (1, self.batch_size, self.hidden_size))

#     def test_forward(self):
#         input_feature = torch.randn(self.batch_size, 1, self.input_size)
#         hidden_state, cell_state = self.decoder.init_hidden(self.batch_size)
#         output, hidden_state, new_cell_state = self.decoder.forward(input_feature, hidden_state, cell_state)
        
#         print(hidden_state.shape)
#         print(cell_state.shape)
#         # self.assertEqual(output.shape, (self.batch_size, self.output_size))
#         # self.assertEqual(hidden_state.shape, (1, self.batch_size, self.hidden_size))
#         # self.assertEqual(new_cell_state.shape, (1, self.batch_size, self.hidden_size))
#         # self.assertNotEqual(cell_state, new_cell_state)


# # %%
# test= TestPolyLineDecoder()
# test.setUp()
# test.test_init_hidden()
# test.test_forward()
# # %%

#%%
# encoder = PolyLineEncoder(2, 128)
# hist_traj = torch.rand(5, 20, 2)
# hidden_state, cell_state = encoder(hist_traj)
# output, (hidden_state, cell_state) = encoder.lstm(hist_traj)
# %%
