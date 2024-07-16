import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import math
from ipdb import set_trace as st
import args
from args import get_parser
def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

nq = 16
num_layers = 2
hidden_size = 2048
vis_dim = 512
num_head = 8
droup_out = 0.1
class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        # hidden_size = c4_dims//factor
        # c4_size = c4_dims
        # c3_size = c4_dims//(factor**1)
        # c2_size = c4_dims//(factor**2)
        # c1_size = c4_dims//(factor**3)

        # self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_4 = nn.BatchNorm2d(hidden_size)
        # self.relu1_4 = nn.ReLU()
        # self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_4 = nn.BatchNorm2d(hidden_size)
        # self.relu2_4 = nn.ReLU()

        # self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_3 = nn.BatchNorm2d(hidden_size)
        # self.relu1_3 = nn.ReLU()
        # self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_3 = nn.BatchNorm2d(hidden_size)
        # self.relu2_3 = nn.ReLU()

        # self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_2 = nn.BatchNorm2d(hidden_size)
        # self.relu1_2 = nn.ReLU()
        # self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_2 = nn.BatchNorm2d(hidden_size)
        # self.relu2_2 = nn.ReLU()
        self.query_neck = FQFPN(in_channels=[256, 512, 1024], out_channels=[256, 512, 1024],query_number=nq)
        
        self.query_gen1 = Query_Generation(d_model=512, input_dim=900,word_len=20)
        self.query_gen2 = Query_Generation(d_model=512, input_dim=900,word_len=20)
        self.query_gen3 = Query_Generation(d_model=512, input_dim=900,word_len=20)
        self.query_gen4 = Query_Generation(d_model=512, input_dim=900,word_len=20)
        self.fvq_proj2 = conv_layer(vis_dim,nq,3,1)
        self.fvq_proj3 = conv_layer(vis_dim,nq,3,1)
        self.fvq_proj4 = conv_layer(vis_dim,nq,3,1)
        self.balance = Query_Balance(d_model=512,
                                     query_number=nq,
                                     nhead=8,
                                     dropout=0.1)
        # self.proj = Projector(512, 256, 3,16)
        self.mask_proj = Projectorr(512,256,3)
        self.decoder1 = TransformerDecoder(num_layers,vis_dim,num_head,hidden_size,droup_out,False)
        self.decoder2 = TransformerDecoder(num_layers,vis_dim,num_head,hidden_size,droup_out,False)
        self.decoder3 = TransformerDecoder(num_layers,vis_dim,num_head,hidden_size,droup_out,False)
        self.relu2_2 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(1, 2, 1)
        self.reconlanguegae = LanguageReconstruct(512,512)
        self.t1 = nn.Parameter(torch.Tensor(1))
        self.t2 = nn.Parameter(torch.Tensor(1))
        self.initialize_parameters()
        self.reconlanguegae = LanguageReconstruct(vis_dim,vis_dim)
    def initialize_parameters(self):
        # 使用均匀分布进行随机初始化，范围为[0, 1]
        torch.nn.init.uniform_(self.t1, 0, 1)
        torch.nn.init.uniform_(self.t2, 0, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1,l_feats, class_token,l_mask):
        # fuse Y4 and Y3
        # if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
        #     x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        # x = torch.cat([x_c4, x_c3], dim=1)
        # x = self.conv1_4(x)
        # x = self.bn1_4(x)
        # x = self.relu1_4(x)
        # x = self.conv2_4(x)
        # x = self.bn2_4(x)
        # x = self.relu2_4(x)
        # # fuse top-down features and Y2 features
        # if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
        #     x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        # x = torch.cat([x, x_c2], dim=1)
        # x = self.conv1_3(x)
        # x = self.bn1_3(x)
        # x = self.relu1_3(x)
        # x = self.conv2_3(x)
        # x = self.bn2_3(x)
        # x = self.relu2_3(x)
        # # fuse top-down features and Y1 features
        # if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
        #     x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        # x = torch.cat([x, x_c1], dim=1)
        # x = self.conv1_2(x)
        # x = self.bn1_2(x)
        # x = self.relu1_2(x)
        # x = self.conv2_2(x)
        # x = self.bn2_2(x)
        # x = self.relu2_2(x)
        vis = (x_c2, x_c3, x_c4)
        fvq,fq = self.query_neck(vis,class_token)
        fgq1 = self.query_gen1(fvq,l_feats,l_mask)
        
        # score = self.balance(fgq)
        # fgq = fgq*score
        fbq = torch.sum(fgq1,dim=0,keepdim=False)
        b, c, h, w = fq.size()
        
        out1 = self.decoder1(fq, fgq1)
        out1 = out1.reshape(b,c,h,w)
        fgq2 = self.query_gen2(self.fvq_proj2(out1),l_feats,l_mask)
        fgq2 = fgq1 + (self.t1*fgq2)
        
        out2 = self.decoder2(out1, fgq2)
        out2 = out2.reshape(b,c,h,w)
        fgq3 = self.query_gen3(self.fvq_proj3(out2),l_feats,l_mask)
        fgq3 = fgq2 + (self.t2*fgq3)
        # fq = fq.reshape(b, c, h, w)
        out3 = self.decoder3(out2, fgq3)
        out3 = out3.reshape(b, c, h, w)
        pred = self.mask_proj(out3, class_token)
        # print(predd.shape)
        # pred = self.proj(fq,fgq,score)
        fgq4 = self.query_gen4(self.fvq_proj4(out3),l_feats,l_mask)
        re_loss = self.reconlanguegae(l_feats,class_token,fgq4)
        
        return pred,re_loss

class FQFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 1024, 1024],
                 out_channels=[256, 512, 1024],query_number=12):
        super(FQFPN, self).__init__()
        # text projection
        # self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        self.fvq_proj = conv_layer(in_channels[1],query_number,3,1)
        self.state_proj = linear_layer(768,1024)
    def forward(self, imgs,class_token):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.state_proj(class_token).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
    
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)

        # fq = self.fvq_proj(fq)
        # b, 512, 26, 26
        
        return self.fvq_proj(fq),fq

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Query_Generation(nn.Module):
    def __init__(self,d_model=512, input_dim=900,word_len = 20):
        super(Query_Generation, self).__init__()
        self.text_proj1 = nn.Sequential(nn.Linear(768,d_model), nn.ReLU(True))
        self.fvq_proj1 = linear_layer(input_dim,d_model)
        self.global_proj1 = linear_layer(d_model,d_model*2)
        self.reu = nn.ReLU(True)
        # self.text_proj2 = nn.Sequential(nn.Linear(d_model*2,d_model), nn.ReLU(True))
        #self.cat = nn.Sequential(nn.Conv1d(in_dim=(word_len+1), out_dim=word_len, kernel_size=1, padding=0, stride=1))
        self.query_attention = QueryAttention(v_in_channels=900, l_in_channels=512, key_channels=512, value_channels=512)
    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)
    def forward(self,fq1,text,pad_mask):
        
         #b Nq 512
        # global_feat = self.global_proj1(global_feat)
        # global_feat = global_feat.unsqueeze(1)
        text = self.text_proj1(text.permute(0,2,1))
        # text = self.reu(text*global_feat)
        # text = self.text_proj2(text)
     
        B, C, H, W = fq1.size()
        fq1 = fq1.reshape(B,C,-1)

        #fq1 = self.fvq_proj1(fq1)
        _, Lt, Dt = text.size()

        text_pos = self.pos1d(Dt, Lt)
        
        text = text.permute(1, 0, 2)
        fq1_pos = self.pos2d(C, H, W)
        fq1 = fq1.permute(2, 0, 1)
        vis = self.with_pos_embed(fq1, fq1_pos).permute(1,0,2)
        l = self.with_pos_embed(text, text_pos).permute(1,2,0)
        pad_mask = pad_mask.permute(0,2,1)
        
        """         fgq,att_map = self.multihead_attn(query=self.with_pos_embed(fq1, fq1_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text,
                                   key_padding_mask=pad_mask) """
        out = self.query_attention(vis,l,pad_mask)
        fgq = out.permute(2,0,1)
        return fgq #(Nq=16,b=32,512)

class QueryAttention(nn.Module):
    def __init__(self, v_in_channels=676, l_in_channels=512, key_channels=512, value_channels=512):
        super(QueryAttention, self).__init__()
        # v:(b,v_in_channels,Nq=16)
        # l:(b,l_in_channels,n_words=17)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.key_proj = nn.Sequential(nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1))
        self.query_proj = nn.Sequential(nn.Conv1d(self.v_in_channels,self.key_channels,kernel_size=1, stride=1),nn.InstanceNorm1d(self.key_channels))
        self.value_proj = nn.Sequential(nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1))
                # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.value_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.value_channels),
        )
    def forward(self, vis, l, l_mask):

        #l = torch.zeros_like(l).cuda(non_blocking=True)
        query = self.query_proj(vis)
        key = self.key_proj(l)
        value = self.value_proj(l)

        #value = torch.zeros_like(value).cuda(non_blocking=True)

        #query = torch.zeros_like(query).cuda(non_blocking=True)
        
        #key = torch.zeros_like(key).cuda(non_blocking=True)
        
        query = query.permute(0, 2, 1)
        key = key * l_mask
        value = value * l_mask
        att_map = torch.matmul(query, key)
        att_map = (self.key_channels ** -.5) * att_map
        att_map = att_map + (1e4*l_mask - 1e4)
        att_map = F.softmax(att_map, dim=-1)
        out = torch.matmul(att_map, value.permute(0,2,1))
        out = self.W(out.permute(0, 2, 1))
        
        return out

class Query_Balance(nn.Module):
    def __init__(self,d_model,query_number,nhead,dropout):
        super().__init__()
        self.f1_project = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU())
        self.f2_project = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU())
        self.self_attn = nn.MultiheadAttention(d_model, 1,dropout=0)
        self.score_proj = nn.Sequential(nn.Linear(d_model, 1),nn.Softmax(dim=0))
        #nn.Softmax(dim=0)
        self.self_attn_norm = nn.LayerNorm(d_model)
    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)    

    def forward(self,fgq):
        #fgq Nq,b,C

        Lg, _, Dg = fgq.size()
        fq1 = self.f1_project(fgq)
        fq1_pos = self.pos1d(Dg,Lg)
        q = k = self.with_pos_embed(fq1, fq1_pos)
        fq1 = self.self_attn(q, k, value=fq1)[0]
        fq1 = self.self_attn_norm(fq1)
        score = self.score_proj(fq1)
        #fq2 = self.f2_project(fgq)
        #print('fq2:',fq2.shape)
        #print('score:',score.shape)
        #weighted_output = fq2 * score
        
        return score

class Projector(nn.Module):
    def __init__(self, word_dim=512, in_dim=256, kernel_size=3,query_number = 16):
        super().__init__()
        self.nq = query_number
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim , 3, padding=1),
            nn.Conv2d(in_dim, in_dim , 1))
        # textual projector
        out_dim =  (in_dim * kernel_size * kernel_size + 1)
        self.txt = nn.Linear(word_dim, out_dim)
        # self.out = conv_layer(1, 2 , 3, padding=1)
    def forward(self, x, word,score):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
    
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        
        x = x.reshape(1, B * C, H, W)
        
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
    
        word = self.txt(word).permute(1,0,2)
        weight_out = torch.zeros([B,1,H,W]).cuda(non_blocking=True)
        
        for i in range(self.nq):
            each_word = word[:,i,:]
            each_score = score[i,:,:]
            
            weight, bias = each_word[:, :-1], each_word[:, -1]
            weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
            
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
            each_out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
            
            each_out = each_out.transpose(0, 1)*each_score.unsqueeze(dim=1).unsqueeze(dim=2)
            weight_out = weight_out + each_out
            # out = self.out(weight_out)
        return weight_out

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, fgq):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        Lg, Bg, Dg = fgq.size()
        B, C, H, W = vis.size()
    
        
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        
        fgq_pos = self.pos1d(Dg,Lg)

        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        
        
        #add = fgq[1,:,:].unsqueeze(0)
        
        
        #fgq = fgq.cuda(non_blocking=True)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output,fgq, vis_pos, fgq_pos)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):

        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask = None):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        #print('vis_o:',vis.shape)
        #print('txt_o:',txt.shape)
        # Self-Attention
        
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        #print('vis_o:',vis.shape)
        #print('fgq:',txt.shape)
        vis2,att_map= self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)
        
        #print('att_map:',att_map.shape)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis

class LanguageReconstruct(nn.Module):
    def __init__(self,d_model=512,l_channels=512,nhead=8,dropout=0.1):
        super(LanguageReconstruct, self).__init__()
        self.state_channels = l_channels=512
        self.state_proj = linear_layer(768,768)
        self.Wp = nn.Sequential(nn.Conv1d(768, 512, kernel_size=1, stride=1),nn.ReLU())
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        self.Wr = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, stride=1),nn.ReLU(),nn.Conv1d(512, 512, kernel_size=1, stride=1),nn.ReLU())

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512
    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)
    def forward(self,word,state,fgq):
        _, Lt, Dt = word.size()
      
        # Lg, Bg, Dg = fgq.size()
        # position encoding
        word_pos = self.pos1d(Dt, Lt)
        word = word.permute(1, 0, 2)
        word = self.with_pos_embed(word, word_pos).permute(1,2,0)
        word = word.permute(0,2,1)
        state = self.state_proj(state).unsqueeze(-1)
        word = torch.cat([word,state],dim=2)
        word = self.Wp(word)
        
        word = torch.mean(word,dim=2,keepdim=True)
        # mean_word = torch.mean(word, dim=1, keepdim=True)

        recon_language = fgq
        re_l = self.Wr(recon_language.permute(0,2,1))

        
        re_l = torch.mean(re_l,dim=0,keepdim=True).permute(1,0,2)
        re_l = re_l.permute(2,1,0)
        word = word.permute(0,2,1)

        return F.mse_loss(re_l, word)

class Projectorr(nn.Module):
    def __init__(self, word_dim=512, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(768, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        
        weight, bias = word[:, :-1], word[:, -1]
        
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        #print('out:',out.shape)
        # b, 1, 104, 104
        return out