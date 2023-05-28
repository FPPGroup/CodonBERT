import math
import torch

import random
import numpy as np
import torch.nn.functional as FC
from torch import nn, einsum

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat


# helpers
G = ['GGT','GGC','GGA','GGG']
A = ['GCT','GCC','GCA','GCG']
V = ['GTT','GTC','GTA','GTG']
L = ['CTT','CTC','CTA','CTG','TTA','TTG']
I = ['ATT','ATC','ATA']
P = ['CCT','CCA','CCG','CCC']
F = ['TTT','TTC']
Y = ['TAT','TAC']
W = ['TGG']
S = ['TCT','TCA','TCC','TCG','AGT','AGC']
T = ['ACT','ACC','ACG','ACA']
M = ['ATG']
C = ['TGT','TGC']
N = ['AAT','AAC']
Q = ['CAA','CAG']
D = ['GAT','GAC']
E = ['GAA','GAG']
K = ['AAA','AAG']
R = ['CGT','CGC','CGG','CGA','AGA','AGG']
H = ['CAT','CAC']
X = ['TAA','TAG','TGA']
homonym_codon = {'G':G,'A':A,'V':V,'L':L,'I':I,'P':P,'F':F,'Y':Y,'W':W,'S':S,'T':T,'M':M,'C':C,'N':N,'Q':Q,'D':D,'E':E,'K':K,'R':R,'H':H,'X':X} 
dict_raw_int = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, 'a': 29, 'b': 30, 'c': 31, 'd': 32, 'e': 33, 'f': 34, 'g': 35, 'h': 36, 'i': 37, 'j': 38, 'k': 39, 'l': 40, 'm': 41, 'n': 42, 'o': 43, 'p': 44, 'q': 45, 'r': 46, 's': 47, 't': 48, 'u': 49, 'v': 50, 'w': 51, 'x': 52, 'y': 53, 'z': 54, '0': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, ':': 65, ';': 66}
codon_int = {'GGT':'a','GGC':'b','GGA':'c','GGG':'d','GCT':'e','GCC':'f','GCA':'g','GCG':'h','GTT':'i','GTC':'j','GTA':'k','GTG':'m','CTT':'l','CTC':'n','CTA':'o','CTG':'p','TTA':'q','TTG':'r','ATT':'s','ATC':'t','ATA':'u',
           'CCT':'v','CCA':'w','CCG':'x','CCC':'y','TTT':'z','TTC':'A','TAT':'B','TAC':'C','TGG':'D','TCT':'E','TCA':'F','TCC':'G','TCG':'H','AGT':'I','AGC':'J',
           'ACT':'K','ACC':'M','ACG':'L','ACA':'N','ATG':'O','TGT':'P','TGC':'Q','AAT':'R','AAC':'S','CAA':'T','CAG':'U','GAT':'V','GAC':'W',
           'GAA':'X','GAG':'Y','AAA':'Z','AAG':'1','CGT':'2','CGC':'3','CGG':'4','CGA':'5','AGA':'6','AGG':'7','CAT':'8','CAC':'9','TAA':'0','TAG':':','TGA':';'}

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def GC_con(seq):
    length = len(seq)
    G_num = seq.count('G')
    C_num = seq.count('C')
    GC_content = (G_num+C_num)/length
    GC_content = round(GC_content, 4)
    return GC_content
# helper classes

def convert_list_to_dict(dict_raw_int,value):
    # dict_int2aa = {}
    # for k, v in dict_raw_int.items():
    #     dict_int2aa[v] = k
    # return dict_int2aa[value]

    # print('value:',value)
    # return [dict[str(i)] for i in lst]
    # return [k for k, v in dict_raw_int.items() if v == value]
    return [k for k, v in dict_raw_int.items() if v == (value)][0]

def DNA_to_AA(DNA_seq):
    AA_list = ""
    start = 0
    end = 3
    DNA_seq = DNA_seq.replace('U','T')
    while(end<=len(DNA_seq)+1):
        codon = DNA_seq[start:end]
        start+=3
        end+=3
        # print(codon)
        for AA,codons in homonym_codon.items():
            if codon in codons:
                AA_list += AA
    return AA_list

def annotation_pre_to_AA(pre_list):
    #概率分布还原成字母
    raw_label = ''
    for i in range(len(pre_list)):
        raw_label=raw_label + str(convert_list_to_dict(dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    # print(raw_label)
    DNA_label = ''
    #字母还原成DNA
    for i in range(len(raw_label)):
        DNA_label=DNA_label + str(convert_list_to_dict(codon_int,raw_label[i])).replace('[','').replace(']','').replace('\'','')
    # print(DNA_label)
    #DNA还原成AA
    AA_result = DNA_to_AA(DNA_label)
    return DNA_label, AA_result




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class GlobalLinearSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask = None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)
        # q = q.Logsoftmax(dim = -1)
        # k = k.Logsoftmax(dim = -2)
        

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        out = einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head = 64,
        qk_activation = nn.Tanh()
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))

    def forward(self, x, context, mask = None, context_mask = None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        null_k, null_v = map(lambda t: repeat(t, 'd -> b h () d', b = b, h = h), (self.null_key, self.null_value))
        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype = torch.bool, device = device)

            if exists(context_mask):
                context_mask = FC.pad(context_mask, (1, 0), value = True)
            else:
                context_mask = torch.ones(b, j, dtype = torch.bool, device = device)

            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)
        # attn = sim.Logsoftmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# class Layer(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_global,
#         narrow_conv_kernel = 9,
#         wide_conv_kernel = 9,
#         wide_conv_dilation = 5,
#         attn_heads = 8,
#         attn_dim_head = 64,
#         attn_qk_activation = nn.Tanh(),
#         local_to_global_attn = False,
#         local_self_attn = False,
#         glu_conv = False
#     ):
#         super().__init__()

#         self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None

#         conv_mult = 2 if glu_conv else 1

#         self.narrow_conv = nn.Sequential(
#             nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
#             nn.GELU() if not glu_conv else nn.GLU(dim = 1)
#         )

#         wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

#         self.wide_conv = nn.Sequential(
#             nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
#             nn.GELU() if not glu_conv else nn.GLU(dim = 1)
#         )

#         self.local_to_global_attn = local_to_global_attn

#         if local_to_global_attn:
#             self.extract_global_info = CrossAttention(
#                 dim = dim,
#                 dim_keys = dim_global,
#                 dim_out = dim,
#                 heads = attn_heads,
#                 dim_head = attn_dim_head
#             )
#         else:
#             self.extract_global_info = nn.Sequential(
#                 Reduce('b n d -> b d', 'mean'),
#                 nn.Linear(dim_global, dim),
#                 nn.GELU(),
#                 Rearrange('b d -> b () d')
#             )

#         self.local_norm = nn.LayerNorm(dim)

#         self.local_feedforward = nn.Sequential(
#             Residual(nn.Sequential(
#                 nn.Linear(dim, dim),
#                 nn.GELU(),
#             )),
#             nn.LayerNorm(dim)
#         )

#         self.global_attend_local = CrossAttention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head, qk_activation = attn_qk_activation)

#         self.global_dense = nn.Sequential(
#             nn.Linear(dim_global, dim_global),
#             nn.GELU()
#         )

#         self.global_norm = nn.LayerNorm(dim_global)

#         self.global_feedforward = nn.Sequential(
#             Residual(nn.Sequential(
#                 nn.Linear(dim_global, dim_global),
#                 nn.GELU()
#             )),
#             nn.LayerNorm(dim_global),
#         )

#     def forward(self, tokens, annotation, mask = None):
#         if self.local_to_global_attn:
#             global_info = self.extract_global_info(tokens, annotation, mask = mask)############## no
#         else:
#             global_info = self.extract_global_info(annotation)#################### yes

#         # process local (protein sequence)

#         global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0  ###################### 选self.seq_self_attn(tokens)

#         conv_input = rearrange(tokens, 'b n d -> b d n')

#         if exists(mask): ############################################ yes
#             conv_input_mask = rearrange(mask, 'b n -> b () n')
#             conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

#         narrow_out = self.narrow_conv(conv_input) ##### narrow_out and conv_input 形状都是torch.Size([32, 512, 512])
#         narrow_out = rearrange(narrow_out, 'b d n -> b n d')
#         wide_out = self.wide_conv(conv_input) ########### wide_out and conv_input形状都是torch.Size([32, 512, 512])
#         wide_out = rearrange(wide_out, 'b d n -> b n d')

#         tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn ########### 形状都是torch.Size([32, 512, 512])
#         tokens = self.local_norm(tokens)

#         tokens = self.local_feedforward(tokens)########### 形状都是torch.Size([32, 512, 512])

#         # process global (annotations)

#         annotation = self.global_attend_local(annotation, tokens, context_mask = mask)#### mask形状[32,512],其他是torch.Size([32, 512, 512])
#         annotation = self.global_dense(annotation)
#         annotation = self.global_norm(annotation)
#         annotation = self.global_feedforward(annotation)##### 这片annotation都是torch.Size([32, 512, 512])
#         # print('layerout_annotation:',annotation)

#         return tokens, annotation

class Layer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_global,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        glu_conv = False
    ):
        super().__init__()

        self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None #seq attention 设置 ***c2设置***

        conv_mult = 2 if glu_conv else 1

        ####### narrow 设置########
        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        ####### wide 设置########
        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        ##################################### annotaton 的wide 和narrow
        ####### narrow 设置########
        self.global_narrow_conv = nn.Sequential(
            nn.Conv1d(dim_global, dim_global * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim_global = 1)
        )

        global_wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        ####### wide 设置########
        self.global_wide_conv = nn.Sequential(
            nn.Conv1d(dim_global, dim_global * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = global_wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim_global = 1)
        )
        ##############################################################

        self.local_to_global_attn = local_to_global_attn

        if local_to_global_attn:   ##################### no
            self.extract_global_info = CrossAttention(
                dim = dim,
                dim_keys = dim_global,
                dim_out = dim,
                heads = attn_heads,
                dim_head = attn_dim_head
            )
        else:
            self.extract_global_info = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_global, dim),
                nn.GELU(),
                Rearrange('b d -> b () d')
            )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )),
            nn.LayerNorm(dim)
        )

        self.global_attend_local = CrossAttention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head, qk_activation = attn_qk_activation)

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU()
        )

        self.global_norm = nn.LayerNorm(dim_global)

        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim_global, dim_global),
                nn.GELU()
            )),
            nn.LayerNorm(dim_global),
        )

    def forward(self, tokens, annotation, mask = None):
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation, mask = mask)############## no 
        else:
            global_info = self.extract_global_info(annotation)#################### yes # ******* c1 ********

        # process local (protein sequence)

        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0 ###################### 选self.seq_self_attn(tokens) # ************ seq多了个attention处理  ************

        conv_input = rearrange(tokens, 'b n d -> b d n')

        if exists(mask):############################################ yes
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input) ##### narrow_out and conv_input 形状都是torch.Size([32, 512, 512]) # ******* a2.1********
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input) ########### wide_out and conv_input形状都是torch.Size([32, 512, 512]) # ******* a2.2********
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn ########### 形状都是torch.Size([32, 512, 512])
        tokens = self.local_norm(tokens) # ******* a3.2******** no add

        tokens = self.local_feedforward(tokens) ########### 形状都是torch.Size([32, 512, 512]) # ******* a4 and a5.2******** no add no 上一个norm的输出

        # process global (annotations)
        annotation_input = annotation

        # annotation = self.global_attend_local(annotation, tokens, context_mask = mask)#### mask形状[32,512],其他是torch.Size([32, 512, 512])  # ******* c2 ********
        annotation = self.global_attend_local(tokens, annotation, context_mask = mask)#### mask形状[32,512],其他是torch.Size([32, 512, 512])  # ******* c2 ********  annotation作为value
        # annotation = self.global_dense(annotation) # ******* b2 ********
        ############ change b2
        global_conv_input = rearrange(annotation, 'b n d -> b d n')

        if exists(mask):############################################ yes
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            global_conv_input = global_conv_input.masked_fill(~conv_input_mask, 0.)

        global_narrow_out = self.global_narrow_conv(global_conv_input)  # ******* b2.1********
        global_narrow_out = rearrange(global_narrow_out, 'b d n -> b n d')
        global_wide_out = self.global_narrow_conv(global_conv_input)      # ******* b2.2********
        global_wide_out = rearrange(global_wide_out, 'b d n -> b n d')
        # b2_out = global_narrow_out + global_wide_out + annotation_input
        b2_out = global_narrow_out + global_wide_out
        annotation = self.global_norm(b2_out) # ******* b3.2 ******** no add b3输入也要改
        #####################################
        # annotation = self.global_norm(annotation) # ******* b3.2 ******** no add
        annotation = self.global_feedforward(annotation)##### 这片annotation都是torch.Size([32, 512, 512]) # ******* b4 and b5.2******** no add
        # print('layerout_annotation:',annotation)

        return tokens, annotation
# main model

class ProteinBERT(nn.Module):
    def __init__(
        self,
        *,
        # num_tokens = 26,
        num_tokens = 24,#0,1,2 + 3~23
        num_annotation = 8943,
        num_annotation_class = 65,#0,1,2 + 3~64
        dim = 512,
        dim_global = 256,
        depth = 6,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        num_global_tokens = 1,
        glu_conv = False
    ):
        super().__init__()
        self.num_annotation = num_annotation
        self.num_tokens = num_tokens #
        self.token_emb = nn.Embedding(num_tokens, dim)#token get embedding 26->512   #############a1
        

        # self.num_global_tokens = num_global_tokens
        # self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)#annotation get embedding 8943->256
        ################################
        # self.num_annotation = num_annotation
        # self.to_global_emb = nn.Embedding(num_annotation, dim_global)#8943->256
        self.num_annotation_class = num_annotation_class
        self.to_global_emb = nn.Embedding(num_annotation_class, dim_global)#63->256    #############b1
        
       

        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel, attn_qk_activation = attn_qk_activation, local_to_global_attn = local_to_global_attn, local_self_attn = local_self_attn, glu_conv = glu_conv) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens) #token 512->26

        # self.to_annotation_logits = nn.Sequential(
        #     # Reduce('b n d -> b d', 'mean'),
        #     nn.Linear(dim_global, num_annotation)#annotation 256->8943
        # )
        ########################################
        # self.to_annotation_logits = nn.Linear(dim_global, num_annotation)#256->63
        self.to_annotation_logits = nn.Linear(dim_global, num_annotation_class)#256->63

    def forward(self, seq, annotation, mask = None):
        # print('seq_shape:',seq.shape,'annotation_shape:',annotation.shape)#seq_shape: torch.Size([32, 122]) annotation_shape: torch.Size([32, 122])
        tokens = self.token_emb(seq)

        annotation = self.to_global_emb(annotation)
        # annotation = rearrange(annotation, 'b (n d) -> b n d', n = self.num_global_tokens)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)

        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        # print('proteinBERT_out_annotation',annotation)
        # print('tokens_shape:',tokens.shape,'annotation_shape:',annotation.shape)#tokens_shape: torch.Size([32, 122, 24]) annotation_shape: torch.Size([32, 122, 65])
        return tokens, annotation

# pretraining wrapper

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len) #ceil() 向上取整

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil()) #cumsum(dim=-1) 除了第一列，后面每一列都加上它前面的所有列
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class PretrainingWrapper(nn.Module):
    def __init__(
        self,
        model,
        # seq_length = 122,
        seq_length = 512,
        random_replace_token_prob = 0.05,
        remove_annotation_prob = 0.25,
        add_annotation_prob = 0.01,
        remove_all_annotations_prob = 0.5,
        seq_loss_weight = 1.,
        annotation_loss_weight = 1.,
        # exclude_token_ids = (0, 1, 2)   # for excluding padding, start, and end tokens from being masked
        # exclude_token_ids = (25, 23, 24),
        # RNA_exclude_token_ids = (67, 65, 66) #############
        exclude_token_ids = (0, 1, 2),
        RNA_exclude_token_ids = (0, 1, 2) #############
    ):
        super().__init__()
        assert isinstance(model, ProteinBERT), 'model must be an instance of ProteinBERT'

        self.model = model

        self.seq_length = seq_length ###################################

        self.random_replace_token_prob = random_replace_token_prob
        self.remove_annotation_prob = remove_annotation_prob
        self.add_annotation_prob = add_annotation_prob
        self.remove_all_annotations_prob = remove_all_annotations_prob

        self.seq_loss_weight = seq_loss_weight
        self.annotation_loss_weight = annotation_loss_weight

        #######################
        self.exclude_token_ids = exclude_token_ids
        self.RNA_exclude_token_ids = RNA_exclude_token_ids

    # def forward(self, seq, annotation, mask_05, mask = None):
    def forward(self, seq, annotation, epoch, mask = None):
        # self.model.train()
        # print('-----------------train---------------------')
        # for k,v in self.model.named_parameters():
        #     print('{}: {}'.format(k, v.requires_grad))
        # print('------------------------','seq_shape',seq.shape, 'annotation_shape',annotation.shape,'-------------------')
        batch_size, device = seq.shape[0], seq.device

        seq_labels = seq
        annotation_labels = annotation

        if not exists(mask):
            mask = torch.ones_like(seq).bool()#[batch_size,seq_length]

        # prepare masks for noising sequence

        excluded_tokens_mask = mask
        

        for token_id in self.exclude_token_ids:
            AA_excluded_tokens_mask = excluded_tokens_mask & (seq != token_id) #pad start end 的excluded_tokens_mask位置为0，其他为1
            # print('excluded_tokens_mask:',excluded_tokens_mask)
        for token_id in self.RNA_exclude_token_ids:
            RNA_excluded_tokens_mask = excluded_tokens_mask & (annotation != token_id)
            # RNA_excluded_tokens_mask = mask_05 & (annotation != token_id)

        # print('excluded_tokens_mask',excluded_tokens_mask,'excluded_tokens_mask_shape',excluded_tokens_mask.shape,'counts(False))',(excluded_tokens_mask==False).sum())
        # print('self.random_replace_token_prob',self.random_replace_token_prob,'counts_pre)',self.random_replace_token_prob*seq.shape[0]*seq.shape[1])
        
        random_replace_token_prob_mask = get_mask_subset_with_prob(AA_excluded_tokens_mask, self.random_replace_token_prob)#5%错误
        
        #############################change 我也随机替换5%

        # generate random tokens
        random_tokens=torch.zeros_like(seq) #seq 随机替换token成0
        # random_tokens = torch.randint(0, self.model.num_tokens, seq.shape, device=seq.device)

        # for token_id in self.exclude_token_ids:
        #     random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence
        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)##torch.where 合并a,b两个tensor，如果a中元素大于0，取b的值，否则取c的值,每一个位置依次比较,第一个位置，要变的是True

        ####################################################################### annotation 随机替换token
        # RNA_random_replace_token_prob_mask = get_mask_subset_with_prob(RNA_excluded_tokens_mask, self.random_replace_token_prob)#我也随机替换5%错误
        # RNA_random_tokens = torch.randint(0, self.model.num_annotation_class, annotation.shape, device=annotation.device)
        # for token_id in self.RNA_exclude_token_ids:
        #     RNA_random_replace_token_prob_mask = RNA_random_replace_token_prob_mask & (RNA_random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # # noise sequence
        # noised_annotation = torch.where(RNA_random_replace_token_prob_mask, RNA_random_tokens, annotation)
        #######################################################################

        ####################################################################### annotation 随机替换token成0
        RNA_random_replace_token_prob_mask = get_mask_subset_with_prob(RNA_excluded_tokens_mask, self.random_replace_token_prob)#我也随机替换5%
        # RNA_random_tokens = torch.randint(0, self.model.num_annotation_class, annotation.shape, device=annotation.device)
        
        RNA_random_tokens=torch.zeros_like(annotation)
        # for token_id in self.RNA_exclude_token_ids:
        #     RNA_random_replace_token_prob_mask = RNA_random_replace_token_prob_mask & (RNA_random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence
        noised_annotation = torch.where(RNA_random_replace_token_prob_mask, RNA_random_tokens, annotation)
        #######################################################################

        ############################################################################ 做随机的mask50%整条序列 end
        # index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.5))
        # noised_annotation[index_list,:]=0
        ################################################################### 所有epoch做10%的mask
        # index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.1))
        # noised_annotation[index_list,:]=0

        ####################################渐进mask
        # if epoch<10:
        #     pass
        # elif epoch<20:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.01))
        #     noised_annotation[index_list,:]=0
        # elif epoch<40:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.1))
        #     noised_annotation[index_list,:]=0
        # elif epoch<60:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.3))
        #     noised_annotation[index_list,:]=0
        # elif epoch<100:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.5))
        #     noised_annotation[index_list,:]=0
        # elif epoch<120:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.75))
        #     noised_annotation[index_list,:]=0
        # elif epoch<150:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.9))
        #     noised_annotation[index_list,:]=0
        # elif epoch>150:
        #     index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*1))
        #     noised_annotation[index_list,:]=0

        
        # index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*1))
        # noised_annotation[:,:]=0 #全部mask掉
        ############################################################################

        ############################################################################ 15 gap 320 epoch
        if epoch<15:
            pass
        elif epoch<30:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.05))
            noised_annotation[index_list,:]=0
        elif epoch<45:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.1))
            noised_annotation[index_list,:]=0
        elif epoch<60:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.15))
            noised_annotation[index_list,:]=0
        elif epoch<75:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.2))
            noised_annotation[index_list,:]=0
        elif epoch<90:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.25))
            noised_annotation[index_list,:]=0
        elif epoch<105:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.3))
            noised_annotation[index_list,:]=0
        elif epoch<120:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.35))
            noised_annotation[index_list,:]=0
        elif epoch<135:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.4))
            noised_annotation[index_list,:]=0
        elif epoch<150:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.45))
            noised_annotation[index_list,:]=0
        elif epoch<165:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.5))
            noised_annotation[index_list,:]=0
        elif epoch<180:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.55))
            noised_annotation[index_list,:]=0
        elif epoch<195:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.6))
            noised_annotation[index_list,:]=0
        elif epoch<210:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.65))
            noised_annotation[index_list,:]=0
        elif epoch<225:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.7))
            noised_annotation[index_list,:]=0
        elif epoch<240:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.75))
            noised_annotation[index_list,:]=0
        elif epoch<255:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.8))
            noised_annotation[index_list,:]=0
        elif epoch<270:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.85))
            noised_annotation[index_list,:]=0
        elif epoch<285:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.9))
            noised_annotation[index_list,:]=0
        elif epoch<300:
            index_list = random.sample(range(0,noised_annotation.shape[0]-1),int(noised_annotation.shape[0]*0.95))
            noised_annotation[index_list,:]=0
        elif epoch>300:
            noised_annotation[:,:]=0
        ############################################################################


        # ############################################################################ 不做随机的mask50%整条序列 end
        # mask_row = torch.zeros(1, annotation.shape[1], device=annotation.device)
        # mask_row = mask_row.long()
        # # mask_row = np.ones((1,mask_05.shape[1]), dtype=bool)
        # index_list = random.sample(range(0,annotation.shape[0]-1),int(annotation.shape[0]*0.5))
        # noised_annotation = annotation
        # for index in index_list:
        #     noised_annotation[[index],:]=mask_row
        # ############################################################################

        # noise annotation
        # print('add_annotation_prob_mask.type',add_annotation_prob_mask.type,'annotation.dtype',annotation.dtype)
        #raw mask
        # noised_annotation = annotation + add_annotation_prob_mask.type(annotation.dtype) #随机添加注释的概率，默认为1%
        # noised_annotation = noised_annotation * remove_annotation_mask.type(annotation.dtype) #25%和50%的删除
        #change mask
        # noised_annotation = torch.where(random_replace_token_prob_mask, random_tokens, annotation)
        #annotation no mask
        # noised_annotation = annotation
        ######################################## change
        # RNA_random_tokens = torch.randint(0, self.model.num_annotation_class, annotation.shape, device=annotation.device)
        # for token_id in self.RNA_exclude_token_ids:
        #     RNA_random_replace_token_prob_mask = RNA_random_replace_token_prob_mask & (RNA_random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # # noise sequence
        # noised_annotation = torch.where(RNA_random_replace_token_prob_mask, RNA_random_tokens, annotation)
        ####################################################################

        ######################################### 不做随机的mask了
        # RNA_random_tokens = torch.randint(0, self.model.num_annotation_class, annotation.shape, device=annotation.device)#生成随机token
        # # all_zeros = torch.zeros(annotation.shape[0], annotation.shape[1], device=annotation.device)
        
        # for token_id in self.RNA_exclude_token_ids:
        #     RNA_excluded_tokens_mask = RNA_excluded_tokens_mask & (RNA_random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # # noise sequence
        # noised_annotation = torch.where(RNA_excluded_tokens_mask, RNA_random_tokens, annotation)##############随机替换的mask，随机替换的tokens，需要进行替换的数据，50%序列mask
        
        

        # print('------------------------','seq_shape',noised_seq.shape, 'annotation_shape',noised_annotation.shape,'-------------------')[64,120][64,120]

        # denoise with model


        seq_logits, annotation_logits = self.model(noised_seq, noised_annotation, mask = mask)#############调用模型
        # print('------------------------','seq_shape',seq_logits.shape, 'annotation_shape',annotation_logits.shape,'-------------------')#seq_shape torch.Size([64, 120, 21]) annotation_shape torch.Size([64, 120, 63])
        
        # print('123')

        # calculate loss
        # print('mask:',mask,'mask_shape:',mask.shape)

        seq_logits = seq_logits[mask] #mask控制了哪些元素被保留，哪些元素被丢弃,mask为true的位置，相应的seq_logits元素保留，mask的shape为seq_logits[0],seq_logits[1],结果后的seq_logits形状为[(mask[0]*mask[1]),seq_logits[2]]
        seq_labels = seq_labels[mask]
        # print('------------------------','seq_shape',seq_logits.shape, 'annotation_shape',annotation_logits.shape,'-------------------')#seq_shape torch.Size([7680, 21]) annotation_shape torch.Size([64, 120, 63])
        ##################################################################
        annotation_logits = annotation_logits[mask]
        annotation_labels = annotation_labels[mask]
        # print('------------------------','seq_shape',seq_logits.shape, 'annotation_shape',annotation_logits.shape,'-------------------')#seq_shape torch.Size([7680, 21]) annotation_shape torch.Size([7680, 63])
        ##################################################################
        # print('seq_logits:',seq_logits)
        # print('seq_logits_shape:',seq_logits.shape)
        # print('seq_labels:',seq_labels)
        # print('seq_labels_shape:',seq_labels.shape)
        ####################################################################GC含量计算
        # annotation_logits还原成DNA序列
        annotation_logits_array = np.array(annotation_logits.cpu().detach().numpy())#######执行之后
        result_annotation = np.argmax(annotation_logits_array, axis=1)#######执行之后
        DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation)
        GC_con_single = GC_con(DNA_annotation_result)
        # DNA_annotation_result_all = []
        # AA_annotation_result_all = []
        # for annotation_index in range(len(result_annotation)):
        #     DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation[annotation_index])
        #     DNA_annotation_result_all.append(DNA_annotation_result)
        #     AA_annotation_result_all.append(AA_annotation_result)
        
        
        # 求所有序列的GC_con
        # GC_con_all = 0
        # for DNA_seq_index in range(len(DNA_annotation_result_all)):
        #     GC_con_single = GC_con(DNA_annotation_result_all[DNA_seq_index])
        #     GC_con_all = GC_con_all + GC_con_single
        # GC_con_all = GC_con_all/len(DNA_annotation_result_all)
        
        
        

        ####################################################################

        # seq_loss = FC.cross_entropy(seq_logits, seq_labels, reduction = 'sum')
        seq_loss = FC.cross_entropy(seq_logits, seq_labels, reduction = 'mean')
        # annotation_loss = FC.binary_cross_entropy_with_logits(annotation_logits, annotation_labels, reduction = 'sum')
        # print('annotation_logits:',annotation_logits)
        # print('annotation_logits_shape:',annotation_logits.shape)
        # print('annotation_labels:',annotation_labels)
        # print('annotation_labels_shape:',annotation_labels.shape)

        # annotation_loss = FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'sum')
        # annotation_loss = FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'mean')
        propor = 0.8 # GC_con_single起作用的比例
        annotation_loss = FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'mean') - GC_con_single * propor * FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'mean') 
        
        # print('seq_loss',seq_loss, 'annotation_loss',annotation_loss)
        #####################################################################
        # seq_logits = rearrange(seq_logits, '(b n) d -> b n d', n = self.seq_length)
        # annotation_logits = rearrange(annotation_logits, '(b n) d -> b n d', n = self.seq_length)
        

        # return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight, seq_loss, annotation_loss, seq_logits, annotation_logits
        return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight, seq_loss, annotation_loss, seq_logits, annotation_logits, seq_labels, annotation_labels
