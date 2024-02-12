import math
import torch
import random
import torch.nn.functional as FC
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
from config_function import *


def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

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
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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

        self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None 

        conv_mult = 2 if glu_conv else 1

        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        self.global_narrow_conv = nn.Sequential(
            nn.Conv1d(dim_global, dim_global * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim_global = 1)
        )

        global_wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.global_wide_conv = nn.Sequential(
            nn.Conv1d(dim_global, dim_global * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = global_wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim_global = 1)
        )

        self.local_to_global_attn = local_to_global_attn

        if local_to_global_attn:   
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
            global_info = self.extract_global_info(tokens, annotation, mask = mask)
        else:
            global_info = self.extract_global_info(annotation)

        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0 

        conv_input = rearrange(tokens, 'b n d -> b d n')

        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input) 
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input) 
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn 
        tokens = self.local_norm(tokens) 

        tokens = self.local_feedforward(tokens) 

        
        annotation = self.global_attend_local(tokens, annotation, context_mask = mask)
        global_conv_input = rearrange(annotation, 'b n d -> b d n')

        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            global_conv_input = global_conv_input.masked_fill(~conv_input_mask, 0.)

        global_narrow_out = self.global_narrow_conv(global_conv_input)  
        global_narrow_out = rearrange(global_narrow_out, 'b d n -> b n d')
        global_wide_out = self.global_narrow_conv(global_conv_input)      
        global_wide_out = rearrange(global_wide_out, 'b d n -> b n d')
        b2_out = global_narrow_out + global_wide_out
        annotation = self.global_norm(b2_out) 
        annotation = self.global_feedforward(annotation)

        return tokens, annotation


class CodonBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = 24,
        num_annotation = 8943,
        num_annotation_class = 65,
        dim = 128,
        dim_global = 128,
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
        self.num_tokens = num_tokens 
        self.token_emb = nn.Embedding(num_tokens, dim)
        
        self.num_annotation_class = num_annotation_class
        self.to_global_emb = nn.Embedding(num_annotation_class, dim_global)

        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel, attn_qk_activation = attn_qk_activation, local_to_global_attn = local_to_global_attn, local_self_attn = local_self_attn, glu_conv = glu_conv) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens) 
        self.to_annotation_logits = nn.Linear(dim_global, num_annotation_class)

    def forward(self, seq, annotation, mask = None):
        tokens = self.token_emb(seq)
        annotation = self.to_global_emb(annotation)
        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)
        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        return tokens, annotation

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len) 

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil()) 
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
        seq_length = 512,
        random_replace_token_prob = 0.1,
        remove_annotation_prob = 0.25,
        add_annotation_prob = 0.01,
        remove_all_annotations_prob = 0.5,
        seq_loss_weight = 1.,
        annotation_loss_weight = 1.,
        exclude_token_ids = (0, 1, 2),
        RNA_exclude_token_ids = (0, 1, 2) 
    ):
        super().__init__()
        assert isinstance(model, CodonBERT), 'model must be an instance of CodonBERT'

        self.model = model

        self.seq_length = seq_length 

        self.random_replace_token_prob = random_replace_token_prob
        self.remove_annotation_prob = remove_annotation_prob
        self.add_annotation_prob = add_annotation_prob
        self.remove_all_annotations_prob = remove_all_annotations_prob

        self.seq_loss_weight = seq_loss_weight
        self.annotation_loss_weight = annotation_loss_weight

        self.exclude_token_ids = exclude_token_ids
        self.RNA_exclude_token_ids = RNA_exclude_token_ids

    def forward(self, seq, annotation, epoch, mask = None):
        batch_size, device = seq.shape[0], seq.device

        seq_labels = seq
        annotation_labels = annotation

        if not exists(mask):
            mask = torch.ones_like(seq).bool()

        excluded_tokens_mask = mask
        

        for token_id in self.exclude_token_ids:
            AA_excluded_tokens_mask = excluded_tokens_mask & (seq != token_id) 
        for token_id in self.RNA_exclude_token_ids:
            RNA_excluded_tokens_mask = excluded_tokens_mask & (annotation != token_id)
        
        random_replace_token_prob_mask = get_mask_subset_with_prob(AA_excluded_tokens_mask, self.random_replace_token_prob)
        random_tokens=torch.zeros_like(seq)
        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)
        RNA_random_replace_token_prob_mask = get_mask_subset_with_prob(RNA_excluded_tokens_mask, self.random_replace_token_prob)
        RNA_random_tokens=torch.zeros_like(annotation)
        noised_annotation = torch.where(RNA_random_replace_token_prob_mask, RNA_random_tokens, annotation)

        if epoch < 15:
            pass
        elif epoch < 300:
            mask_ratio = min((epoch // 15) * 0.05, 0.95)
            index_list = random.sample(range(0, noised_annotation.shape[0] - 1), int(noised_annotation.shape[0] * mask_ratio))
            noised_annotation[index_list, :] = 0
        else:
            noised_annotation[:, :] = 0

        seq_logits, annotation_logits = self.model(noised_seq, noised_annotation, mask = mask)
        seq_logits = seq_logits[mask] 
        seq_labels = seq_labels[mask]
        annotation_logits = annotation_logits[mask]
        annotation_labels = annotation_labels[mask]

        seq_loss = FC.cross_entropy(seq_logits, seq_labels, reduction = 'mean')
        annotation_loss = FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'mean')
   
        return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight, seq_loss, annotation_loss
