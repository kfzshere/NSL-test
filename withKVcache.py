import numpy as np
import torch
import time
import math

torch.set_printoptions(8)
# 设置tensor在打印时显示的小数位数为 8 位
global past_key_value
global n_seq
# 激活函数gelu，决定神经网络是否传递信息，关乎后面masked环节
def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    inner1 = x + 0.044715 * (x ** 3)
    inner2 = (2 / math.pi) ** 0.5
    # 计算 tanh 的输入
    tanh_input = inner1 * inner2
    # 计算最终结果：0.5x * (1 + tanh(...))
    return 0.5 * x * (1 + torch.tanh(tanh_input))


# 样本属于对应类别的概率
def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
        #将输入化为指数是因为保证正数，不论输入啥，都会被映射成和为一的正值
    """
    # dim-1表示在最后一个维度上进行操作,keepdim = True保持输入输出维度相同不被压缩
    exp_x = torch.exp(x - torch.max(x, dim = -1, keepdim = True)[0])
    sum_exp_x = torch.sum(exp_x, dim = -1, keepdim = True)
    return exp_x / sum_exp_x


# 归一化
# 加速模型收敛+稳定性
def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
        本质就是标准化
        比batch norm 好在只针对单个样本的不同特征做操作（也因此可以不受样本数的限制）毕竟要均值方差，mini样本也保险
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    # 这个缩放参数和偏移参数是可学习的
    
    #均值+方差
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1, keepdim = True, unbiased = False)
    #unbiased = False 有偏估计
    #归一化计算
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return g * x_norm + b   #再乘以缩放参数和偏移参数


# 就是最基本的线性变换，方便执行下一步操作的
def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
        y=xW+b
    """
    # w权重矩阵， b偏置向量
    w, b = w_b['w'], w_b['b']    
    # 执行矩阵乘法
    out = x @ w + b 
    return out
    
    
# fully-connected feed-forward network
# 升维激活降维
def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
        FFN层是一个顺序结构：包括一个全连接层(FC) + gelu激活层 + 第二个全连接层
    """
    #w_b1第一个线性层的权重和偏置,w_b2第二个线性层的权重和偏置
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    #扩展维度
    hidden = linear(x, w_b1)
    #激活
    hidden = gelu(hidden)
    #降维
    out = linear(hidden, w_b2)
    return out


# attention is all you need
# 根据相关性求权重
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer ps(这跟ffn类似
        Output: Tensor
    """
    # 计算注意力分数,他是key转置矩阵和query矩阵乘积，后面需要乘权重
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))# [n_q, n_k]
    # 是否masked，要把后面的盖掉
    if mask is not None:
        # mask保证了自回归性
        # 值为 0 或 -inf，0是不管，-inf与原本相加了就是0，确保模型在生成第i个位置的token时，只能看到位置0到i-1的信息，不能看到位置i及之后的信息
        scores = scores + mask  
    # 计算注意力权重
    attn_weights = softmax(scores)  # [n_q, n_k]
    # 计算加权和
    out = attn_weights @ v  # [n_q, n_k] @ [n_k, d_v] -> [n_q, d_v]
    return out
    

# multi-head attention
# 多头和上面attention的区别在，一个输入的ai，他有多组kqv，你这时候一个a生成的b，也是多组的
# 这样可以让模型在不同的子空间中学习不同的表示
def mha(x, attn, n_head, layer_idx):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """ 
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    global past_key_value
    global n_seq
    # c_attn: 将输入转换为 Q、K、V 三个矩阵的权重和偏置
    # 是一个大的线性变换层，将输入 x 映射到 3 倍维度（因为要生成 Q、K、V 三个矩阵）
    # c_proj: 将多头注意力的输出映射回原始维度的线性变换层
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [n_seq] -> [1, n_seq]
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    #这里拉伸是按照qkv的顺序来的
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    q = x[..., :x.size(-1) // 3]  # 直接索引前 1/3
    k = x[..., x.size(-1) // 3 : 2 * x.size(-1) // 3]  # 索引中间 1/3
    v = x[..., 2 * x.size(-1) // 3 :]  # 索引最后 1/3
    
    # Split into heads
    # chunk将tensor在指定维度上平均分成n_head份
    q_heads = q.chunk(n_head, dim=-1) #q11,q12,q13.....q21,q22,q23
    k_heads = k.chunk(n_head, dim=-1)
    v_heads = v.chunk(n_head, dim=-1)
    qkv_heads = list(zip(q_heads, k_heads, v_heads))
    # n_head * [n_seq, n_embd/n_head]

    k_heads_tensor = torch.stack(k_heads, dim=0).unsqueeze(0)  # [1, n_head, n_seq, n_embd/n_head]
    v_heads_tensor = torch.stack(v_heads, dim=0).unsqueeze(0)  # [1, n_head, n_seq, n_embd/n_head] 
    

    # first time generate
    if past_key_value[layer_idx] == None:
        # 第一次赋值
        past_key_value[layer_idx] = {
            # batch_size * n_head * [n_seq, n_embd/n_head]
            "key" : k_heads_tensor,
                #torch.randn(1, n_head, x.size(0), q.size(-1) // n_head).to(x.device),
            "value" : v_heads_tensor
                #torch.randn(1, n_head, x.size(0), q.size(-1) // n_head).to(x.device)
            #这里batch_size默认为1奥，因为只有一个句子
        }
        # 将 k_heads 和 v_heads 转换为 Tensor，并添加 batch_size 维度 
    else:
        past_key_value[layer_idx]["key"] = torch.cat((past_key_value[layer_idx]["key"], k_heads_tensor), dim=2)  # [1, n_head, n_seq + new_seq, d_k]
        past_key_value[layer_idx]["value"] = torch.cat((past_key_value[layer_idx]["value"], v_heads_tensor), dim=2)  # [1, n_head, n_seq + new_seq, d_v]
        
    
    k_heads = torch.unbind(past_key_value[layer_idx]['key'].squeeze(0), dim=0)
    v_heads = torch.unbind(past_key_value[layer_idx]['value'].squeeze(0), dim=0)
    qkv_heads = list(zip(q_heads, k_heads, v_heads))

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    # 这里mask跟上面attention的意思差不多
    # 使用 torch.full 构造一个 [n_seq, n_seq] 的全 -inf 矩阵；
    # 再用 torch.triu(..., diagonal=1) 保留严格上三角部分，其余（对角线及以下）自动为 0（未显式设置的 tensor 默认值为 0）；
    # .to(x.device)是确保掩码矩阵和输入在同一设备上（CPU或GPU）
    seq_len = n_seq  # n_seq
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(x.device)
    # 截取mask，因为后面每次只有单个token传入的时候，只有一行，所以动态调整为传入x的大小
    causal_mask = causal_mask[-q.size(0):, :]

    

    # Perform attention over each head 每个头单独计算
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    #合并多头输出，沿一维拼接
    x = torch.cat(out_heads, dim=-1)  # [n_seq, n_embd]
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head,layer_idx):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, layer_idx=layer_idx)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    global past_key_value
    global n_seq
    n_seq = len(inputs)
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    if past_key_value is None:
        past_key_value = [None] * len(blocks)
        # 初始化放在执行的时候
        # token + positional embeddings
        x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    else:
        x = wte[inputs[-1]] + wpe[len(inputs) - 1]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, layer_idx=layer_idx)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    global past_key_value
    past_key_value = None
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, params, n_head=n_head)  # 启用KV Cache
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate :]

def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)