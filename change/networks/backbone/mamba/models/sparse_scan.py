import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def masked_cross_scan(x, mask):
    B, C, H, W = x.shape
    xs = x.new_empty((B, 4, C, H * W))         # 预分配输出：4条扫描序列
    masks = mask.new_empty((B, 4, 1, H * W))         # 预分配输出：4条扫描序列
    # mask out input 
    xs[:, 0] = x.flatten(2, 3)                  # (B,C,H,W) -> (B,C,L) 行优先展平
    xs[:, 1] = x.transpose(2, 3).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描

    masks[:, 0] = mask.flatten(2, 3)                  # (B,C,H,W) -> (B,C,L) 行优先展平
    masks[:, 1] = mask.transpose(2, 3).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    masks[:, 2:4] = torch.flip(masks[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描
    # 挑选mask=1的位置
    seq_list = []
    idx_list = []
    for i in range(B):
        # 为每个批次创建一个单独的掩码
        batch_mask = masks[i].expand(4, C, H*W)
        # 使用布尔索引来提取该批次的所有元素
        selected_elements = xs[i].reshape(4*C, H*W)[batch_mask.reshape(4*C, H*W)]
        if selected_elements is None or selected_elements.numel() == 0:
            # 如果张量为空，保留所有位置特征
            selected_elements = xs[i].unsqueeze(0)
        else:
            # 如果张量非空，执行重采样操作
            selected_elements = selected_elements.reshape(4, C, -1)
            selected_elements = selected_elements.unsqueeze(0) # (1, 4, C, L)
            
            selected_elements = F.interpolate(
                selected_elements, 
                size=(C, H*W),
                mode='bilinear',
                align_corners=False
            )
        seq_list.append(selected_elements)
        idx_list.append(batch_mask)
    xs = torch.cat(seq_list, 0)
    return xs, idx_list

def masked_cross_merge(ys, idx_list):
    B, K, L = ys.shape
    H, W = int(L**0.5), int(L**0.5)
    C = K // 4
    ys = ys.view(B, 4, C, L)
    out_list = []
    for i in range(B):
        y = ys[i].unsqueeze(0)  # 1, 4, c, hw
        idx = idx_list[i]  # h*w
        n_valid =idx[0,0].sum()
        if n_valid > 0:
            y = F.interpolate(
                    y, 
                    size=(C, n_valid),
                    mode='bilinear',
                    align_corners=False
                )[0] # 4, c, l
            # 先建空输出
            out = torch.zeros((1, 4, C, H*W), device=y.device, dtype=y.dtype)

            # mask 扩展到 (1,4,c,h*w)，让每个通道共享同一个空间 mask
            idx = idx.unsqueeze(0)

            # scatter: 按 mask 填充
            out[idx] = y.view(-1)
            out_list.append(out)
        else:
            out_list.append(y)
    out = torch.cat(out_list, 0)
    out = out[:, 0:2] + out[:, 2:4].flip(dims=[-1]).view(B, 2, C, -1)
    y = out[:, 0] + out[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
    return y
    
def masked_cross_scan_v2(x, mask, debug=False):
    B, C, H, W = x.shape
    xs = x.new_empty((B, 4, C, H * W))         # 预分配输出：4条扫描序列
    masks = mask.new_empty((B, 4, 1, H * W))         # 预分配输出：4条扫描序列
    # mask out input 
    xs[:, 0] = x.flatten(2, 3)                  # (B,C,H,W) -> (B,C,L) 行优先展平
    xs[:, 1] = x.transpose(2, 3).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描

    masks[:, 0] = mask.flatten(2, 3)                  # (B,C,H,W) -> (B,C,L) 行优先展平
    masks[:, 1] = mask.transpose(2, 3).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    masks[:, 2:4] = torch.flip(masks[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描
    # 获取所有batch中最大有效位置数量
    max_seq_len = mask.squeeze(1).flatten(1).sum(1).max()
    # 挑选mask=1的位置
    seq_list = []
    idx_list = []
    if debug:
        import ipdb;ipdb.set_trace()
    for i in range(B):
        # 为每个批次创建一个单独的掩码
        batch_mask = masks[i].expand(4, C, H*W)
        # 使用布尔索引来提取该批次的所有元素
        selected_elements = xs[i].reshape(4*C, H*W)[batch_mask.reshape(4*C, H*W)]
        if selected_elements is None or selected_elements.numel() == 0:
            selected_elements = xs[i].unsqueeze(0)
            if max_seq_len > 0:
                # 如果张量为空，保留所有位置特征
                selected_elements = F.interpolate(
                        selected_elements, 
                        size=(C, max_seq_len),
                        mode='bilinear',
                        align_corners=False
                    )
        else:
            # 如果张量非空，执行重采样操作
            selected_elements = selected_elements.reshape(4, C, -1)
            selected_elements = selected_elements.unsqueeze(0) # (1, 4, C, L)
            selected_elements = F.pad(selected_elements, (0, max_seq_len-selected_elements.shape[-1]), 'constant', 0)
        seq_list.append(selected_elements)
        idx_list.append(batch_mask)
    xs = torch.cat(seq_list, 0)
    return xs, idx_list

def masked_cross_merge_v2(ys, idx_list, H, W):
    B, K, L = ys.shape
    C = K // 4
    ys = ys.view(B, 4, C, L)
    # 获取所有batch中有效位置数量
    out_list = []
    for i in range(B):
        y = ys[i].unsqueeze(0)  # 1, 4, c, hw
        idx = idx_list[i]  # 4, 64, h*w
        n_valid = idx[0,0].sum()
        if n_valid > 0:
            
            out = torch.zeros((1, 4, C, H*W), device=y.device, dtype=y.dtype)

            idx = idx.unsqueeze(0)
            # scatter: 按 mask 填充
            out[idx] = y[:,:,:,0:n_valid].contiguous().view(-1)
            out_list.append(out)
        else:
            y = F.interpolate(
                    y, 
                    size=(C,  H*W),
                    mode='bilinear',
                    align_corners=False
                )
            out_list.append(y)
    
    out = torch.cat(out_list, 0)
    out = out[:, 0:2] + out[:, 2:4].flip(dims=[-1])
    y = out[:, 0] + out[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)
    return y
        
def masked_cross_scan_v3(x, mask, debug=False):
    B, C, H, W = x.shape
    b, c, h, w = mask.shape
    # 先窗口化x的排列
    window_size = H // h  if H > h else 1# 窗口大小
    max_seq_len = int(mask.squeeze(1).flatten(1).sum(1).max())
    mask = F.interpolate(mask, size=(H,W), mode="nearest")
    x = x.reshape((B, C, H//window_size, window_size, W//window_size, window_size)).permute(0, 1, 2, 4, 3, 5) # (B,C,h,w,window_size, window_size)
    mask = mask.reshape((B, 1, H//window_size, window_size, W//window_size, window_size)).permute(0, 1, 2, 4, 3, 5) # (B,C,h,w,window_size, window_size)

    xs = x.new_empty((B, 4, C, h*w, window_size**2))         # 预分配输出：4条扫描序列
    masks = mask.new_empty((B, 4, 1, h*w, window_size**2))         # 预分配输出：4条扫描序列
    # mask out input 
    xs[:, 0] = x.flatten(4, 5).flatten(2, 3)            # (B, C, H//ws, W//ws, ws, ws) -> (B, C, h*w, ws**2) 先窗口内展平,再全局展平
    xs[:, 1] = x.permute(0,1,3,2,5,4).flatten(4, 5).flatten(2, 3)    # 先交换H/W，再展平 => 列优先展平
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描

    masks[:, 0] = mask.flatten(4, 5).flatten(2, 3)                  # (B, C, H//ws, W//ws, ws, ws) -> (B, C, h*w, ws**2) 行优先展平
    masks[:, 1] = mask.permute(0,1,3,2,5,4).flatten(4, 5).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    masks[:, 2:4] = torch.flip(masks[:, 0:2], dims=[-1])  # 将前两条序列在最后一维(L)反转，得到反向扫描
    # 获取所有batch中最大有效位置数量
    # 挑选mask=1的位置
    seq_list = []
    idx_list = []
    if debug:
        import ipdb;ipdb.set_trace()
    for i in range(B):
        # 为每个批次创建一个单独的掩码
        batch_mask = masks[i].expand(4, C, h*w, window_size**2).bool()
        # 使用布尔索引来提取该批次的所有元素
        selected_elements = xs[i].reshape(4*C * h*w, window_size**2)[batch_mask.reshape(4*C * h*w, window_size**2)]
        if selected_elements is None or selected_elements.numel() == 0:
            selected_elements = xs[i].unsqueeze(0) # (1, 4, C,  h*w, window_size**2)
            if max_seq_len > 0:
                # 如果张量为空，保留所有位置特征
                selected_elements = selected_elements.reshape(1, 4, C, -1) # (1, 4, C,  h*w,*window_size**2)
                selected_elements = F.interpolate(
                        selected_elements, 
                        size=(C, max_seq_len*window_size**2),
                        mode='bilinear',
                        align_corners=False
                    ).reshape(1, 4, C, max_seq_len, window_size**2)
            seq_list.append(selected_elements)
        else:
            # 如果张量非空，执行稀疏化操作
            selected_elements = selected_elements.reshape(4, C, -1, window_size**2)
            selected_elements = F.pad(selected_elements, (0, 0, 0, max_seq_len-selected_elements.shape[-2], 0, 0), 'constant', 0)
            seq_list.append(selected_elements.unsqueeze(0))
        idx_list.append(batch_mask)
    xs = torch.cat(seq_list, 0).flatten(3)
    return xs, idx_list, window_size

def masked_cross_merge_v3(ys, idx_list, H, W, ws):
    B, K, L = ys.shape
    C = K // 4
    ys = ys.view(B, 4, C, -1, ws**2)
    # 获取所有batch中有效位置数量
    out_list = []
    for i in range(B):
        y = ys[i] # 4, c, l, ws**2
        idx = idx_list[i]  # 4, C, h*w, ws**2
        n_valid = idx[0,0].sum() // ws**2
        if n_valid > 0:
            out = torch.zeros((4, C, H*W//ws**2, ws**2), device=y.device, dtype=y.dtype)
            # scatter: 按 mask 填充
            out[idx] = y[:,:,0:n_valid,:].contiguous().view(-1)
            out_list.append(out.unsqueeze(0))
        else:
            # import ipdb;ipdb.set_trace()
            y = y.reshape(4, C, -1).unsqueeze(0) # 1, 4, C, h*w*ws**2
            y = F.interpolate(y, size=(C, H*W), mode='bilinear', align_corners=False).reshape(1, 4, C, H*W//ws**2, ws**2)
            out_list.append(y)
    
    out = torch.cat(out_list, 0) # B, 4, C, h*w, ws**2
    out = (out[:, 0:2] + out[:, 2:4].flip(dims=[-1])).reshape(B, 2, C, H//ws, W//ws, ws, ws).permute(0,1,2,3,5,4,6) # (B, 2, C, H/ws, ws, W/ws, ws)
    y = out[:, 0].reshape(B, C, H, W) + \
        out[:, 1].transpose(3, 5).transpose(2, 4).contiguous().reshape(B, C, H, W)
    return y.view(B, C, -1)

def masked_cross_scan_v4(x, mask, backw, debug=False):
    B, C, H, W = x.shape
    b, c, h, w = mask.shape
    # 先窗口化x的排列
    window_size_h = H // h  if H > h else 1# 窗口大小
    window_size_w = W // w  if W > w else 1# 窗口大小
    # 加上位置编码
        
    pos_embed = PositionEmbeddingSine(C//2, normalize=True) # use sine positional embedding
    pos_mask = torch.zeros((B, H, W), dtype=torch.uint8).bool().to(x.device)
    x = pos_embed(x, pos_mask) + x
    mask_up = F.interpolate(mask, size=(H,W), mode="nearest")
    # x = x * (mask_up + (1 - mask_up) * backw)
    x = x.view((B, C, h, window_size_h, w, window_size_w)).permute(0, 1, 2, 4, 3, 5) # (B,C,h,w,window_size_h,window_size_w)
    mask_up = mask_up.view((B, 1, h, window_size_h, w, window_size_w)).permute(0, 1, 2, 4, 3, 5) # (B,C,h,w,window_size_h,window_size_w)

    xs = x.new_empty((B, 4, C, h*w, window_size_h*window_size_w))         # 预分配输出：4条扫描序列
    masks = mask_up.new_empty((B, 4, 1, h*w, window_size_h*window_size_w))         # 预分配输出：4条扫描序列
    # mask out input 
    xs[:, 0] = x.flatten(4, 5).flatten(2, 3)            # (B, C, H//ws, W//ws, ws, ws) -> (B, C, h*w, ws**2) 先窗口内展平,再全局展平
    xs[:, 1] = x.permute(0,1,3,2,5,4).flatten(4, 5).flatten(2, 3)    # 先交换H/W，再展平 => 列优先展平
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-2, -1])  # 将前两条序列在最后一维(L)反转，得到反向扫描

    masks[:, 0] = mask_up.flatten(4, 5).flatten(2, 3)                  # (B, C, H//ws, W//ws, ws, ws) -> (B, C, h*w, ws**2) 行优先展平
    masks[:, 1] = mask_up.permute(0, 1, 3, 2, 5, 4).flatten(4, 5).flatten(2, 3)  # 先交换H/W，再展平 => 列优先展平
    masks[:, 2:4] = torch.flip(masks[:, 0:2], dims=[-2, -1])  # 将前两条序列在最后一维(L)反转，得到反向扫描

    # 获取所有batch中最大有效位置数量
    # 挑选mask=1的位置
    seq_list = []
    idx_list = []
    if debug:
        import ipdb;ipdb.set_trace()
    for i in range(B):
        # 为每个批次创建一个单独的掩码
        batch_mask = masks[i].expand(4, C, h*w, window_size_h*window_size_w).bool()
        # 使用布尔索引来提取该批次的所有元素
        selected_elements = xs[i][batch_mask]
        if selected_elements.numel() == 0:
            selected_elements = xs[i].unsqueeze(0) # (1, 4, C,  h*w, window_size_h*window_size_w)
        else:
            # 如果张量非空，执行稀疏化操作
            selected_elements = selected_elements.reshape(4, C, -1, window_size_h*window_size_w).unsqueeze(0)
        seq_list.append(selected_elements.flatten(3)) # [1, 4, C, h*w * window_size_h*window_size_w] * bs
        idx_list.append(batch_mask)
    return seq_list, idx_list, window_size_h, window_size_w, mask_up

def masked_cross_merge_v4(seq_list, idx_list, H, W, ws_h, ws_w):

    B = len(seq_list)
    # 获取所有batch中有效位置数量
    out_list = []
    for i in range(B):
        y = seq_list[i] # 4*C, l, ws_h*ws_w
        idx = idx_list[i]  # 4, C, h*w, ws_h*ws_w
        C = idx.shape[1]
        n_valid = idx[0,0].sum() // (ws_h*ws_w)
        if n_valid > 0:
            out = torch.zeros((4, C, H*W//(ws_h*ws_w), ws_h*ws_w), device=y.device, dtype=y.dtype)
            # scatter: 按 mask 填充
            # if ws == 1:
            #     y = torch.sigmoid(y.contiguous().view(-1))
            # else:
            #     y = torch.softmax(y.reshape(4, C, -1, ws_h*ws_w), -1)
            out[idx] = y.contiguous().view(-1)
            out_list.append(out.unsqueeze(0))
        else:
            out_list.append(y.reshape(1, 4, C, -1,ws_h*ws_w))
    
    out = torch.cat(out_list, 0) # B, 4, C, h*w,ws_h*ws_w
    out = out[:, 0:2] + out[:, 2:4].flip(dims=[-2, -1]) #(B, 2, C, H/ws_h, ws_h, W/ws_w, ws_w)
    y = out[:, 0].view(B, -1, H//ws_h, W//ws_w, ws_h, ws_w).permute(0,1,2,4,3,5).reshape(B, -1, H, W) + \
        out[:, 1].view(B, -1, W//ws_w, H//ws_h, ws_w, ws_h).permute(0,1,3,5,2,4).contiguous().view(B, -1, H, W)
    return y.flatten(2) / 4

def window_partition(x, window_size_h, window_size_w):
    """
    将特征图分割为窗口序列
    Args:
        x: (B, C, H, W)
    Returns:
        windows: (B, num_windows, window_area, C) -> Flattened to (B, L, C)
    """
    B, C, H, W = x.shape
    # Reshape to (B, C, h_win, ws_h, w_win, ws_w)
    x = x.view(B, C, H // window_size_h, window_size_h, W // window_size_w, window_size_w)
    # Permute to (B, h_win, w_win, ws_h, ws_w, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    # Flatten to (B, L, C)
    return x.view(B, -1, C)

def window_reverse(windows, window_size_h, window_size_w, H, W):
    """
    将窗口序列还原为特征图
    Args:
        windows: (B, L, C)
    Returns:
        x: (B, C, H, W)
    """
    B, L, C = windows.shape
    # Reshape back to (B, h_win, w_win, ws_h, ws_w, C)
    x = windows.view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, C)
    # Permute back to (B, C, h_win, ws_h, w_win, ws_w)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)

def masked_cross_scan_v5(x, mask):
    """
    修复版：当 Mask 全为 0 时，强制视为全 1（保留所有特征）
    """
    B, C, H, W = x.shape
    _, _, h, w = mask.shape
    pos_embed = PositionEmbeddingSine(C//2, normalize=True) # use sine positional embedding
    pos_mask = torch.zeros((B, H, W), dtype=torch.uint8).bool().to(x.device)
    x = pos_embed(x, pos_mask) + x
    # ============================================================
    # 🔴【修复核心】Mask 全 0 检测与修正
    # ============================================================
    # 计算每个样本 Mask 的非零元素个数
    # mask shape: (B, 1, h, w)
    mask_sums = mask.sum(dim=(1, 2, 3)) # (B,)
    
    # 找到全 0 的样本索引
    is_empty_mask = (mask_sums == 0)
    
    # 如果存在全 0 的 mask，将这些样本的 mask 变为全 1
    # 注意：我们 clone 一下，避免修改原始输入 mask 导致外部逻辑错误
    if is_empty_mask.any():
        mask = mask.clone()
        mask[is_empty_mask] = 1.0
        # 此时，原来全 0 的位置变成了全 1，Scan 阶段会选取所有 token
    # ============================================================

    ws_h = H // h if H > h else 1
    ws_w = W // w if W > w else 1
    
    # 1. Mask 上采样 (B, 1, H, W)
    mask_up = F.interpolate(mask.float(), size=(H, W), mode="nearest")
    # Mask 展平为 (B, L)
    mask_flat = window_partition(mask_up, ws_h, ws_w).squeeze(-1)

    # 2. 生成 4 个方向的输入序列 (B, L, C)
    x0 = window_partition(x, ws_h, ws_w)
    m0 = mask_flat
    
    x_trans = x.permute(0, 1, 3, 2)
    m_trans = mask_up.permute(0, 1, 3, 2)
    x1 = window_partition(x_trans, ws_w, ws_h)
    m1 = window_partition(m_trans, ws_w, ws_h).squeeze(-1)
    
    x2 = torch.flip(x0, dims=[1])
    m2 = torch.flip(m0, dims=[1])
    x3 = torch.flip(x1, dims=[1])
    m3 = torch.flip(m1, dims=[1])
    
    # 堆叠 (B, 4, L, C)
    xs_all = torch.stack([x0, x1, x2, x3], dim=1)
    ms_all = torch.stack([m0, m1, m2, m3], dim=1).bool()
    
    # 3. 稀疏化
    # 此时如果原 mask 全为 0，现在已经是全 1，所以 sort 后所有元素都在前面
    sorted_mask, indices = torch.sort(ms_all.int(), dim=2, descending=True) 
    
    valid_lens = sorted_mask.sum(dim=2)
    max_len = valid_lens.max().item()
    
    # Gather: (B, 4, L, C)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, C)
    xs_sorted = torch.gather(xs_all, 2, indices_expanded)
    
    # 4. 裁剪并转置为 (B, 4, C, L)
    # 此时，对于全 0 Mask 的样本，max_len 等于 Total_Len，保留了全部特征
    xs_out = xs_sorted[:, :, :max_len, :].permute(0, 1, 3, 2).contiguous()
    
    return xs_out, indices, (H, W, ws_h, ws_w), valid_lens

def masked_cross_merge_v5(xs_out, indices, shapes):
    """
    Args:
        xs_out: (B, 4*C, L_valid) <-- 你的输入形状
        indices: (B, 4, Total_Len)
        shapes: (H, W, ws_h, ws_w)
    """
    # 1. 解析维度
    B, FourC, L = xs_out.shape
    H, W, ws_h, ws_w = shapes
    Total_Len = indices.shape[2]
    
    # 确保通道数是 4 的倍数
    assert FourC % 4 == 0, f"Input channels {FourC} is not divisible by 4 directions"
    C = FourC // 4
    
    # 2. 【核心修正】拆分通道并转置
    # (B, 4*C, L) -> (B, 4, C, L)
    xs_out = xs_out.view(B, 4, C, L)
    
    # (B, 4, C, L) -> (B, 4, L, C) 准备 Scatter
    xs_permuted = xs_out.permute(0, 1, 3, 2).contiguous()
    
    # 3. Padding (补齐长度)
    if L < Total_Len:
        pad_len = Total_Len - L
        # 只在 L 维度(倒数第2维) 右侧补 0
        xs_permuted = F.pad(xs_permuted, (0, 0, 0, pad_len))
        
    # 4. Scatter 还原 (初始化为 0)
    restored = torch.zeros((B, 4, Total_Len, C), device=xs_out.device, dtype=xs_out.dtype)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, C)
    
    restored.scatter_(2, indices_expanded, xs_permuted)
    
    # 5. 分离方向并还原
    r0, r1, r2, r3 = restored[:, 0], restored[:, 1], restored[:, 2], restored[:, 3]
    
    r2 = torch.flip(r2, dims=[1])
    r3 = torch.flip(r3, dims=[1])
    
    y_h = r0 + r2
    y_v = r1 + r3
    
    out_h = window_reverse(y_h, ws_h, ws_w, H, W)
    out_v_trans = window_reverse(y_v, ws_w, ws_h, W, H)
    out_v = out_v_trans.permute(0, 1, 3, 2)
    
    return (out_h + out_v) / 4.0


def visualize_scan_logic(H=8, W=8, ws=4):
    """
    可视化 4 个方向的扫描路径
    """
    print(f"Generating Visualization for {H}x{W} Grid (Window Size: {ws})...")
    
    # --- A. 构造坐标 ID图 ---
    # 我们用像素的索引值 (0 到 63) 代替特征值
    # 这样扫描出来的结果就是像素的访问顺序
    coords = torch.arange(H * W).view(1, 1, H, W).float()
    
    # --- B. 构造一个 Mask (中间挖空) ---
    mask = torch.ones(1, 1, H//ws, W//ws)
    # 为了演示 Mask 效果，我们把 (0,1) 这个窗口设为 0 (无效)
    # 如果 H=8, W=8, ws=4，那么一共有 2x2=4 个窗口
    # 我们 mask 掉右上角的窗口
    mask[0, 0, 0, 2] = 0 
    mask[0, 0, 1, 0] = 0 
    mask[0, 0, 1, 2] = 0 
    mask[0, 0, 2, 1] = 0 
    
    # 上采样 Mask 以便应用
    mask_up = F.interpolate(mask, size=(H, W), mode='nearest')
    
    # --- C. 执行扫描 (手动展开逻辑以便提取) ---
    # Dir 0: Horizontal Window
    x0 = window_partition(coords, ws, ws)          # 展平
    m0 = window_partition(mask_up, ws, ws).squeeze(-1) # Mask
    
    # Dir 1: Vertical Window (Transpose H/W)
    coords_t = coords.permute(0, 1, 3, 2) 
    mask_t = mask_up.permute(0, 1, 3, 2)
    x1 = window_partition(coords_t, ws, ws)
    m1 = window_partition(mask_t, ws, ws).squeeze(-1)
    
    # Dir 2 & 3: Flip
    x2 = torch.flip(x0, dims=[1])
    m2 = torch.flip(m0, dims=[1])
    x3 = torch.flip(x1, dims=[1])
    m3 = torch.flip(m1, dims=[1])
    
    # Stack
    xs_all = torch.stack([x0, x1, x2, x3], dim=1) # (1, 4, L, 1)
    ms_all = torch.stack([m0, m1, m2, m3], dim=1).bool()
    
    # --- D. 绘图 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    titles = ["Dir 0: Horizontal", "Dir 1: Vertical", "Dir 2: Flip Horizontal", "Dir 3: Flip Vertical"]
    
    for i, ax in enumerate(axes.flat):
        # 1. 获取当前方向的序列
        current_seq_vals = xs_all[0, i, :, 0] # 坐标ID
        current_mask = ms_all[0, i, :]        # Mask
        
        # 提取 Mask=1 的有效路径
        valid_path = current_seq_vals[current_mask]
        
        # 将 ID 转回 (y, x) 坐标
        path_y = (valid_path // W).numpy()
        path_x = (valid_path % W).numpy()
        
        # 2. 绘制背景网格
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlim(-0.5, W-0.5)
        ax.set_ylim(-0.5, H-0.5)
        ax.invert_yaxis() # 图像坐标系 y 向下
        ax.set_aspect('equal')
        ax.set_title(titles[i])
        
        # 3. 标记 Mask 区域
        # 绘制被 Mask 掉的区域 (灰色背景)
        mask_np = mask_up[0,0].numpy()
        for y in range(H):
            for x in range(W):
                if mask_np[y, x] == 0:
                    rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, color='lightgray', alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(x, y, "X", ha='center', va='center', color='gray')

        # 4. 绘制路径箭头
        # 起点
        if len(path_x) > 0:
            ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
            
            # 绘制箭头流
            ax.quiver(path_x[:-1], path_y[:-1], 
                      path_x[1:]-path_x[:-1], path_y[1:]-path_y[:-1], 
                      scale_units='xy', angles='xy', scale=1, 
                      width=0.005, color='blue', alpha=0.8)
            
            # 标上序号
            for t, (px, py) in enumerate(zip(path_x, path_y)):
                ax.text(px, py, str(t), fontsize=12, ha='center', va='center', 
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="blue", alpha=0.7))

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_hybrid_scan_path():
    # ============================
    # 1. 参数设置
    # ============================
    H, W = 16, 16      # 图像大小
    ws = 4             # 窗口大小 (4x4)
    h_win, w_win = H // ws, W // ws

    # 定义 Mask: 1=前景(FG), 0=背景(BG)
    # 我们设置中间区域为前景，四周为背景
    mask = np.zeros((h_win, w_win))
    mask[1:3, 1:3] = 1  # 中间 2x2 个窗口是前景

    # ============================
    # 2. 生成路径坐标序列
    # ============================
    # 我们按照 Window-Major 的顺序扫描 (即先扫完 Win0, 再 Win1...)
    # 坐标格式 (x, y) 用于绘图
    path_coords = []
    node_types = [] # 'fg' or 'bg'

    # 遍历每一个窗口
    for wy in range(h_win):
        for wx in range(w_win):
            is_fg = mask[wy, wx] == 1
            
            # 计算窗口在原图的左上角坐标
            base_x = wx * ws
            base_y = wy * ws
            
            if is_fg:
                # === 前景窗口：扫描每一个像素 ===
                # 这里演示简单的光栅扫描 (Row-Major)
                # 如果你的 Mamba 用的是 Snake (S形)，这里可以改为 S 形逻辑
                for i in range(ws):
                    for j in range(ws):
                        # 像素坐标 (注意 y 是行, x 是列)
                        # 这里为了绘图清晰，我们将点画在像素网格中心，所以 +0.5
                        px = base_x + j + 0.5
                        py = base_y + i + 0.5
                        path_coords.append((px, py))
                        node_types.append('fg')
            else:
                # === 背景窗口：只扫描一个点 (均值中心) ===
                # 点位于窗口中心
                cx = base_x + ws / 2.0
                cy = base_y + ws / 2.0
                path_coords.append((cx, cy))
                node_types.append('bg')

    # ============================
    # 3. 绘图可视化
    # ============================
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- A. 绘制底图网格 ---
    # 设置背景色区分 FG 和 BG 区域
    for wy in range(h_win):
        for wx in range(w_win):
            color = '#e6f2ff' if mask[wy, wx] == 1 else '#ffe6e6' # 蓝底FG, 红底BG
            rect = patches.Rectangle((wx*ws, wy*ws), ws, ws, linewidth=1, edgecolor='gray', facecolor=color)
            ax.add_patch(rect)
            
            # 标注窗口类型
            label = "FG\n(Full)" if mask[wy, wx] == 1 else "BG\n(Mean)"
            ax.text(wx*ws + 0.5, wy*ws + 0.5, label, fontsize=8, color='black', alpha=0.5, verticalalignment='top')

    # 画像素网格线 (虚线)
    for i in range(H + 1):
        ax.axhline(i, color='lightgray', linestyle=':', linewidth=0.5)
    for j in range(W + 1):
        ax.axvline(j, color='lightgray', linestyle=':', linewidth=0.5)

    # --- B. 绘制扫描路径 ---
    coords = np.array(path_coords)
    X, Y = coords[:, 0], coords[:, 1]
    
    # 画连线 (表示扫描顺序)
    ax.plot(X, Y, color='green', alpha=0.6, linewidth=1, label='Scan Path')
    
    # 画箭头 (每隔几个点画一个，避免太乱)
    # 对于 BG -> FG 的跳跃处画箭头更有意义
    for i in range(len(X) - 1):
        # 如果距离较远(跨窗口)，画明显的箭头
        dist = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2)
        if dist > 1.5: 
            ax.arrow(X[i], Y[i], (X[i+1]-X[i])*0.8, (Y[i+1]-Y[i])*0.8, 
                     head_width=0.3, head_length=0.3, fc='green', ec='green', alpha=0.8)

    # --- C. 绘制节点 ---
    # 分离 FG 和 BG 节点以便用不同样式绘制
    fg_x = [p[0] for p, t in zip(path_coords, node_types) if t == 'fg']
    fg_y = [p[1] for p, t in zip(path_coords, node_types) if t == 'fg']
    bg_x = [p[0] for p, t in zip(path_coords, node_types) if t == 'bg']
    bg_y = [p[1] for p, t in zip(path_coords, node_types) if t == 'bg']

    ax.scatter(fg_x, fg_y, c='blue', s=20, label='FG Pixel Token')
    ax.scatter(bg_x, bg_y, c='red', marker='*', s=200, label='BG Mean Token', edgecolors='black')

    # --- D. 设置坐标轴 ---
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0) # 图像坐标系 y 轴向下
    ax.set_xticks(np.arange(0, W+1, ws))
    ax.set_yticks(np.arange(0, H+1, ws))
    ax.grid(False) # 我们自己画了网格
    ax.set_title(f"Hybrid Granularity Scan Path\nImage: {H}x{W}, Window: {ws}x{ws}", fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 辅助函数
# ==========================================
def window_partition_mixed(x, ws_h, ws_w):
    """专门为混合扫描设计的 Partition，保留窗口维度"""
    B, C, H, W = x.shape
    # (B, C, h, ws_h, w, ws_w)
    x = x.view(B, C, H // ws_h, ws_h, W // ws_w, ws_w)
    # (B, h, w, ws_h, ws_w, C) -> (B, N, Area, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(B, -1, ws_h * ws_w, C)

def window_reverse(windows, ws_h, ws_w, H, W):
    B = windows.shape[0]
    C = windows.shape[-1]
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)
# ==========================================
# 1. Hybrid Scan (混合粒度扫描) - Soft-Pooling版
# ==========================================
def masked_cross_scan_v6(x, mask, pe):
    """
    Efficiency Optimized Version: Uses sort/gather to avoid Python loops.
    Feature: Soft-Pooling Background Aggregation (No Mask Dilation)
    """
    B, C, H, W = x.shape
    _, _, h, w = mask.shape
    ws_h, ws_w = H // h, W // w
    ws_area = ws_h * ws_w
    
    # 1. Mask Correction (直接使用输入的 mask，不进行膨胀)
    mask_sums = mask.sum(dim=(1, 2, 3))
    is_empty_mask = (mask_sums == 0)
    if is_empty_mask.any():
        mask = mask.clone()
        mask[is_empty_mask] = 1.0
    
    pos_embed = PositionEmbeddingSine(C//2, normalize=True) # use sine positional embedding
    pos_mask = torch.zeros((B, H, W), dtype=torch.uint8).bool().to(x.device)
    x = pos_embed(x, pos_mask) + x

    # 2. View Generation (Stacking)
    x0, m0 = x, mask
    x1, m1 = x.transpose(-1, -2), mask.transpose(-1, -2)
    x2, m2 = x.flip(-1), mask.flip(-1)
    x3, m3 = x.transpose(-1, -2).flip(-1), mask.transpose(-1, -2).flip(-1)
    
    x_all = torch.cat([x0, x1, x2, x3], dim=0) # (4B, C, H, W)
    m_all = torch.cat([m0, m1, m2, m3], dim=0) # (4B, 1, h, w)
    
    # 3. Partition & Mix
    # x_wins: (4B, N, Area, C)
    x_wins = window_partition_mixed(x_all, ws_h, ws_w)
    
    # Process Mask
    mask_flat = m_all.view(4*B, -1, 1) # (4B, N, 1)
    is_fg = (mask_flat > 0.5)
    
    # ============================================================
    # [修改核心] A. Soft-Pooling Background Injection
    # 原理：根据特征能量(L2 Norm)加权，保留高频/高亮特征，丢弃无效背景
    # ============================================================
    
    # 1. 计算重要性分数 (L2 Norm, 越大代表特征越显著)
    # score: (4B, N, Area, 1)
    # keepdim=True 保证维度方便后续广播
    pixel_scores = torch.norm(x_wins, dim=-1, keepdim=True)
    
    # 2. 计算注意力权重 (在窗口内部 Area 维度做 Softmax)
    # 这样窗口内能量高的像素会获得更大的权重
    pixel_weights = F.softmax(pixel_scores, dim=2) 
    
    # 3. 加权融合生成 Representative Token
    # bg_feature: (4B, N, 1, C)
    bg_feature = (x_wins * pixel_weights).sum(dim=2, keepdim=True)

    # 4. 注入到背景窗口的 index 0
    x_mixed = x_wins.clone()
    
    # 仅替换背景窗口 (is_bg) 的第0个位置
    is_bg_idx0 = (~is_fg).unsqueeze(-1).expand(-1, -1, 1, C)
    x_mixed[:, :, 0:1, :] = torch.where(is_bg_idx0, bg_feature, x_mixed[:, :, 0:1, :])
    
    # ============================================================
    
    # B. Construct Keep Mask (Priority Map)
    priority_mask = torch.ones((4*B, h*w, ws_area), device=x.device, dtype=torch.int32)
    
    # Background: index 1~Area-1 are 0 (Discard)
    is_fg_expanded = is_fg.expand(-1, -1, ws_area - 1).int()
    priority_mask[:, :, 1:] = is_fg_expanded 
    
    # Flatten: (4B, N*Area)
    flat_priority = priority_mask.view(4*B, -1)
    flat_mixed = x_mixed.view(4*B, -1, C)
    
    # C. Vectorized Gather
    # Sort mask descending: 1s move to front, 0s move to back
    sorted_mask, indices = torch.sort(flat_priority, dim=1, descending=True, stable=True)
    
    # Determine Max Len
    valid_lens = sorted_mask.sum(dim=1) # (4B,)
    max_len = valid_lens.max().item()
    
    # Expand indices for gather
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
    
    # Gather
    xs_gathered = torch.gather(flat_mixed, 1, indices_expanded)
    
    # Slice to max_len
    xs_out = xs_gathered[:, :max_len, :]
    
    # Reshape for output
    xs_out = xs_out.view(B, 4, max_len, C).permute(0, 1, 3, 2).reshape(B, 4, C, max_len)

    restore_info = {
        "B": B, "C": C, "H": H, "W": W,
        "ws_h": ws_h, "ws_w": ws_w,
        "indices": indices,            
        "valid_lens": valid_lens,
        "is_fg": is_fg,
        "num_windows": h*w,
        "total_tokens": flat_mixed.shape[1]
    }
    
    return xs_out, restore_info


# ==========================================
# 2. Hybrid Merge - Vectorized
# ==========================================
def masked_cross_merge_v6(xs_out, restore_info):
    """
    Efficiency Optimized Version: Uses scatter with saved indices.
    """
    B = restore_info["B"]
    C = restore_info["C"]
    H, W = restore_info["H"], restore_info["W"]
    ws_h, ws_w = restore_info["ws_h"], restore_info["ws_w"]
    ws_area = ws_h * ws_w
    
    indices = restore_info["indices"]          
    valid_lens = restore_info["valid_lens"]    
    num_windows = restore_info["num_windows"]
    is_fg = restore_info["is_fg"]              
    total_tokens = restore_info["total_tokens"]
    
    # 1. Unpack Input
    L = xs_out.shape[-1]
    xs_out = xs_out.view(B, 4, C, L).permute(0, 1, 3, 2).reshape(4*B, L, C)
    
    # 2. Vectorized Restore
    container = torch.zeros((4*B, total_tokens, C), device=xs_out.device, dtype=xs_out.dtype)

    gather_indices = indices[:, :L].unsqueeze(-1).expand(-1, -1, C)
    
    # Scatter back!
    container.scatter_(1, gather_indices, xs_out)
    
    # 3. Reshape & Broadcast
    x_wins = container.view(4*B, num_windows, ws_area, C)
    
    # ============================================================
    # [还原] 使用 Soft-Pooled Feature 填充背景
    # ============================================================
    # 取出 index 0 (这是我们在 Scan 阶段精心计算的 Attention 特征)
    bg_feature = x_wins[:, :, 0:1, :] # (4B, N, 1, C)
    
    # 广播填满整个背景窗口
    bg_expanded = bg_feature.expand(-1, -1, ws_area, -1)
    
    # 前景用 x_wins (还原的像素), 背景用 bg_expanded (广播的特征)
    is_fg_exp = is_fg.unsqueeze(-1).expand(-1, -1, ws_area, C)
    x_restored = torch.where(is_fg_exp, x_wins, bg_expanded)
    
    # 4. Inverse Transform & Merge
    x_img_all = window_reverse(x_restored, ws_h, ws_w, H, W)
    x0, x1, x2, x3 = x_img_all.chunk(4, dim=0)
    
    x1 = x1.transpose(-1, -2)
    x2 = x2.flip(-1)
    x3 = x3.flip(-1).transpose(-1, -2)
    return (x0 + x1 + x2 + x3) / 4.0

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mcoll
import numpy as np

# =========================================================================
# 1. 准备工作：确保 masked_cross_scan_v6 在上下文中
#    (这里直接复用你刚才生成的函数，无需修改)
# =========================================================================
# 请确保 masked_cross_scan_v6 和 window_partition_mixed 已经定义
# 如果没有定义，请将上一条回复中的函数复制到这里。

def visualize_real_torch_scan():
    # ============================
    # 1. 构建特殊的"坐标特征图"
    # ============================
    H, W = 9, 9
    ws = 3
    
    # 生成网格坐标 (H, W)
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # 加上 0.5 让我们画在像素中心
    grid_y = grid_y.float() + 0.5
    grid_x = grid_x.float() + 0.5
    
    # 构建 Mask: 中间区域为前景 (1), 四周为背景 (0)
    # Mask 需要是 (B, 1, h, w)
    h_win, w_win = H // ws, W // ws
    mask_small = torch.ones(1, 1, h_win, w_win)
     
    mask_small[0, 0, 0, 2] = 0 
    mask_small[0, 0, 1, 0] = 0 
    mask_small[0, 0, 1, 2] = 0 
    mask_small[0, 0, 2, 1] = 0 

    
    # 构建类型标记图 (Type Map)
    # 我们需要知道哪个像素是前景。Mask 是粗粒度的，我们需要细粒度的 Map。
    # 上采样 Mask 到 (H, W)
    mask_pixel = F.interpolate(mask_small, size=(H, W), mode='nearest').squeeze() # (H, W)
    
    # 构造 Input Tensor (B=1, C=3, H, W)
    # Ch 0: Y 坐标
    # Ch 1: X 坐标
    # Ch 2: 类型标记 (前景=1.0, 背景=-1.0) 
    #       这样背景压缩取均值后依然是 -1.0，我们可以通过 >0 判断类型
    type_map = torch.where(mask_pixel > 0.5, 1.0, -1.0)
    
    input_tensor = torch.stack([grid_y, grid_x, type_map], dim=0).unsqueeze(0) # (1, 3, 16, 16)
    
    # ============================
    # 2. 运行真实的 Scan 函数
    # ============================
    print("正在运行 masked_cross_scan_v6 ...")
    # xs_out: (B, 4*C, L) -> (1, 4*3, L)
    # info: dict
    with torch.no_grad():
        xs_out, info = masked_cross_scan_v6(input_tensor, mask_small)
        
    print(f"Scan 完成! 输出序列形状: {xs_out.shape}")
    
    # ============================
    # 3. 解析输出数据
    # ============================
    # xs_out 的 shape 是 (1, 12, L)。因为输入 C=3，4个方向堆叠在 Channel 维。
    # 我们需要拆解出 4 个方向的数据。
    
    # Reshape: (1, 4*3, L) -> (4, 3, L)
    # 这里的顺序对应: 0:Orig, 1:Trans, 2:Flip, 3:Trans+Flip
    L = xs_out.shape[-1]
    scan_data = xs_out.view(4, 3, L).cpu().numpy()
    
    # ============================
    # 4. 可视化绘图 (复用之前的绘图逻辑)
    # ============================
    directions = ["0: Horizontal", "1: Vertical", "2: Flip H", "3: Flip V"]
    cmap = plt.get_cmap('turbo') 
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    for i in range(4):
        ax = axes[i]
        
        # 获取当前方向的数据
        # Y坐标, X坐标, 类型标记
        curr_y = scan_data[i, 0, :]
        curr_x = scan_data[i, 1, :]
        curr_type = scan_data[i, 2, :]
        
        # 过滤掉 Padding 的部分 (如果有的话)
        # 我们的 Scan 代码做了 Padding，填充的是 0。
        # 我们可以通过 mask 或者坐标是否为 0 来判断（因为我们输入坐标加了 0.5，不可能是 0）
        valid_idx = (curr_y != 0) & (curr_x != 0)
        
        Y = curr_y[valid_idx]
        X = curr_x[valid_idx]
        T = curr_type[valid_idx]
        
        # 生成时间索引
        steps = np.linspace(0, 1, len(Y))
        
        # --- A. 绘制底图网格 ---
        for wy in range(h_win):
            for wx in range(w_win):
                # 画物理网格
                rect = patches.Rectangle((wx*ws, wy*ws), ws, ws, linewidth=1, 
                                       edgecolor='#dddddd', facecolor='none', linestyle=':')
                ax.add_patch(rect)
                
        # --- B. 绘制渐变路径 ---
        # 构造线段
        points = np.array([X, Y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(steps[:-1])
        lc.set_linewidth(1.5)
        lc.set_alpha(0.8)
        ax.add_collection(lc)
        
        # --- C. 绘制节点 ---
        # 判断类型: T > 0 是前景(1.0), T < 0 是背景(-1.0)
        fg_mask = T > 0
        bg_mask = T < 0
        
        # 前景节点
        if np.any(fg_mask):
            ax.scatter(X[fg_mask], Y[fg_mask], c=steps[fg_mask], cmap=cmap, 
                       s=15, marker='o', label='FG Pixel')
        
        # 背景节点
        if np.any(bg_mask):
            ax.scatter(X[bg_mask], Y[bg_mask], c=steps[bg_mask], cmap=cmap, 
                       s=200, marker='*', edgecolors='black', label='BG Mean')
            
        # --- D. 标注 ---
        ax.text(X[0], Y[0], "Start", fontsize=10, fontweight='bold', color=cmap(0.0), ha='right')
        ax.text(X[-1], Y[-1], "End", fontsize=10, fontweight='bold', color=cmap(1.0), ha='left')
        
        # 样式
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"{directions[i]}\n(Real PyTorch Output)", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        if i == 0:
            ax.legend(loc='lower left')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(lc, cax=cbar_ax)
    cbar.set_label('Mamba Scan Order', rotation=270, labelpad=15)
    
    plt.suptitle("Visualize Real masked_cross_scan_v6 Output (via Coordinate Embedding)", fontsize=16)
    plt.show()


if __name__ == "__main__":
    # 使用 8x8 的图，窗口大小 4x4，这样刚好有 4 个窗口
    # visualize_scan_logic(H=9, W=9, ws=3)
    
    visualize_real_torch_scan()
