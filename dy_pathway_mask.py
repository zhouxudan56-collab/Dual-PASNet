import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPathwayMask(nn.Module):
    """动态通路掩码模块：学习基因与通路的关联强度"""

    def __init__(self, init_mask, temperature=1.0):
        super(DynamicPathwayMask, self).__init__()  # 先初始化父类
        self.num_pathways, self.num_genes = init_mask.shape
        # 温度参数设为不可训练（requires_grad=False）
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

        # 修复：用clone().detach()避免复制警告，保持与输入张量设备一致
        init_mask = init_mask.clone().detach().float()  # 替代 torch.tensor(init_mask)
        self.mask_logits = nn.Parameter(torch.log(init_mask + 1e-8))  # 可学习的掩码参数

        # 保存先验掩码（用register_buffer避免被视为可训练参数）
        self.register_buffer('prior_mask', init_mask)

    def forward(self):
        # 生成连续值掩码（添加微小噪声增强稳定性）
        gumbel_noise = torch.rand_like(self.mask_logits) * 1e-10
        dynamic_mask = F.softmax(
            (self.mask_logits + gumbel_noise) / self.temperature,
            dim=-1  # 沿基因维度归一化（每个通路内的基因权重求和为1）
        )
        return dynamic_mask


class PASNetFeatureExtractor(nn.Module):
    """PASNet特征提取器，修复全连接层维度不匹配与初始化顺序问题"""

    def __init__(self, in_nodes, pathway_nodes, hidden_nodes, pathway_mask):
        # 修复1：先调用父类初始化（必须放在最前面）
        super(PASNetFeatureExtractor, self).__init__()

        # 修复2：父类初始化后，再定义子模块（DynamicPathwayMask）
        self.dynamic_mask = DynamicPathwayMask(
            init_mask=pathway_mask,
            temperature=0.5  # 控制掩码稀疏性：温度越低，掩码越接近0/1
        )
        self.pathway_nodes = pathway_nodes  # 通路数量（如37）
        self.hidden_nodes = hidden_nodes  # 隐藏层维度（如100）

        # 通路特征处理层（维度匹配：pathway_nodes → pathway_nodes → hidden_nodes）
        self.layer1 = nn.Linear(self.pathway_nodes, self.pathway_nodes)
        self.layer2 = nn.Linear(self.pathway_nodes, self.hidden_nodes)
        self.relu = nn.ReLU()

    def forward(self, x, do_m1=None, do_m2=None):
        # x形状：[batch_size, in_nodes]（如 [64, 1024]，in_nodes为基因数）

        # 1. 生成动态掩码并与基因特征加权融合
        mask = self.dynamic_mask()  # 形状：[num_pathways, in_nodes]（如 [37, 1024]）
        mask = mask.unsqueeze(0)  # 扩展 batch 维度：[1, 37, 1024]
        weighted_gene = x.unsqueeze(1) * mask  # 广播匹配：[batch, 37, 1024]
        pathway_features = weighted_gene.sum(dim=2)  # 沿基因维度求和：[batch, 37]

        # 2. 第一层全连接（通路特征变换）
        x = self.layer1(pathway_features)  # [batch, 37] → [batch, 37]
        x = self.relu(x)

        # 3. 应用通路层dropout（若do_m1不为None）
        if do_m1 is not None:
            # 确保do_m1维度匹配：[1, 37]（广播到batch维度）
            x = x * do_m1

        # 4. 第二层全连接（映射到隐藏层）
        x = self.layer2(x)  # [batch, 37] → [batch, hidden_nodes]（如 [64, 100]）
        x = self.relu(x)

        # 5. 应用隐藏层dropout（若do_m2不为None）
        if do_m2 is not None:
            # 确保do_m2维度匹配：[1, hidden_nodes]
            x = x * do_m2

        # 修复3：移出条件判断，确保无论do_m2是否存在都返回值
        # 返回：提取的特征（x）、动态掩码（mask）（掩码用于可解释性分析）
        return x, mask.squeeze(0)  # mask挤压掉batch维度：[37, 1024]

    def get_mask_analysis(self):
        """获取掩码分析信息（用于模型解释：当前掩码、先验掩码、变化最大的基因）"""
        # 计算当前稳定掩码（无噪声）
        current_mask = F.softmax(self.dynamic_mask.mask_logits / self.dynamic_mask.temperature, dim=-1)
        # 计算掩码变化量（当前掩码 - 先验掩码的绝对值）
        mask_change = torch.abs(current_mask - self.dynamic_mask.prior_mask)
        # 返回：当前掩码、先验掩码、变化最大的前5个基因索引（每个通路）
        return current_mask, self.dynamic_mask.prior_mask, torch.topk(mask_change, k=5, dim=-1).indices


class MultimodalPASNet(nn.Module):
    """多模态PASNet模型（修复共享层定义，确保维度匹配）"""

    def __init__(self, pathway_extractor, image_dim=768, fusion_dim=256):
        super(MultimodalPASNet, self).__init__()
        self.pathway_extractor = pathway_extractor  # 注入通路特征提取器
        self.image_dim = image_dim  # 图像特征维度（如768）
        self.fusion_dim = fusion_dim  # 多模态融合维度（如256）

        # 图像特征投影层（将图像特征映射到融合维度）
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),  # 批归一化增强稳定性
            nn.ReLU()
        )

        # 通路特征投影层（将通路隐藏层特征映射到融合维度）
        # 注意：输入维度需与PASNetFeatureExtractor的hidden_nodes一致（如100）
        self.pathway_proj = nn.Sequential(
            nn.Linear(self.pathway_extractor.hidden_nodes, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU()
        )

        # 注意力融合层（学习通路/图像特征的权重）
        self.attention = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),  # 输入：拼接后的特征（2*fusion_dim）
            nn.Tanh(),
            nn.Linear(self.fusion_dim, 2)  # 输出：2个权重（通路、图像）
        )

        # 共享特征提取层（融合后特征进一步抽象）
        self.shared_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU()
        )

        # 分类头（二分类：输出概率0~1）
        self.cls_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout防止过拟合
            nn.Linear(128, 1),
            nn.Sigmoid()  # 二分类概率激活
        )

        # 风险得分头（输出连续值：无激活函数，范围根据任务调整）
        self.risk_head = nn.Linear(128, 1)

    def forward(self, gene_feat, image_feat, do_m1=None, do_m2=None):
        # 1. 提取通路特征与动态掩码（从PASNetFeatureExtractor获取）
        pathway_feat, dynamic_mask = self.pathway_extractor(gene_feat, do_m1, do_m2)
        # 2. 投影通路特征到融合维度
        pathway_proj = self.pathway_proj(pathway_feat)  # [batch, 100] → [batch, 256]
        # 3. 投影图像特征到融合维度
        image_proj = self.image_proj(image_feat)  # [batch, 768] → [batch, 256]

        # 4. 注意力融合（学习特征权重）
        combined = torch.cat([pathway_proj, image_proj], dim=1)  # [batch, 512]
        attn_weights = torch.softmax(self.attention(combined), dim=1)  # [batch, 2]
        # 加权融合：通路特征*通路权重 + 图像特征*图像权重
        fused_feat = attn_weights[:, 0].unsqueeze(1) * pathway_proj + \
                     attn_weights[:, 1].unsqueeze(1) * image_proj  # [batch, 256]

        # 5. 共享特征提取
        shared_feat = self.shared_layer(fused_feat)  # [batch, 256] → [batch, 128]

        # 6. 双输出：分类概率 + 风险得分 + 注意力权重 + 动态掩码
        cls_pred = self.cls_head(shared_feat)  # [batch, 1]（二分类概率）
        risk_score = self.risk_head(shared_feat)  # [batch, 1]（连续风险得分）

        return cls_pred, risk_score, attn_weights, dynamic_mask



