import torch
import torch.linalg as la
import scipy.linalg as sla
import typing
import numpy as np

class LinearCoModel:
    def __init__(self, loss_type='l2', lambda1=0.1,device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # self.X = torch.tensor(X, device=self.device, dtype=torch.float32) 
        # self.batch_size, self.n, self.d = X.shape
        self.loss_type = loss_type
        self.lambda1 = lambda1
        # 使用正确的einsum形式来计算协方差矩阵

    def _score(self, W: torch.Tensor) -> typing.Tuple[float, torch.Tensor]:
        """计算因果结构得分函数及其梯度
        
        参数:
            W (torch.Tensor): 待评估的因果权重矩阵，形状为 [batch_size, d, d]，
                             其中d表示变量数量
            
        返回:
            Tuple[float, torch.Tensor]: 
                - 第一个元素为平均损失值（标量）
                - 第二个元素为梯度矩阵，形状为 [d, d]
        
        数学原理:
            基于最小二乘损失函数：
            L(W) = 1/(2d²) * Σ_{batch} ||(I - W) @ X||²_F
            其中：
            - I 是单位矩阵
            - ||·||_F 表示Frobenius范数
            - X 是输入数据（通过cov矩阵隐含）
        
        计算过程:
            1. 计算残差矩阵: difs = (I - W)
            2. 计算右乘项: rhs = cov @ (I - W)
            3. 计算损失值: losses = 0.5 * trace(difs^T @ rhs) / d²
            4. 计算梯度: G_losses = -Σ(rhs)
        """
        W = W.to(self.device)  # 确保 W 在正确的设备
        Ids = self.Id.expand(self.batch_size, self.d, self.d)
        difs = Ids - W
        rhs = torch.einsum('bij,bjk->bik', self.cov, difs)
        losses = 0.5 * torch.einsum('bij,bji->b', difs, rhs) / (self.d * self.d)
        G_losses = -rhs.sum(axis=0)
        return losses.mean(), G_losses

    def _h(self, W: torch.Tensor, s: float = 1.0) -> typing.Tuple[float, torch.Tensor]:

        """计算DAG约束函数h(W)及其梯度
    
        参数:
        W (torch.Tensor): 因果权重矩阵，形状为 [d, d]
        s (float): 初始缩放因子（默认1.0），用于确保矩阵正定性
        
        返回:
        Tuple[float, torch.Tensor]: 
            - h(W): 约束函数值（标量）
            - ∇h(W): 梯度矩阵，形状同W [d, d]
            
        数学原理:
        定义矩阵 M = sI - W∘W
        h(W) = log|det(M)| + d*log(s)
        该函数满足 h(W)=0 ⇔ W是无环图
        
        计算过程:
        1. 通过Cholesky分解验证M的正定性
        2. 计算M的逆矩阵（带数值稳定性处理）
        3. 计算行列式的对数
        4. 计算梯度：∇h(W) = 2W(sI - WᵀW)⁻ᵀ
        
        注意:
        - 使用epsilon保证数值稳定性
        - 自动调整s直到M正定
        """
        
        W = W.to(self.device)  
        epsilon = 1e-6
        M = None
        success = False

        while not success:
            M = s * self.Id -torch.matmul( W , W)
            try:
                # 尝试使用 Cholesky 分解来确认 M 是正定的
                torch.linalg.cholesky(M)
                success = True
            except RuntimeError:
                # 如果 M 不是正定的，增加 s
                s *= 2


        # M = s * self.Id - torch.matmul(W,W)
        M_inv = torch.linalg.inv(M)
        # 在 M_inv 上加上一个小的常数 epsilon
        M_inv += epsilon * torch.eye(self.d, dtype=torch.float32, device=self.device)

        sign, logabsdet = torch.linalg.slogdet(M)
        h = logabsdet + self.d * torch.log(torch.tensor(s, device=self.device))
        M_inv_transposed = M_inv.T  # 注意这里取了转置
        G_h = 2 *torch.matmul( W , M_inv_transposed)
        return h, G_h

    def integrated_loss(self,pred:typing.List[torch.Tensor], W: torch.Tensor, mu: float, s: float = 1.0) -> typing.Tuple[float, torch.Tensor]:
       
        W = W.to(self.device)  
        self.X = torch.stack(pred,dim=2).squeeze(dim=3).to(self.device)  
        self.batch_size, self.n, self.d = self.X.shape
        self.Id = torch.eye(self.d, dtype=torch.float32,device=self.device)
        self.cov = torch.einsum('bni,bnj->bij', self.X, self.X) / self.n

        score_loss, score_grad = self._score(W)
        h_loss, h_grad = self._h(W, s)
        total_loss = mu * (score_loss + self.lambda1 * torch.abs(W).sum()) + h_loss
        total_grad = mu * (score_grad + self.lambda1 * torch.sign(W)) + h_grad
        total_loss = total_loss.mean()
        # import pdb
        # pdb.set_trace()
        total_loss = total_loss / (self.d * self.d)  # 为了保持和原始代码的一致性，除以 d^2
        return total_loss.mean(), total_grad



