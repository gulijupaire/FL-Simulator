import torch
from utils import *
from client import *
from .server import Server
from utils import get_params_list_with_shape, get_mdl_params, set_mdl_params

# ✅ 新增 MUD/AAD 相关的导入
from optimizer.utils_mud import build_flatten_plan


class FedSMOO(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        # 首先调用父类的 __init__
        super(FedSMOO, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)

        # FedSMOO 特有的状态变量
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print("    Dual Variable List     --->  {:d} * {:d}".format(
            self.h_params_list.shape[0], self.h_params_list.shape[1]))

        self.mu_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print("   Dyn-Dual Variable List  --->  {:d} * {:d}".format(
            self.mu_params_list.shape[0], self.mu_params_list.shape[1]))

        self.global_dynamic_dual = torch.zeros(init_par_list.shape[0])

        # 客户端类的指定
        self.Client = fedsmoo

        # 注意：comm_vecs 的初始化在基类 Server 中完成，这里不需要重新定义
        # 我们只需要在 process_for_communication 中填充它

    def process_for_communication(self, client, Averaged_update):
        # 填充 comm_vecs 字典，准备发送给客户端
        if not self.args.use_RI:
            self.comm_vecs['Params_list'] = self.server_model_params_list.clone().detach()
        else:
            self.comm_vecs['Params_list'] = (self.server_model_params_list + self.args.beta \
                                             * (self.server_model_params_list - self.clients_params_list[
                        client])).clone().detach()

        self.comm_vecs['Local_dual_correction'] = (
                    self.h_params_list[client] - self.comm_vecs['Params_list']).clone().detach()
        self.comm_vecs['Dynamic_dual'] = get_params_list_with_shape(self.server_model, self.mu_params_list[client],
                                                                    self.device)
        self.comm_vecs['Dynamic_dual_correction'] = get_params_list_with_shape(self.server_model,
                                                                               self.global_dynamic_dual, self.device)

        # 附带 MUD/AAD 需要的元数据
        self.comm_vecs['round'] = int(getattr(self, 'current_round', 0))
        if self.args.use_mud and self.args.use_aad:
            self.comm_vecs['aad_seed'] = self.aad_seed

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSMOO 的全局更新逻辑 (保持不变)
        Averaged_dynamic_dual = torch.mean(self.mu_params_list[selected_clients], dim=0)
        _l2_ = torch.norm(Averaged_dynamic_dual, p=2, dim=0) + 1e-7
        self.global_dynamic_dual = Averaged_dynamic_dual / _l2_ * self.args.rho

        return Averaged_model + torch.mean(self.h_params_list, dim=0)

    def postprocess(self, client, received_vecs):
        # =====================================================================
        #  ↓↓↓ 改动 4: 简化 postprocess，只更新 FedSMOO 特有状态 ↓↓↓
        # =====================================================================
        # 步骤 A: 更新 h_params_list (累加本轮的 ΔW)
        # 本轮的 ΔW (delta_vec) 已经在基类 server.py 的训练循环中被正确计算并存储
        delta_vec = self.clients_updated_params_list[client]
        self.h_params_list[client] += delta_vec

        # 步骤 B: 更新 mu_params_list (动态对偶变量)
        if 'local_dynamic_dual' in received_vecs and received_vecs['local_dynamic_dual'] is not None:
            mu = [x.clone().detach().cpu().reshape(-1) for x in received_vecs['local_dynamic_dual']]
            self.mu_params_list[client] = torch.cat(mu)

        # 注意: clients_params_list 和 clients_updated_params_list 的更新已移至基类 Server 的 train 循环中
        # =====================================================================