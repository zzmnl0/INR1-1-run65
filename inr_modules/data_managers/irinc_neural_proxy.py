import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SIREN Layer (The Core for Derivatives)
# ==========================================
class SineLayer(nn.Module):
    """
    See Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                k = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-k, k)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# ==========================================
# 2. IRI Neural Proxy Class
# ==========================================
class IRINeuralProxy(nn.Module):
    def __init__(self, layers=[4, 64, 64, 64, 1], bounds=None):
        """
        Args:
            layers: List of ints defining architecture. Input dim must be 4 (Lat, Lon, Alt, Time).
            bounds: Dict containing min/max for normalization. 
        """
        super().__init__()
        
        self.net = nn.Sequential()
        
        # Build SIREN Network
        for i in range(len(layers) - 2):
            self.net.add_module(f'sine_layer_{i}', 
                                SineLayer(layers[i], layers[i+1], 
                                          is_first=(i==0), omega_0=30))
            
        # Final layer is Linear
        self.final_linear = nn.Linear(layers[-2], layers[-1])
        
        # Initialize Bounds
        if bounds is None:
            self.bounds = {
                'lat': (-90.0, 90.0),
                'lon': (-180.0, 180.0),
                'alt': (120.0, 500.0),
                'time': (0.0, 720.0)  # 0h - 720h (30天)
            }
        else:
            self.bounds = bounds

        # Buffer for normalization parameters
        self.register_buffer('lat_min', torch.tensor(self.bounds['lat'][0]))
        self.register_buffer('lat_scale', torch.tensor(self.bounds['lat'][1] - self.bounds['lat'][0]))
        self.register_buffer('lon_min', torch.tensor(self.bounds['lon'][0]))
        self.register_buffer('lon_scale', torch.tensor(self.bounds['lon'][1] - self.bounds['lon'][0]))
        self.register_buffer('alt_min', torch.tensor(self.bounds['alt'][0]))
        self.register_buffer('alt_scale', torch.tensor(self.bounds['alt'][1] - self.bounds['alt'][0]))
        self.register_buffer('time_min', torch.tensor(self.bounds['time'][0]))
        self.register_buffer('time_scale', torch.tensor(self.bounds['time'][1] - self.bounds['time'][0]))

    def normalize_inputs(self, x):
        """
        Normalize inputs to [-1, 1].
        Input x: [Batch, 4] -> (Lat, Lon, Alt, Time)
        """
        x_norm = torch.zeros_like(x)
        x_norm[:, 0] = 2 * (x[:, 0] - self.lat_min) / self.lat_scale - 1
        x_norm[:, 1] = 2 * (x[:, 1] - self.lon_min) / self.lon_scale - 1
        x_norm[:, 2] = 2 * (x[:, 2] - self.alt_min) / self.alt_scale - 1
        x_norm[:, 3] = 2 * (x[:, 3] - self.time_min) / self.time_scale - 1
        return x_norm

    def forward(self, x, return_features=False):
        """
        Args:
            x: [Batch, 4] (Lat, Lon, Alt, Time)
            return_features: if True, also return last hidden state [Batch, hidden_dim]
        Returns:
            Ne_log10: [Batch, 1]
            h: [Batch, hidden_dim] — last SIREN hidden state (only when return_features=True)
        """
        x_norm = self.normalize_inputs(x)
        h = self.net(x_norm)          # [Batch, hidden_dim]
        out = self.final_linear(h)    # [Batch, 1]
        if return_features:
            return out, h
        return out

    def freeze(self):
        """Freezes parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def fit_to_grid(self, coords_tensor, values_tensor, epochs=100, batch_size=65536, lr=1e-3, device='cpu', 
                    patience=15, min_delta=1e-5):
        """
        CPU极速训练版 (动态LR+早停机制)
        """
        import copy
        
        if device == 'cpu':
            num_threads = 4 
            torch.set_num_threads(num_threads)
            print(f"使用 {num_threads} 线程并行计算")

        self.to(device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        loss_fn = nn.MSELoss()
        total_samples = coords_tensor.shape[0]
        
        coverage_ratio = 0.8
        steps_per_epoch = int((total_samples * coverage_ratio) // batch_size)
        if steps_per_epoch < 1: steps_per_epoch = 1
        
        print(f"Starting Training: {total_samples} points | {steps_per_epoch} steps/epoch (Batch={batch_size})")
        print(f"Config: Early Stopping (Patience={patience}), Dynamic LR (Patience=3)")
        
        best_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for step in range(steps_per_epoch):
                indices = torch.randint(0, total_samples, (batch_size,))
                batch_coords = coords_tensor[indices]
                batch_vals = values_tensor[indices]
                
                optimizer.zero_grad()
                preds = self.forward(batch_coords)
                loss = loss_fn(preds, batch_vals)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
                if (step + 1) % 2000 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{steps_per_epoch}] Loss: {loss.item():.5f}")
            
            avg_loss = epoch_loss / steps_per_epoch
            current_lr = optimizer.param_groups[0]['lr']
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                early_stop_counter = 0
                best_model_state = copy.deepcopy(self.state_dict())
                print(f"Epoch [{epoch+1}/{epochs}] Finished | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.1e} (*) Best")
            else:
                early_stop_counter += 1
                print(f"Epoch [{epoch+1}/{epochs}] Finished | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.1e} | Patience: {early_stop_counter}/{patience}")
                
                if early_stop_counter >= patience:
                    print(f"\n[Early Stopping] Loss 未下降，停止训练。最佳 Loss: {best_loss:.6f}")
                    break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print("已恢复至最佳模型参数。")
            
        print("Training Complete. Proxy is ready.")
        self.eval()


# ==========================================
# 3. NPY数据加载函数
# ==========================================
def load_dataset_from_npy(npy_path, time_step=3):
    """
    从预处理的NPY文件加载数据，构建神经场训练集。
    
    NPY文件格式: (241, 14, 181, 360) - (时间帧, 高度, 纬度, 经度)
    时间范围: 0h (2024-09-01 00:00) -> 720h (2024-10-01 00:00)
    
    Args:
        npy_path: NPY文件路径
        time_step: 时间步长（小时），默认3h
        
    Returns:
        inputs_tensor: [N, 4] - (Lat, Lon, Alt, Time)
        targets_tensor: [N, 1] - Log10电子密度
    """
    print(f"正在加载NPY数据: {npy_path}")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY文件不存在: {npy_path}")
    
    # 加载数据 - 形状: (241, 14, 181, 360) 即 (Time, Alt, Lat, Lon)
    data_tensor = np.load(npy_path)
    print(f"NPY数据形状: {data_tensor.shape}")
    
    n_time, n_alt, n_lat, n_lon = data_tensor.shape
    
    # ================= 定义坐标网格 =================
    # 根据IRI标准网格定义
    # 纬度: -90 到 90, 共181点 (1度分辨率)
    lats = np.linspace(-90, 90, n_lat).astype(np.float32)
    
    # 经度: -180 到 179, 共360点 (1度分辨率)
    lons = np.linspace(-180, 179, n_lon).astype(np.float32)
    
    # 高度: 根据IRI模型常用高度层 (需根据实际数据调整)
    # 假设14层高度，从100km到某个最大值
    # 常见IRI高度: 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 800 km
    alts = np.array([100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 800], 
                    dtype=np.float32)
    
    # 如果高度层数不匹配，使用线性插值
    if len(alts) != n_alt:
        print(f"警告: 预定义高度层数({len(alts)})与数据不符({n_alt})，使用线性插值")
        alts = np.linspace(100, 800, n_alt).astype(np.float32)
    
    # 时间: 0h 到 720h, 每3小时一帧, 共241帧
    # time_hours[i] = i * time_step
    time_hours = np.arange(n_time) * time_step  # [0, 3, 6, ..., 720]
    
    print(f"坐标范围:")
    print(f"  纬度: {lats.min()} ~ {lats.max()} ({n_lat}点)")
    print(f"  经度: {lons.min()} ~ {lons.max()} ({n_lon}点)")
    print(f"  高度: {alts.min()} ~ {alts.max()} ({n_alt}点)")
    print(f"  时间: {time_hours.min()} ~ {time_hours.max()} ({n_time}帧)")
    
    # ================= 构建训练数据 =================
    print("正在构建训练数据集...")
    
    input_list = []
    target_list = []
    
    # 预先创建空间网格 (Alt, Lat, Lon) -> 用于每个时间步
    A, La, Lo = np.meshgrid(alts, lats, lons, indexing='ij')  # (14, 181, 360)
    spatial_flat = np.stack([La.flatten(), Lo.flatten(), A.flatten()], axis=1)  # [N_spatial, 3]
    n_spatial = spatial_flat.shape[0]
    
    print(f"每帧空间点数: {n_spatial}")
    
    for t_idx in range(n_time):
        # 当前时间帧的数据 - (14, 181, 360)
        frame_data = data_tensor[t_idx]
        
        # 展平为 (N_spatial,) - 注意顺序要与meshgrid一致
        # meshgrid indexing='ij' 生成 (alt, lat, lon) 顺序
        log_ne = frame_data.flatten()
        
        # 当前时间（小时）
        current_time = time_hours[t_idx]
        time_col = np.full((n_spatial, 1), current_time, dtype=np.float32)
        
        # 拼接: [Lat, Lon, Alt, Time]
        current_inputs = np.hstack([spatial_flat, time_col])
        
        input_list.append(current_inputs)
        target_list.append(log_ne)
        
        if (t_idx + 1) % 50 == 0:
            print(f"  已处理 {t_idx + 1}/{n_time} 帧...")
    
    # 合并所有数据
    print("正在合并Tensor...")
    inputs_np = np.concatenate(input_list, axis=0)
    targets_np = np.concatenate(target_list, axis=0)
    
    inputs_tensor = torch.from_numpy(inputs_np).float()
    targets_tensor = torch.from_numpy(targets_np).float().unsqueeze(1)
    
    print(f"数据集构建完成:")
    print(f"  输入形状: {inputs_tensor.shape}")  # [N_total, 4]
    print(f"  目标形状: {targets_tensor.shape}")  # [N_total, 1]
    print(f"  总样本数: {inputs_tensor.shape[0]:,}")
    
    return inputs_tensor, targets_tensor, {
        'lats': lats,
        'lons': lons,
        'alts': alts,
        'time_hours': time_hours
    }


# ==========================================
# 4. 主函数
# ==========================================
def main():
    # ================= 配置参数 =================
    NPY_PATH = r"D:\code11\IRI01\output_results\iri_processed_tensor.npy"
    MODEL_SAVE_PATH = r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth"
    VIZ_SAVE_PATH = r"D:\code11\IRI01\output_results\IRI_Interpolation_Test_0100.png"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")
    
    EPOCHS = 150
    BATCH_SIZE = 4096
    LR = 1e-4
    TIME_STEP = 3  # 3小时分辨率
    
    # ================= 1. 加载NPY数据集 =================
    inputs_tensor, targets_tensor, coords_info = load_dataset_from_npy(NPY_PATH, time_step=TIME_STEP)
    
    # 动态计算边界
    lat_min, lat_max = inputs_tensor[:, 0].min().item(), inputs_tensor[:, 0].max().item()
    lon_min, lon_max = inputs_tensor[:, 1].min().item(), inputs_tensor[:, 1].max().item()
    alt_min, alt_max = inputs_tensor[:, 2].min().item(), inputs_tensor[:, 2].max().item()
    time_min, time_max = inputs_tensor[:, 3].min().item(), inputs_tensor[:, 3].max().item()

    bounds = {
        'lat': (lat_min, lat_max),
        'lon': (lon_min, lon_max),
        'alt': (alt_min, alt_max),
        'time': (time_min, time_max) 
    }
    print(f"模型归一化边界: {bounds}")

    # ================= 2. 初始化与训练 =================
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1], bounds=bounds)
    proxy.to(DEVICE)

    proxy.fit_to_grid(inputs_tensor, targets_tensor, 
                      epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE)
    
    # 保存模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(proxy.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存: {MODEL_SAVE_PATH}")

    # ================= 3. 可视化插值效果检验 =================
    # 目标：绘制 2024-09-01 01:00 (即 Time = 1.0h)
    # 训练数据只有 00:00 (Time=0) 和 03:00 (Time=3)，此处检验插值
    
    TEST_TIME_HOUR = 1.0 
    print(f"生成插值验证图 (Time={TEST_TIME_HOUR}h, 对应 9月1日 01:00)...")
    proxy.eval()
    
    # 设置绘图网格
    viz_lats = np.linspace(lat_min, lat_max, 181)
    viz_lons = np.linspace(lon_min, lon_max, 360)
    G_lat, G_lon = np.meshgrid(viz_lats, viz_lons, indexing='ij')
    flat_G_lat = G_lat.flatten()
    flat_G_lon = G_lon.flatten()
    
    # 设定可视化高度层
    target_alts = [180, 270, 360, 450]
    
    # 先进行批量预测，获取数据的全局 Min/Max 以统一色标
    pred_results = []
    
    print("正在计算所有高度层数据以确定色标范围...")
    with torch.no_grad():
        for alt in target_alts:
            flat_G_alt = np.full_like(flat_G_lat, alt)
            flat_G_time = np.full_like(flat_G_lat, TEST_TIME_HOUR)
            
            viz_inputs = np.stack([flat_G_lat, flat_G_lon, flat_G_alt, flat_G_time], axis=1)
            viz_tensor = torch.tensor(viz_inputs, dtype=torch.float32).to(DEVICE)
            
            pred_log_ne = proxy(viz_tensor).cpu().numpy().reshape(181, 360)
            pred_results.append(pred_log_ne)

    # 自动计算 min 和 max
    all_preds = np.array(pred_results)
    auto_vmin = np.min(all_preds)
    auto_vmax = np.max(all_preds)
    print(f"自动色标范围: vmin={auto_vmin:.2f}, vmax={auto_vmax:.2f}")

    # 统一绘图
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    im = None
    for i, (alt, pred_data) in enumerate(zip(target_alts, pred_results)):
        ax = axes[i]
        im = ax.pcolormesh(viz_lons, viz_lats, pred_data, cmap='plasma', shading='auto', 
                           vmin=auto_vmin, vmax=auto_vmax)
        
        ax.set_title(f"Neural Proxy Interpolation\nAlt={alt}km, Time={TEST_TIME_HOUR:.1f}h")
        ax.set_xlabel("Longitude")
        if i == 0: ax.set_ylabel("Latitude")

    # 共享 Colorbar
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), label='Log10 Electron Density')
    plt.suptitle(f"IRI-2024 Neural Surrogate Field Interpolation Test (t={TEST_TIME_HOUR}h)", fontsize=16)
    
    os.makedirs(os.path.dirname(VIZ_SAVE_PATH), exist_ok=True)
    plt.savefig(VIZ_SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"可视化图已保存: {VIZ_SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
