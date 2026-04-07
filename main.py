import torch
import numpy as np
import os
import math
import time
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

# load specific modules for LPBF project
from src.dataset_fast import AeroGtoDataset
from src.dataset_2d import AeroGtoDataset2D
from src.dataset_cut_fast import CutAeroGtoDataset
# from src.physgto import Model
from src.train import train, validate
from src.utils import set_seed, init_weights, parse_args, load_json_config

def get_dataloader(args, path_record, device_type):
    data_cfg = args.data
    model_cfg = args.model
    space_dim = model_cfg.get("space_size", 3)
    
    if space_dim == 3:
        if data_cfg.get("cut", False):
            Datasetclass = CutAeroGtoDataset
        else:
            Datasetclass = AeroGtoDataset
    elif space_dim == 2:
        Datasetclass = AeroGtoDataset2D
        
    # 构建数据集
    train_dataset = Datasetclass(
        args=args,
        mode="train"
    )

    test_dataset = Datasetclass(
        args=args,
        mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
    )
    
    # 共享 Normalizer
    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()  # 同步 norm_mean/norm_std 缓存

    # 构建 DataLoader
    pin_memory = True if "cuda" in device_type else False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg['train'].get("batchsize", 1),
        shuffle=True,
        num_workers=data_cfg['train'].get("num_workers", 0),
        pin_memory=pin_memory,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg['test'].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    # 记录数据集信息
    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    edge_num = train_dataset.meta_cache[train_dataset.file_paths[0]]["edges"].shape[0]
    
    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Node num: {train_dataset.node_num}, Edge num: {edge_num}, Cond dim: {cond_dim}\n")
        file.write(f"Mean dt: {train_dataset.dt:.4e}\n")

    return train_dataloader, test_dataloader, train_dataset.normalizer, cond_dim, train_dataset.dt

def get_model(args, device, cond_dim, default_dt):
    model_cfg = args.model
    model_name = model_cfg.get("name", "PhysGTO")

    if model_name == "PhysGTO":
        from src.physgto import Model
    elif model_name == "gto_res":
        from src.physgto_res import Model
    elif model_name == "gto_lnn":
        from src.gto_lnn import Model
    elif model_name == "gto_attnres_multi":
        from src.physgto_attnres_multi import Model
    elif model_name == "gto_attnres_multi_v2":
        from src.physgto_attnres_multi_v2 import Model
    elif model_name == "gto_res_attnres":
        from src.physgto_res_attnres import Model
    elif model_name == "v3":
        from src.gto_res_attnres_v3_self import Model

    common_kwargs = dict(
        space_size=model_cfg.get("space_size", 3),
        pos_enc_dim=model_cfg.get("pos_enc_dim", 5),
        cond_dim=cond_dim,
        N_block=model_cfg.get("N_block", 4),
        in_dim=model_cfg.get("in_dim", 4),
        out_dim=model_cfg.get("out_dim", 4),
        enc_dim=model_cfg.get("enc_dim", 128),
        n_head=model_cfg.get("n_head", 4),
        n_token=model_cfg.get("n_token", 64),
        dt=model_cfg.get("dt", default_dt),
        stepper_scheme=model_cfg.get("stepper_scheme", "euler")
    )

    if model_name in ("gto_attnres_multi", "gto_attnres_multi_v2", "gto_res_attnres", "v3"):
        common_kwargs["n_fields"] = model_cfg.get("n_fields", model_cfg.get("in_dim", 2))
        common_kwargs["cross_attn_heads"] = model_cfg.get("cross_attn_heads", 4)

    if model_name in ("gto_attnres_multi_v2", "gto_res_attnres", "v3"):
        common_kwargs["attn_res_mode"] = model_cfg.get("attn_res_mode", "block_inter")

    model = Model(**common_kwargs).to(device)

    load_path = model_cfg.get("load_path")
    checkpoint = None

    if load_path:
        # 如果指定了加载路径
        model_path = os.path.join(load_path, f"{args.name}_best.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
    elif model_cfg.get("if_init", True):
        model.apply(init_weights)
        
    return model, checkpoint

def main(args, path_logs, path_nn, path_record):

    # setting
    device_str = args.device
    if "cuda" in device_str and not torch.cuda.is_available():
        print("! Warning: CUDA not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    
    EPOCH = int(args.train["epoch"])
    real_lr = float(args.train["lr"])
    fields = args.data.get("fields", ["T"]) # 获取物理场列表，例如 ['T', 'Ux', 'Uy', 'Uz']

    # dataloader & normalizer
    train_dataloader, test_dataloader, normalizer, cond_dim, default_dt = get_dataloader(args, path_record, device_str)

    # model
    model, checkpoint = get_model(args, device, cond_dim, default_dt)    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    print(f"EPOCH: {EPOCH}, #params: {params/1e6:.2f}M")      

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Using device: {device}\n")
        file.write(f"{args.name}, #params: {params/1e6:.2f}M\n")
        file.write(f"EPOCH: {EPOCH}\n")
        file.write(f"Fields: {fields}\n")

    # log_dir = f"{path_logs}/{args.name}"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{path_logs}/{args.name}_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
        
    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=real_lr, weight_decay=real_lr/50.0)
    if EPOCH < 10:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr)  
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr/20.0)
   
    start_epoch, best_val_error = 0, float("inf")

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Restored optimizer state.")

        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Restored scheduler state.")

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        
        if 'best_val_error' in checkpoint:
            best_val_error = checkpoint['best_val_error']
            print(f"Restored best val error: {best_val_error:.4e}")

    if start_epoch >= EPOCH:
        print(f"Warning: Start epoch {start_epoch} >= Total EPOCH {EPOCH}. Training may perform 0 steps.")
    
    for epoch in range(start_epoch, EPOCH):
        start_time = time.time()
        # Train
        train_error = train(
            args,
            model,
            train_dataloader,
            optimizer,
            device,
            normalizer
        )
        end_time = time.time()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        training_time = (end_time - start_time)
        
        # 提取 Metrics
        train_loss = train_error['loss']
        train_mean_l2 = train_error['mean_l2']
        each_t_l2 = train_error['each_l2']

        # 动态构建 Log 字符串和 Tensorboard 记录
        log_str = f"Training, Epoch: {epoch + 1}/{EPOCH}, train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}"
        writer.add_scalar('lr/lr', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L2/train_mean_l2', train_mean_l2, epoch)
        
        # 动态记录每个物理场 (Field)
        l2_details = []
        rmse_details = []
        region_l2_details = []
        for fname in fields:
            l2_val = train_error[f"L2_{fname}"]
            rmse_val = train_error[f"RMSE_{fname}"]

            l2_details.append(f"{fname}: {l2_val:.4e}")
            rmse_details.append(f"{fname}: {rmse_val:.4e}")

            writer.add_scalar(f'L2/train_L2_{fname}', l2_val, epoch)
            writer.add_scalar(f'RMSE/train_RMSE_{fname}', rmse_val, epoch)

            # Region metrics per field
            for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                rval = train_error.get(f"{prefix}_{fname}")
                if rval is not None and not math.isnan(rval):
                    writer.add_scalar(f'{prefix}/train_{prefix}_{fname}', rval, epoch)

            a_l2 = train_error.get(f"active_L2_{fname}")
            i_l2 = train_error.get(f"inactive_L2_{fname}")
            if a_l2 is not None:
                region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

        # Region mean metrics
        for key in ("active_mean_l2", "inactive_mean_l2"):
            val = train_error.get(key)
            if val is not None and not math.isnan(val):
                writer.add_scalar(f'L2/train_{key}', val, epoch)
        for key in ("active_loss", "inactive_loss"):
            val = train_error.get(key)
            if val is not None:
                writer.add_scalar(f'Loss/train_{key}', val, epoch)

        print(log_str)
        value_loss, grad_loss = train_error.get("value_loss", 0), train_error.get("grad_loss", 0)
        print(f"value_loss:{value_loss} | grad_loss:{grad_loss}")
        print(f"L2 details: {', '.join(l2_details)}")
        print(f"RMSE details: {', '.join(rmse_details)}")
        if region_l2_details:
            print(f"Region L2: {', '.join(region_l2_details)}")
        print(f"each time step loss: {each_t_l2.tolist()}")
        print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
        print("--------------")

        # 写入文件日志
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"Training, epoch: {epoch + 1}/{EPOCH}\n")
            file.write(f"Train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}\n")
            file.write(f"L2 details: {', '.join(l2_details)}\n")
            file.write(f"RMSE details: {', '.join(rmse_details)}\n")
            if region_l2_details:
                file.write(f"Region L2: {', '.join(region_l2_details)}\n")
            file.write(f"each time step loss: {each_t_l2.tolist()}\n")
            file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
        
        # Validation
        eval_every = args.train.get("eval_every", 5)
        if (epoch+1) % eval_every == 0 or epoch == 0 or (epoch+1) == EPOCH:
            start_time = time.time() 
            test_error = validate(args, model, test_dataloader, device, normalizer, epoch+1)
            end_time = time.time()
            
            val_time = (end_time - start_time)

            test_mean_l2 = test_error['mean_l2']   
            test_each_t_l2 = test_error['each_l2']
            
            # 动态构建 Test Log
            test_l2_details = []
            test_rmse_details = []
            test_region_l2_details = []

            writer.add_scalar('L2/test_mean_l2', test_mean_l2, epoch)

            for fname in fields:
                l2_val = test_error[f"L2_{fname}"]
                rmse_val = test_error[f"RMSE_{fname}"]

                test_l2_details.append(f"{fname}: {l2_val:.4e}")
                test_rmse_details.append(f"{fname}: {rmse_val:.4e}")

                writer.add_scalar(f'L2/test_L2_{fname}', l2_val, epoch)
                writer.add_scalar(f'RMSE/test_RMSE_{fname}', rmse_val, epoch)

                # Region metrics per field
                for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                    rval = test_error.get(f"{prefix}_{fname}")
                    if rval is not None and not math.isnan(rval):
                        writer.add_scalar(f'{prefix}/test_{prefix}_{fname}', rval, epoch)

                a_l2 = test_error.get(f"active_L2_{fname}")
                i_l2 = test_error.get(f"inactive_L2_{fname}")
                if a_l2 is not None:
                    test_region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

            # Region mean metrics
            for key in ("active_mean_l2", "inactive_mean_l2"):
                val = test_error.get(key)
                if val is not None and not math.isnan(val):
                    writer.add_scalar(f'L2/test_{key}', val, epoch)

            print("---Inference---")
            print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 details: {', '.join(test_l2_details)}")
            print(f"RMSE details: {', '.join(test_rmse_details)}")
            if test_region_l2_details:
                print(f"Region L2: {', '.join(test_region_l2_details)}")
            print(f"each time step loss: {test_each_t_l2.tolist()}")
            print(f"time pre test epoch/s:{val_time:.2f}")
            print("--------------")

            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"Inference, epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"L2 details: {', '.join(test_l2_details)}\n")
                file.write(f"RMSE details: {', '.join(test_rmse_details)}\n")
                if test_region_l2_details:
                    file.write(f"Region L2: {', '.join(test_region_l2_details)}\n")
                file.write(f"each time step loss: {test_each_t_l2.tolist()}\n")
                file.write(f"time pre test epoch/s:{val_time:.2f}\n")
            
            # Save Best
            if args.if_save and test_mean_l2 < best_val_error:
                best_val_error = test_mean_l2
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': args # 保存配置方便复现
                }
                torch.save(checkpoint, f"{path_nn}/{args.name}_best.pt")

        # Regular Save
        if (epoch+1) % 50 == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': current_lr,
                }
                torch.save(checkpoint, f"{path_nn}/{args.name}_{epoch+1}.pt")

    writer.close()

if __name__ == "__main__":
    # 解析参数 (使用 LPBF 项目的 json config 方式)
    cli_args = parse_args()
    args = load_json_config(cli_args.config)
    
    # 打印参数
    print(args)
    
    # save path
    path_logs = args.save_path + "/logs"
    path_nn = args.save_path + "/nn"
    path_record = args.save_path + "/record"

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_nn, exist_ok=True)
    os.makedirs(path_record, exist_ok=True)

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"{'='*20} Start {'='*20}\n")
        file.write(str(args) + "\n")
        file.write(f"Config file: {cli_args.config}\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    if args.seed is not None:
        set_seed(args.seed)
    
    main(args, path_logs, path_nn, path_record)
    
    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")