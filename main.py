from torch.utils.data import DataLoader, Subset
import torch, time, os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from src.AM_dataset import AM_Dataset
from src.utils import *
from src.train_and_validate import *
from src.model.PhysGTO import Model
from pathlib import Path

if __name__ == "__main__":
    # Save logs
    PATH = os.path.abspath(os.path.dirname(__file__))
    print(PATH)
    log_path = os.path.join(PATH, "Run_Logs", "new_property.log")
    sys.stdout = Logger(filename=log_path)

    # Dataset 
    np.random.seed(42) # 设置种子 保证可复现
    dataset = AM_Dataset(h5_path=str(Path(r"~/MyAI/AMGTO/H5set/Tiny_mesh_series.h5").expanduser()), fields=['T'])
    N = len(dataset)
    indices, split = np.arange(N), int(0.8 * N)
    train_idx, test_idx = indices[:split], indices[split:]
    print(train_idx, test_idx)

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(3, 7, [16, 15, 5], 10, 4, 1, 1, 128, 4, 128).to(device)
    model.train()
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        start_time = time.time()
        train_error = train(model, train_loader, opt, 'cuda', 8)
        end_time = time.time()

        print(f"=====Epoch {epoch}=====")
        print(f"L2 loss: T {train_error['L2_T']:.4e}")
        print(f"each time step loss: {train_error['each_l2']}")
        print(f"time pre test epoch/s:{end_time - start_time:.2f}")
    print()
    
    # Test
    test_time = time.time()
    model.eval()
    test_error = validate(model, test_loader, 'cuda', 8)
    print(f"========Test========")
    print(f"L2 loss: T {test_error['L2_T']:.4e}")
    print(f"each time step loss: {test_error['each_l2']}")
    print(f"time pre test epoch/s:{time.time() - test_time:.2f}")

    

            

            



    





    
