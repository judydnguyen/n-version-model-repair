from argparse import ArgumentParser
import copy
import torch
from reinforcement_learning import *
import pandas as pd
from itertools import cycle

def get_data(pass_test_path, fail_test_path, bz=4):
    pass_test_cases = pd.read_csv(pass_test_path)
    fail_test_cases = pd.read_csv(fail_test_path)

    y_pass = pass_test_cases['actuation']
    x_pass = pass_test_cases.iloc[:, 1:6]

    y_fail = fail_test_cases['actuation']
    x_fail = fail_test_cases.iloc[:, 1:6]

    x_pass = np.array(x_pass)
    y_pass = np.array(y_pass)
    x_fail = np.array(x_fail)
    y_fail = np.array(y_fail)

    x_pass = torch.tensor(x_pass, dtype=torch.float32)
    y_pass = torch.tensor(y_pass, dtype=torch.float32)
    x_fail = torch.tensor(x_fail, dtype=torch.float32)
    y_fail = torch.tensor(y_fail, dtype=torch.float32)

    print(f"Shape of x_pass: {x_pass.shape}")
    print(f"Shape of y_pass: {y_pass.shape}")
    print(f"Shape of x_fail: {x_fail.shape}")
    print(f"Shape of y_fail: {y_fail.shape}")
    
    pass_dataset = torch.utils.data.TensorDataset(x_pass, y_pass)
    fail_dataset = torch.utils.data.TensorDataset(x_fail, y_fail)

    x_pass_loader = torch.utils.data.DataLoader(pass_dataset, batch_size=bz, shuffle=True)
    x_fail_loader = torch.utils.data.DataLoader(fail_dataset, batch_size=bz, shuffle=True)
    
    return x_pass_loader, x_fail_loader

def vectorized_model(model):
    """
    Returns all the parameters concatenated in a single tensor.
    :return: parameters tensor (??)
    """
    params = []
    for pp in list(model.parameters()):
        params.append(pp.view(-1))
    return torch.cat(params)

def set_vectorized_model(model, vector):
    """
    Set the model parameters to the vectorized parameters.
    :param model: model
    :param vector: vectorized parameters
    """
    # print(vector.shape)
    # print(vectorized_model(model).shape)
    assert vector.shape == vectorized_model(model).shape, "Shape mismatch"
    start = 0
    for pp in list(model.parameters()):
        end = start + pp.numel()
        pp.data.copy_(vector[start:end].view_as(pp))
        start = end

def get_grads_list(model):
    """
    Returns a list containing the gradients (a tensor for each layer).
    :return: gradients list
    """
    grads = []
    for pp in list(model.parameters()):
        grads.append(pp.grad.view(-1))
    return torch.cat(grads)
    
def calc_fish(model, dataloader, criterion, device):
    # Calculate the Fisher information matrix
    fisher = torch.zeros_like(vectorized_model(model))
    for inputs, labels in dataloader:
        
        inputs, labels = inputs.to(device), labels.to(device).long()
        outputs = model(inputs).logits
        loss = -criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     fisher[name] += param.grad.data ** 2
    
        grads = get_grads_list(model)
        fisher += grads
    fisher /= len(dataloader.dataset)
    return fisher

def train_dual_loss(failed_loader, passed_loader, net, old_net, optimizer, criterion, num_epochs):
    # from itertools import cycle
    for epoch in range(num_epochs):
        running_loss = 0.0
        repair_loss_total = 0.0
        retain_loss_total = 0.0

        # Cycle through passed_loader if it's smaller than failed_loader
        for (failed_inputs, failed_labels), (passed_inputs, _) in zip(failed_loader, cycle(passed_loader)):
            failed_inputs, failed_labels = failed_inputs.to(device), failed_labels.to(device).long()
            passed_inputs = passed_inputs.to(device)

            optimizer.zero_grad()
            
            # ---- REPAIR LOSS (on failed cases) ----
            failed_outputs = net(failed_inputs)
            repair_logits = failed_outputs.logits
            repair_loss = criterion(repair_logits, failed_labels).mean()

            # ---- RETAIN LOSS (KL divergence on passed cases) ----
            with torch.no_grad():
                original_outputs = old_net(passed_inputs).logits.detach()
            current_outputs = net(passed_inputs).logits

            retain_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(current_outputs, dim=-1),
                torch.nn.functional.softmax(original_outputs, dim=-1),
                reduction="batchmean"
            )

            # ---- Combine losses ----
            loss = 0.5* repair_loss + 0.5* retain_loss

            # # ---- Optional: FIM regularization ----
            # if fim_reg > 0:
            #     penalty = (fim * (vectorized_model(net) - prev_vectorized_net)).sum()
            #     loss += fim_reg * penalty

            # ---- Optimize ----
            loss.backward()
            optimizer.step()

            # ---- Logging ----
            running_loss += loss.item()
            repair_loss_total += repair_loss.item()
            retain_loss_total += retain_loss.item()

        print(f"[Epoch {epoch+1}] Total Loss: {running_loss:.4f} | Repair: {repair_loss_total:.4f} | Retain: {retain_loss_total:.4f}")

def train_fim(merged_loader, net, prev_vectorized_net, fim, optimizer, criterion, num_epochs, fim_reg=0.0):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in merged_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = net(inputs)
            logits = outputs.logits  # assuming net returns a Categorical distribution
            loss = criterion(logits, labels).mean()
            
            # Add FIM regularization
            reg_loss = 0
            penalty = 0
            if fim_reg > 0:
                penalty = (fim * (vectorized_model(net) - prev_vectorized_net)).sum()
                loss += fim_reg * penalty
            
            # print(f"Loss: {loss.item()}| Reg Loss: {penalty.item() if penalty else 0}")
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch + 1}] Training loss: {running_loss:.4f}")
        
def add_masked_noise(model, fim, k=0.1):
    # Add noise to the model parameters based on the Fisher information matrix
    vectorized_m = vectorized_model(model)
    # Get bottom-K indices of the Fisher information matrix
    _, indices = torch.topk(fim, int(k*len(fim)), largest=False)
    # Create a mask for the parameters
    mask = torch.zeros_like(vectorized_m)
    mask[indices] = 1
    # Add noise to the parameters, limited by epsilon
    epsilon = 0.01
    noise = torch.randn_like(vectorized_m) * k * epsilon * mask
    # noise = torch.randn_like(vectorized_m) * * mask
    # Update the model parameters
    vectorized_m += noise
    # Set the model parameters
    set_vectorized_model(model, vectorized_m)
    return model
    
def unlearn_and_finetune(model, x_pass_loader, x_fail_loader, 
                         optimizer, criterion, num_epochs=10):
    # Unlearn the fail test cases
    unlearn_optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in x_fail_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            labels = 1.0 - labels
            labels = labels.long()
            unlearn_optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits  # assuming net returns a Categorical distribution
            loss = -criterion(logits, labels).mean()
            loss.backward()
            unlearn_optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch + 1}] Unlearning loss: {running_loss:.4f}")

    model.train()  # set model to train mode
    finetune_optimizer = optim.Adam(model.parameters(), lr=0.002)
    # Fine-tune on the pass test cases
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in x_pass_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            finetune_optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits  # assuming net returns a Categorical distribution
            loss = criterion(logits, labels).mean()
            loss.backward()
            finetune_optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch + 1}] Fine-tuning loss: {running_loss:.4f}")

def repair_model(pass_test_path, fail_test_path, 
                 data_mode="fail_only",
                 num_epochs=10, fail_ratio = 0.5, 
                 old_ckpt_path="cartpole_reinforce_weights_attacked_seed_1234.pt",
                 lr=0.001, bz=4, 
                 fim_reg=0.0,
                 repair_mode="fail_only"):
    x_pass_loader, x_fail_loader = get_data(pass_test_path, fail_test_path, bz=bz)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # Load the model
    net = Policy(s_size=5)
    net.load_state_dict(torch.load(old_ckpt_path))
    net.to(device)
    net.train()
    
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    fim = calc_fish(net, x_pass_loader, nn.CrossEntropyLoss(), device=device)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    if repair_mode == "fail_only":
        data_mode = "fail_only"
    
    if data_mode == "fail_only":
        # Use only the fail test cases
        merged_loader = x_fail_loader
    elif data_mode == "mixed":
        assert fail_ratio > 0 and fail_ratio < 1, "mixed_ratio should be between 0 and 1"
        # Use both pass and fail test cases
        # mixed_ratio = [fail_ratio]/[pass_ratio + fail_ratio]
        
        # decide which loaders are bigger
        sub_fail_loader = x_fail_loader
        sub_pass_loader = x_pass_loader
        
        if len(x_pass_loader.dataset) > len(x_fail_loader.dataset):
            # Use pass test cases
            sub_pass_subset = torch.utils.data.Subset(x_pass_loader.dataset, range(0, int(len(x_pass_loader.dataset) * fail_ratio)))
            sub_pass_loader = torch.utils.data.DataLoader(sub_pass_subset, batch_size=x_pass_loader.batch_size, shuffle=True)
        else:
            # Use fail test cases
            sub_fail_subset = torch.utils.data.Subset(x_fail_loader.dataset, range(0, int(len(x_fail_loader.dataset) * fail_ratio)))
            sub_fail_loader = torch.utils.data.DataLoader(sub_fail_subset, batch_size=x_fail_loader.batch_size, shuffle=True)
        
        # Merge the pass and fail test cases
        merged_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([sub_pass_loader.dataset, sub_fail_loader.dataset]),
            batch_size=x_pass_loader.batch_size,
            shuffle=True
        )
    else:
        raise ValueError("data_mode should be either 'fail_only' or 'mixed'")

    print("Start fine-tuning on merged dataset...")
    prev_vectorized_net = vectorized_model(net)

    original_net = copy.deepcopy(net)
    
    if repair_mode == "fail_only" or repair_mode == "mixed":
        # Train with dual loss
        train_fim(merged_loader, net, prev_vectorized_net, fim, optimizer, criterion,
                  num_epochs=num_epochs, fim_reg=0.0)
    elif repair_mode == "fim":
        # Train with FIM regularization
        train_fim(merged_loader, net, prev_vectorized_net, fim, optimizer, criterion,
                  num_epochs=num_epochs, fim_reg=fim_reg)
    elif repair_mode == "masked":
        net = add_masked_noise(net, fim, k=0.01)
        train_fim(merged_loader, net, prev_vectorized_net, fim, optimizer, criterion,
                  num_epochs=num_epochs, fim_reg=0.0)
    elif repair_mode == "unlearn":
        unlearn_and_finetune(net, merged_loader, x_fail_loader, optimizer, criterion, num_epochs=num_epochs)
    else:
        raise ValueError("repair_mode should be either 'fail_only', 'mixed', 'fim', 'masked' or 'unlearn'")
    # train_dual_loss(x_fail_loader, x_pass_loader, net, original_net, optimizer, criterion, 
    #                 num_epochs=num_epochs)
    
    # net = add_masked_noise(net, fim, k=0.01)
    
    # train_fim(merged_loader, net, prev_vectorized_net, fim, optimizer, criterion,
    #           num_epochs=num_epochs, fim_reg=fim_reg)
    
    # train_dual_kl_loss(x_fail_loader, x_pass_loader, net, original_net, optimizer,
    #                    criterion, num_epochs=num_epochs)
    
    # unlearn_and_finetune(net, merged_loader, x_fail_loader, optimizer, criterion, num_epochs=num_epochs)
    
    # for epoch in range(num_epochs):
    #     running_loss = 0.0

    #     for inputs, labels in merged_loader:
    #         inputs, labels = inputs.to(device), labels.to(device).long()

    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         logits = outputs.logits  # assuming net returns a Categorical distribution
    #         loss = criterion(logits, labels).mean()
            
    #         # Add FIM regularization
    #         reg_loss = 0
    #         penalty = 0
    #         if fim_reg > 0:
    #             penalty = (fim * (vectorized_model(net) - prev_vectorized_net)).sum()
    #             loss += fim_reg * penalty
            
    #         # print(f"Loss: {loss.item()}| Reg Loss: {penalty.item() if penalty else 0}")
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     print(f"[Epoch {epoch + 1}] Training loss: {running_loss:.4f}")
    # # Save the repaired model
    # # trim the model name to add suffix "_repaired"
    # torch.save(net.state_dict(), old_ckpt_path.replace(".pt", "_repaired.pt"))
    return net

def eval(net, pass_test_path, fail_test_path, device="cpu"):
    # Evaluate on pass test cases
    x_pass_loader, x_fail_loader = get_data(pass_test_path, fail_test_path)
    correct = 0
    total = 0
    net.eval()  # set model to eval mode
    with torch.no_grad():
        for inputs, labels in x_pass_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            
            # Assuming outputs is a Categorical distribution
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
    print(f'Accuracy of the network on the pass test cases: {100 * correct / total:.2f}%')

    # Evaluate on fail test cases
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in x_fail_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            
            # Assuming outputs is a Categorical distribution
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
    print(f'Accuracy of the network on the fail test cases: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pass_test_path", type=str, default="new_passing_cases.csv")
    args.add_argument("--fail_test_path", type=str, default="new_failing_cases.csv")
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--old_ckpt_path", type=str, default="cartpole_reinforce_weights_attacked_seed_1234.pt")
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--bz", type=int, default=16)
    args.add_argument("--fim_reg", type=float, default=1.0)
    args.add_argument("--data_mode", type=str, default="mixed", 
                      choices=["fail_only", "mixed"], help="data mode: 'fail_only' or 'mixed'")
    args.add_argument("--mixed_ratio", type=float, default=0.5,
                      help="ratio of fail test cases to pass test cases in mixed mode")
    args.add_argument("--repair_mode", type=str, default="fail_only",
                      choices=["fail_only", "mixed", "fim", "masked", "unlearn"],
                      help="repair mode: 'fail_only', 'mixed', 'fim', 'masked' or 'unlearn'")
    
    args = args.parse_args()
    
    pass_test_path = "new_passing_cases.csv"
    fail_test_path = "new_failing_cases.csv"
    # Repair the model
    # net = repair_model(pass_test_path, fail_test_path, num_epochs=100, 
    #                    old_ckpt_path="cartpole_reinforce_weights_attacked_seed_1234.pt", 
    #                    lr=0.001, bz=16, fim_reg=1.0)
    
    net = repair_model(args.pass_test_path, args.fail_test_path, num_epochs=args.num_epochs,
                       old_ckpt_path=args.old_ckpt_path, lr=args.lr, bz=args.bz, 
                       fim_reg=args.fim_reg, data_mode=args.data_mode, 
                       fail_ratio=args.mixed_ratio, repair_mode=args.repair_mode)
    # Save the repaired model
    torch.save(net.state_dict(), args.old_ckpt_path.replace(".pt", f"_repaired_mode_{args.repair_mode}.pt"))
    print("Repaired model saved.")
    
    # Evaluate the repaired model
    eval(net, pass_test_path, fail_test_path, device="cuda" if torch.cuda.is_available() else "cpu")
    del net
    torch.cuda.empty_cache()
    print("Memory cleared.")
