from argparse import ArgumentParser
import torch
from reinforcement_learning import *
import pandas as pd

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
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     fisher[name] += param.grad.data ** 2
        
        grads = get_grads_list(model)
        fisher += grads
    fisher /= len(dataloader.dataset)
    return fisher

def repair_model(pass_test_path, fail_test_path, num_epochs=10, 
                 old_ckpt_path="cartpole_reinforce_weights_attacked_seed_1234.pt",
                 lr=0.001, bz=4, fim_reg=0.0):
    x_pass_loader, x_fail_loader = get_data(pass_test_path, fail_test_path, bz=bz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    net = Policy(s_size=5)
    net.load_state_dict(torch.load(old_ckpt_path))
    net.to(device)
    net.train()
    
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    fim = calc_fish(net, x_pass_loader, nn.CrossEntropyLoss(), device=device)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # num_epochs = 40

    # Merge the two datasets
    # just take a subset of the pass test cases
    # x_pass_loader.dataset = torch.utils.data.Subset(x_pass_loader.dataset, range(0, 100))
    
    sub_pass_subset = torch.utils.data.Subset(x_pass_loader.dataset, range(0, 1000))
    sub_pass_loader = torch.utils.data.DataLoader(sub_pass_subset, batch_size=x_pass_loader.batch_size, shuffle=True)
    
    merged_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([sub_pass_loader.dataset, x_fail_loader.dataset,]),
        batch_size=x_pass_loader.batch_size,
        shuffle=True
    )

    print("Start fine-tuning on merged dataset...")
    prev_vectorized_net = vectorized_model(net)
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
                # for name, param in net.named_parameters():
                #     if name in fim:
                #         loss += fim_reg * (fim[name] * param).sum()
                # penalty = (fim * ((vectorized_model(net) - prev_vectorized_net) ** 2)).sum()
                penalty = (fim * (vectorized_model(net) - prev_vectorized_net)).sum()
                loss += fim_reg * penalty
            
            # print(f"Loss: {loss.item()}| Reg Loss: {penalty.item() if penalty else 0}")
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Training loss: {running_loss:.4f}")
    # Save the repaired model
    # trim the model name to add suffix "_repaired"
    torch.save(net.state_dict(), old_ckpt_path.replace(".pt", "_repaired.pt"))
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
    
    args = args.parse_args()
    
    pass_test_path = "new_passing_cases.csv"
    fail_test_path = "new_failing_cases.csv"
    # Repair the model
    # net = repair_model(pass_test_path, fail_test_path, num_epochs=100, 
    #                    old_ckpt_path="cartpole_reinforce_weights_attacked_seed_1234.pt", 
    #                    lr=0.001, bz=16, fim_reg=1.0)
    
    net = repair_model(args.pass_test_path, args.fail_test_path, num_epochs=args.num_epochs,
                       old_ckpt_path=args.old_ckpt_path, lr=args.lr, bz=args.bz, fim_reg=args.fim_reg)
    # Save the repaired model
    torch.save(net.state_dict(), args.old_ckpt_path.replace(".pt", "_repaired.pt"))
    print("Repaired model saved.")
    
    # Evaluate the repaired model
    eval(net, pass_test_path, fail_test_path, device="cuda" if torch.cuda.is_available() else "cpu")
    del net
    torch.cuda.empty_cache()
    print("Memory cleared.")
