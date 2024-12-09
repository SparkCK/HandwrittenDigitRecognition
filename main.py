import torch.nn as nn
import torch.optim as optim
from HandwrittenDigitRecognition import *
import datetime,os


from torch.utils.tensorboard import SummaryWriter

EPOCHS = 5
SAVE_MODEL = True
SAVE_MODEL_DIR = "./checkpoints"
root_dir = "./data"

TRAIN_MODE = False


def main():

    # hwdr_dataset = HwdrDataset("./data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 定义模型
    model = HwdrCNN()

    if not TRAIN_MODE:
        # 加载模型
        para_path = os.path.join(SAVE_MODEL_DIR,os.listdir(SAVE_MODEL_DIR)[-1])
        model.load_state_dict(torch.load(para_path))

    print(model)

    model = model.to(DEVICE)

    if TRAIN_MODE:
        writer = SummaryWriter("logs/train")
        criterion = nn.CrossEntropyLoss()  # 定义损失函数
        optimizer = optim.Adam(model.parameters())  # 定义优化器
        # 训练模型
        train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS,writer=writer)
        writer.close()

    if SAVE_MODEL:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(SAVE_MODEL_DIR):
            os.makedirs(SAVE_MODEL_DIR)
        model_file_name = f"model_HwdrCNN_{current_time}_epochs{EPOCHS}.pth"
        model_file_path =os.path.join(SAVE_MODEL_DIR ,model_file_name)
        torch.save(model.state_dict(), model_file_path)


    # 评估模型
    evaluate_model(model, test_loader)



if __name__ == '__main__':
    main()
