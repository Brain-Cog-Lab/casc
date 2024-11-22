from src.dataset import get_imagenet_data
from tqdm import tqdm
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_loader = get_imagenet_data('/data/datasets', 128, False, 32)

    # 加载模型并移到指定设备
    model = torchvision.models.efficientnet_b0(pretrained=True)
    print(model)

    # 评估模型
    # accuracy = evaluate_model(model, test_loader, device)
    # print(f'Test Accuracy: {accuracy:.2f}%')
