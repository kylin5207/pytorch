"""
模型保存
"""

import torchvision
import torch
import os

def save_feature_extractor(output_file):
    """
    利用resnet进行特征提取
    :return:
    """
    try:
        # 加载resnet模型
        model_name = "resnet"
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.eval()

        # 将最后一层除外，提取2048特征
        newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))

        # 输入的shape
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)

        # 利用预训练模型
        scripted_model = torch.jit.trace(newmodel, input_data).eval()

        # 保存模型
        torch.jit.save(scripted_model, output_file)
        print(f"Saved success! Model saved as {os.getcwd()}/{output_file}")

    except Exception as e:
        raise ValueError("Failed to save the model.")
