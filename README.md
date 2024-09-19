# Handwritten Transformer Framework

[中文说明](README.zh.md)

A handwritten Transformer framework implemented in PyTorch, including all components of the Transformer model and providing simple training examples. This project aims to offer a clear and understandable codebase for learning and researching the Transformer model, with continuous updates and maintenance.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Output](#example-output)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a handwritten Transformer framework with detailed implementation and training code. The main goal is to deeply understand the internal mechanisms of the Transformer model and provide a foundation for future feature expansion and performance optimization.

## Features

- **Complete Transformer Model Implementation**
  - Includes multi-head attention, feed-forward networks, encoder and decoder layers, etc.
- **Modular Design**
  - Code is organized by functional modules for ease of understanding and extension.
- **Custom Training Workflow**
  - Includes custom optimizer and learning rate scheduler, label smoothing, and more.
- **Continuous Updates and Maintenance**
  - Plans to add more features like pre-trained model loading, more complex task examples, etc.

## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- NumPy

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Handwritten-Transformer.git
```

### 2. Navigate to the project directory

```bash
cd Handwritten-Transformer
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

*Ensure you have a `requirements.txt` file in your project directory with the following content:*

```
torch
numpy
```

## Usage

### Running the Training Script

To train the Transformer model on a simple copy task, run:

```bash
python train.py
```

### Expected Output

The script will output training progress, including loss values and tokens processed per second:

```
Epoch Step: 1 Loss: 2.477593 Tokens per Sec: 2490.268555
Epoch Step: 2 Loss: 2.117399 Tokens per Sec: 3856.200928
...
```

## Project Structure

```
Handwritten-Transformer/
├── Transformer.py   # Model definition
├── train.py         # Training script
├── README.md        # Project documentation (English)
├── README.zh.md     # 项目文档 (中文)
├── requirements.txt # Package requirements
└── LICENSE          # License file
```

### Files Description

- **Transformer.py**
  - Contains the implementation of the Transformer model, including all sub-modules like multi-head attention, feed-forward networks, encoder, decoder, etc.
- **train.py**
  - Includes the training loop, data generation for the copy task, loss computation with label smoothing, and optimizer setup.
- **README.md**
  - Provides project documentation and usage instructions in English.
- **README.zh.md**
  - 提供项目的中文文档和使用说明。
- **requirements.txt**
  - Lists the Python packages required to run the project.
- **LICENSE**
  - License information for the project.

## Example Output

After training, the model should show decreasing loss values, indicating learning progress. The loss values and tokens per second provide insights into training performance.

## Future Plans

- **Add More Example Tasks**
  - Such as machine translation, text summarization, etc., to demonstrate model performance on different tasks.
- **Performance Optimization**
  - Introduce GPU acceleration, mixed-precision training, etc., to improve training speed and model performance.
- **Model Expansion**
  - Add pre-trained model loading, model saving, and loading functionalities.
- **Documentation Enhancement**
  - Provide more detailed code comments and usage guides.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

### Steps to Contribute

1. **Fork the repository**

2. **Create your feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit your changes**

   ```bash
   git commit -am 'Add a new feature'
   ```

4. **Push to the branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Please make sure to create a `LICENSE` file in your repository with the content of the MIT License or any other license you prefer.*

## Acknowledgments

- Thanks to [Vaswani et al.](https://arxiv.org/abs/1706.03762) for the original paper "Attention is All You Need".
- Thanks to the PyTorch community for tutorials and documentation support.


# 手写 Transformer 框架

[English Version](README.md)

一个由个人手写实现的 Transformer 框架，基于 PyTorch，实现了 Transformer 模型的各个组件，并提供了简单的训练示例。该项目旨在为学习和研究 Transformer 模型提供一个清晰、易于理解的代码基础，并将持续更新和维护。

## 目录

- [简介](#简介)
- [特性](#特性)
- [安装要求](#安装要求)
- [安装](#安装)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [示例输出](#示例输出)
- [更新计划](#更新计划)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 简介

本项目提供了一个手写的 Transformer 框架，包含了模型的详细实现和训练代码。主要目的是为了深入理解 Transformer 模型的内部机制，并为后续的功能扩展和性能优化提供基础。

## 特性

- **完整的 Transformer 模型实现**
  - 包括多头注意力机制、前馈网络、编码器和解码器层等。
- **模块化设计**
  - 代码按照功能模块划分，便于理解和扩展。
- **自定义训练流程**
  - 包含自定义的优化器和学习率调度器，以及标签平滑等技术。
- **持续更新和维护**
  - 计划添加更多功能，如预训练模型加载、更复杂的任务示例等。

## 安装要求

- Python 3.6 或以上版本
- PyTorch 1.7 或以上版本
- NumPy

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/Handwritten-Transformer.git
```

### 2. 进入项目目录

```bash
cd Handwritten-Transformer
```

### 3. 创建虚拟环境（可选）

```bash
python -m venv venv
source venv/bin/activate  # Windows 用户使用 `venv\Scripts\activate`
```

### 4. 安装依赖包

```bash
pip install -r requirements.txt
```

*请确保在项目目录下创建一个 `requirements.txt` 文件，内容如下：*

```
torch
numpy
```

## 使用方法

### 运行训练脚本

要在简单的复制任务上训练 Transformer 模型，运行：

```bash
python train.py
```

### 预期输出

脚本将输出训练过程中的信息，包括损失值和每秒处理的 Tokens 数量：

```
Epoch Step: 1 Loss: 2.477593 Tokens per Sec: 2490.268555
Epoch Step: 2 Loss: 2.117399 Tokens per Sec: 3856.200928
...
```

## 项目结构

```
Handwritten-Transformer/
├── Transformer.py   # 模型定义
├── train.py         # 训练脚本
├── README.md        # Project documentation (English)
├── README.zh.md     # 项目文档 (中文)
├── requirements.txt # 依赖包列表
└── LICENSE          # 许可证文件
```

### 文件说明

- **Transformer.py**
  - 包含 Transformer 模型的实现，包括所有子模块，如多头注意力机制、前馈网络、编码器、解码器等。
- **train.py**
  - 包含训练循环、用于复制任务的数据生成、带标签平滑的损失计算以及优化器设置。
- **README.md**
  - 提供项目的英文文档和使用说明。
- **README.zh.md**
  - 提供项目的中文文档和使用说明。
- **requirements.txt**
  - 列出了运行项目所需的 Python 包。
- **LICENSE**
  - 项目的许可证信息。

## 示例输出

训练后，模型应显示损失值的下降，表示学习的进展。损失值和每秒处理的 Tokens 数量提供了训练性能的洞察。

## 更新计划

- **添加更多示例任务**
  - 如机器翻译、文本摘要等，以展示模型在不同任务上的性能。
- **性能优化**
  - 引入 GPU 加速、混合精度训练等技术，提升训练速度和模型性能。
- **模型扩展**
  - 添加预训练模型的加载、模型保存和加载功能。
- **完善文档**
  - 提供更详细的代码注释和使用指南。

## 贡献指南

欢迎对本项目的改进提出建议！如果您有想法或新的特性，欢迎提交 Issue 或 Pull Request。

### 贡献步骤

1. **Fork 本仓库**

2. **创建您的特性分支**

   ```bash
   git checkout -b feature/您的特性名称
   ```

3. **提交您的更改**

   ```bash
   git commit -am '添加一个新的特性'
   ```

4. **推送到分支**

   ```bash
   git push origin feature/您的特性名称
   ```

5. **创建一个 Pull Request**

## 许可证

本项目基于 MIT 许可证开源，详情请参阅 [LICENSE](LICENSE) 文件。

*请确保在您的仓库中创建一个 `LICENSE` 文件，内容为 MIT 许可证或您选择的其他许可证。*

## 致谢

- 感谢 [Vaswani 等人](https://arxiv.org/abs/1706.03762)的原始论文“Attention is All You Need”。
- 感谢 PyTorch 社区的教程和文档支持。
