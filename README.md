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

### Files Description

- **Transformer.py**
  - Contains the implementation of the Transformer model, including all sub-modules like multi-head attention, feed-forward networks, encoder, decoder, etc.
- **train.py**
  - Includes the training loop, data generation for the copy task, loss computation with label smoothing, and optimizer setup.
- **README.md**
  - Provides project documentation and usage instructions in English and Chinese.
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
