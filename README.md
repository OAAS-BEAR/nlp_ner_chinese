## Table of Contents

- [Background](#background)
- [Main work](# My work)
- [Environment install](#environment)
- [Usage](#usage)

## background

1. install Anaconda

   visit https://www.anaconda.com/products/individual#windows


2. create vitual environment PyTorch
    ```shell
    # create vitual environment
    conda create -n nlp_chinese python=3.7	

    # Virtual environment related commands
    conda activate nlp_chinese  # activate vitual environment nlp_chinese，
    conda deactivate       # Exit the current virtual environment
    conda info -e          # View all virtual environments, * indicates the current environment

    # install Pytorch 1.6.0 
    # Note: activate nlp_chinese virtual environment before installation
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda install pytorch==1.6.0
    ```
## 

1. install PyCharm

  Install pycharm and use Anaconda virtual environment in pycharm (<a href="https://jingyan.baidu.com/article/f3e34a12e7b015f5eb653523.html">参考</a>)

2. Install other dependendent package

   ```sh
   # Install in nlp_chinese virtual environment
   pip install -r requirements.txt
   ```

3. train

   ```
   
   # prepare data
   python data_u.py
   # train our model 
   # If the GPU related running environment is installed and configured, the command line --cuda can be added to use GPU training
   python run.py
   ```

4. infer

   ```shell
   python infer.py
   ```

   
