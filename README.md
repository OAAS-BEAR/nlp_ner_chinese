

## environment preparation

1. install Anaconda

   <a href="https://zhuanlan.zhihu.com/p/75717350">windows 下安装教程</a>

   ：<a href="https://docs.continuum.io/anaconda/install/">anaconda install</a>

2. create vitual environment PyTorch
    ```shell
    # create vitual environment
    conda create -n nlplab python=3.7	

    # Virtual environment related commands
    conda activate nlplab  # activate vitual environment nlplab，
    conda deactivate       # Exit the current virtual environment
    conda info -e          # View all virtual environments, * indicates the current environment

    # install Pytorch 1.6.0 
    # Note: activate nlplab virtual environment before installation
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda install pytorch==1.6.0
    ```
## 

1. install PyCharm

  Install pycharm and use Anaconda virtual environment in pycharm (<a href="https://jingyan.baidu.com/article/f3e34a12e7b015f5eb653523.html">参考</a>)

2. Install other dependendent package

   ```sh
   # 在 nlplab 虚拟环境中安装
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

   
