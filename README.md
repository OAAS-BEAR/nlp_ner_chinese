## Table of Contents

- [Background](#background)
- [Environment install](#environment)
- [Usage](#usage)
- [Contributing](#contributer)
- [Reference](#reference)

## background
This is a research-oriented project on the task of Chinese word segmentation, which belongs to the catergory of natural language processing.

I explored  different deep network achitectures and  evaluated their  performances on Chinese word Segmentation.Based on the experienments, the network achitecture bert+lstm+crf  significantly outperforms other achitectures,including lstm+crf,bert+crf. The model also outperforms famous [HanLP](https://www.hanlp.com/product-pos.html)

I  collected a vast amount of training data from the different sources on the internet,including Weibo,[icwb2](http://sighan.cs.uchicago.edu/bakeoff2005/). You can find the training data in data/train.txt. The model's performance is evaluated based on its F1 score on the test data.The large training  dataset is very important for the model to achieve high performance. you can find the test data in data/test.txt.

The state-of-art model is trained on GPU A100. you can find the model architecture in file model.py. Any question please contact 1147182925@qq.com(Qunli Li) 



## environment

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


1. install PyCharm

  Install pycharm and use Anaconda virtual environment in pycharm (<a href="https://jingyan.baidu.com/article/f3e34a12e7b015f5eb653523.html">参考</a>)

2. Install other dependendent package

   ```sh
   # Install in nlp_chinese virtual environment
   pip install -r requirements.txt
   ```

## usage
 train

   ```
   
   # prepare data
   python data_u.py
   # train our model 
   # If the GPU related running environment is installed and configured, the command line --cuda can be added to use GPU training
   python run.py
   ```

 infer

   ```shell
   python infer.py
   ```

## contributer
Qunli Li @ Huazhong University of Science and Technology

Prof.Wei wei @ Huazhong University Of Science and Technology

## reference

[State-of-the-art Chinese Word Segmentation with Bi-LSTMs](https://aclanthology.org/D18-1529/)

[Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning](https://arxiv.org/pdf/1903.04190.pdf)

[Neural Word Segmentation with Rich Pretraining](https://aclanthology.org/P17-1078/)

[Word-Context Character Embeddings for Chinese Word Segmentation](https://aclanthology.org/D17-1079/)

   
