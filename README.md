# Adjoined-Network



# Setup
**Setting up conda** <br/>
1. wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh <br/>
2. bash Anaconda3–2018.12-Linux-x86_64.sh <br/>
3. . ~/.bashrc

**Installing libraries** <br/>
1. conda create -n fastai python=3.7 <br/>
2. conda activate fastai <br/>
3. pip install fastai <br/>
4. conda install pytorch torchvision cudatoolkit=10.2 -c pytorch <br/>
5. conda install -c anaconda ipython <br/>

# Running
<code> python train.py --dataset cifar100 --compression_factor 8 --training_type 1</code> <br/>
Running the above command would use default value (set in config.py) for batch_size, image_size, lr, c, epoch and is_sgd. <br/>
If you want to change anyone of these value then use <code>--default_config False</code> <br/>
eg. <code> python train.py --dataset cifar100 --default_config False --compression_factor 16 --lr 0.1 --training_type 1 </code> 


**Arguments** <br/>

<table>
  <tr>
    <th>Argument</th>
    <th>Discription</th>
    <th>Domain</th>
  </tr>
  <tr>
    <td>training_type</td>
    <td>Used to denote training type. 0: Standard training, 1: AN, 2:DAN Seacher, 3:DAN Training, 4: Teacher Student</td>
    <td>{0,1,2,3,4}</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Learning Rate</td>
    <td>Float</td>
  </tr>
  <tr>
    <td>is_adjoint_training</td>
    <td>True for adjoint training, false otherwise</td>
    <td>True/False</td>
  </tr>
  <tr>
    <td>is_sgd</td>
    <td>We support sgd and adam both</td>
    <td>True/False</td>
  </tr>
  <tr>
    <td>classes</td>
    <td>Denote number of classes in the dataset</td>
    <td>Integer</td>
  </tr>
  <tr>
    <td>compression_factor</td>
    <td>Normally used compression factors are 4,8,16. Default value is 4</td>
    <td>Integer</td>
  </tr>
  <tr>
    <td>resnet</td>
    <td>the resnet model to be used</td>
    <td>{18,34,50,101,152}</td>
  </tr>
  <tr>
    <td>dataset</td>
    <td>Dataset supported are cifar100, imagenet</td>
    <td>{cifar100,imagenet}</td>
  </tr>
  <tr>
    <td>default_config</td>
    <td>Setting it to true will be using parameter used in the currect experiemnt for each dataset. By default it's true</td>
    <td>True/False</td>
  </tr>
  <tr>
    <td>batch_size</td>
    <td>Batch size</td>
    <td>Integer</td>
  </tr>
  <tr>
    <td>image_size</td>
    <td>Image size</td>
    <td>Integer</td>
  </tr>
  <tr>
    <td>epoch</td>
    <td>Total number of epochs</td>
    <td>Integer</td>
  </tr>
</table>
