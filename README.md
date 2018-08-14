
# Reference
[Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)  
[Blog](https://duanyzhi.github.io/One-Shot-Learning/)

# Data
[Omniglot](https://github.com/brendenlake/omniglot)

download Omniglot data: python/images_background.zip python images_evaluation.zip  

put Omniglot data in data folder and unzip them and your data folder like this:  

data   
----| images_background  
-------|Alphabet_of_the_Magi  
----------| character01  
-------------| 0709_01.pag      
-------------| 0709_02.png       
              ...  
          | character02    
            ...  
     |Anglo-Saxon_Futhorc  
          |character01  
             ...  
          ...    
     ...  
  | images_evaluation  
      | Angelic  
         | character01    
             | 0965_01.pag  


mkdir data/ckpt    
mkdir data/csv  
mkdir data/png  


# Install
python3.5  
tensorflow-1.3  
opencv2  

# How To Run
python main --pattern train     # training     
python main --pattern test      # test   

# Result
It is easy to get high accuracy for one shot learning model with omniglot  
iter:1000  
batch_size:32  

five-ways-five-shot:   
![ACC](https://github.com/duanyzhi/one_shot_learning/blob/master/data/png/acc.png)  

![LOSS](https://github.com/duanyzhi/one_shot_learning/blob/master/data/png/loss.png)  

I just run 1000 iter, so you can run more for better acc  
