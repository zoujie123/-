# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:09:25 2019

@author: 邹易峰
"""

#导入库文件
import numpy as np
import random

#计算预测值
def interfence(w,x,b):
    pred_y=w*x+b
    return pred_y

#计算梯度
def gradint(pred_y,get_y,x):
    dw,db=0.0,0.0
    dw=(pred_y-get_y)*x
    db=(pred_y-get_y)
    return dw,db

#计算损失函数
def cost(w,b,batch_x,batch_y):
    cost_a=0.0
    for i in range(len(batch_y)):
        pred_y=interfence(w,batch_x[i],b)
        cost_a+=0.5*(pred_y-batch_y[i])**2
    cost_a/=len(batch_y)
    return cost_a

#梯度更新函数
def re_gradint(w,b,lr,batch_x,batch_y):
    avg_w,avg_b=0.0,0.0
    dw,db=0.0,0.0
    for i in range(len(batch_y)):
        pred_y=interfence(w,batch_x[i],b)
        avg_w,avg_b=gradint(pred_y,batch_y[i],batch_x[i])
        dw+=avg_w
        db+=avg_b
    dw/=len(batch_y)
    db/=len(batch_y)
    w-=dw*lr
    b-=db*lr
    return w,b

#随机生成样本数据
def gen_sample_data():
    w=random.randint(0,10)+random.random()
    b=random.randint(0,5)+random.random()
    num_samples=100
    x_list=[]
    y_list=[]
    
    for i in range(num_samples):
        x=random.randint(0,100)*random.random()
        y=w*x+b+random.randint(-1,1)
        x_list.append(x)
        y_list.append(y)
    return x_list,y_list,w,b

def run():
    x_list,y_list,w,b=gen_sample_data()
    lr=0.001
    max_iter=1000
    train(x_list,y_list,50,lr,max_iter)
    

#模型训练函数
def train(x_list,y_list,batch_size,lr,max_iter):
    w,b=0,0
    num_samples=len(x_list)
    for i in range(max_iter):
        batch_index=np.random.choice(len(x_list),batch_size)
        batch_x=[x_list[j] for j in batch_index]
        batch_y=[y_list[j] for j in batch_index]
        w,b=re_gradint(w,b,lr,batch_x,batch_y)
        print('w:{0},b{1}'.format(w,b))
        print('loss is {0}'.format(cost(w,b,batch_x,batch_y)))  
     
if __name__=='__main__':
    run()