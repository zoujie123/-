# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:31:02 2019

@author: 邹易峰
"""

#导入库函数
import numpy as np
import random
import math

def interfence(w,b,x):
    pred_y=w*x+b
    pred_hx=1/(1+math.pow(math.e,-pred_y))
    return pred_hx


#计算代价函数
def cost(batch_y,batch_x,w,b):
    cost_new=0.0
    for i in range(len(batch_y)):
        pred_hx=interfence(w,b,batch_x[i])
        if pred_hx==1.0:
            pred_hx=0.9999
        cost_new+=-batch_y[i]*math.log(pred_hx)-(1-batch_y[i])*math.log(1-pred_hx)
    return cost_new

#求解梯度
def gradint(pred_y,get_y,x):
    dw,db=0.0,0.0
    dw=(pred_y-get_y)*x
    db=(pred_y-get_y)
    return dw,db
    
#更新系数值
def cal_re_gradint(w,b,batch_x,batch_y,lr):
    avg_w,avg_b=0.0,0.0
    for i in range(len(batch_x)):
        pred_y=interfence(w,b,batch_x[i])
        dw,db=gradint(pred_y,batch_y[i],batch_x[i])
        avg_w+=dw
        avg_b+=db
    avg_w/=len(batch_x)
    avg_b/=len(batch_x)
    w-=lr*avg_w
    b-=lr*avg_b
    return w,b
    
#训练函数
def train(x_list,gt_y_list,batch_size,lr,max_iter):
    w,b=0.0,0.0
    num_samples=len(x_list)
    for i in range(max_iter):
        batch_idex=np.random.choice(len(x_list),batch_size)
        batch_x=[x_list[j] for j in batch_idex]
        batch_y=[gt_y_list[j] for j in batch_idex]
        w,b=cal_re_gradint(w,b,batch_x,batch_y,lr)
        print('w:{0},b{1}'.format(w,b))
        print('loss is {0}'.format(cost(gt_y_list,x_list,w,b))) 
    
#参数随机样本
def gen_sample_data():
    w,b=0.0,0.0
    w=random.randint(-10,10)+random.random()
    b=random.randint(-5,5)+random.random()
    num_samples=100
    x_list=[]
    y_list=[]
    for i in range(num_samples):
        x=random.randint(-10,10)+random.random()
        pred_y=interfence(w,b,x)
        if pred_y<0.5:
            pred_y=0
        else:
            pred_y=1
        x_list.append(x)
        y_list.append(pred_y)
    return x_list,y_list,w,b

#主函数
def run():
    x_list,y_list,w,b=gen_sample_data()
    lr=5
    max_iter=10000
    train(x_list,y_list,50,lr,max_iter)
    
if __name__=='__main__':
    run()