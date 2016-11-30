import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from decimal import *

getcontext().prec = 5 #設定湖點數精度4位
getcontext().rounding = ROUND_HALF_UP #4捨5入

#畫出激勵函數函數 Ysigmoid=1/(1+e^(-x)) ,
def sigmoid(x):
	r = 1.00000/(1+math.pow(math.e,-x))
	return r
#畫出激勵函數函數 Ysigmoid=1/(1+e^(-x)) ,
def EE(k):
	r1 = Decimal(math.pow(E[k],2))+Decimal(math.pow(E[k+1],2))+Decimal(math.pow(E[k+2],2))+Decimal(math.pow(E[k+3],2))
	#print(r1)
	return r1
#畫出函數 x13w13+x23w23-s3=0 , x14w14+x24w24-s4=0 ,
def Fx2(x0,s,w1,w2):
	r2 = (s-x0*w1)/w2
	#print(r1)
	return r2
#print(getcontext())
#最大疊代次數Pmax =預設為最大訓練次數
Pmax=50000
 # 產生輸入x1=1,x2=1
  # 產生輸入一維矩陣 x1 ,序列為[1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,1.,0.,........]
xi1=[1.,0.,1.,0.]
x1=xi1*(Pmax//4)
#print(x1)
 # 產生輸入一維矩陣 x2 ,序列為[1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,........]
xi2=[1.,1.,0.,0.]
x2=xi2*(Pmax//4)      
#print(x2)               
#Y5d ,為x1 or x2 的預期輸出 ,故每次疊代後的預期輸出應為序列[0.,1.,1.,0., 0.,1.,1.,0.,0.,1.,1.,0.,....]
Yt=[0.,1.,1.,0. ]
Y5d=Yt*(Pmax//4)   
x13=x1
x14=x1
x23=x2
x24=x2			   

#初始權重wij,及臨界值sx ,學習率a=0.1
w13=np.zeros(Pmax) #初始為0.0	
w14=np.zeros(Pmax) #初始為0.0	
w23=np.zeros(Pmax) #初始為0.0	
w24=np.zeros(Pmax) #初始為0.0	
w35=np.zeros(Pmax) #初始為0.0	
w45=np.zeros(Pmax) #初始為0.0	
s3=np.zeros(Pmax) #初始為0.0  
s4=np.zeros(Pmax) #初始為0.0  
s5=np.zeros(Pmax) #初始為0.0  
w13[0]=0.5
w14[0]=0.9
w23[0]=0.4
w24[0]=1.0
w35[0]=-1.2
w45[0]=1.1
s3[0]=0.8
s4[0]=-0.1
s5[0]=0.3
a=0.2
#宣告初始権重差值矩陣為0,只是for程式設計使用預設值
#Dx 用來修正每次疊代後須修正的權值
DW13 =np.zeros(Pmax) #初始為0.0 0
DW14=np.zeros(Pmax) #初始為0.0  
DW23 =np.zeros(Pmax) #初始為0.0 0
DW24=np.zeros(Pmax) #初始為0.0  
DW35 =np.zeros(Pmax) #初始為0.0 0
DW45=np.zeros(Pmax) #初始為0.0 0
DS3=np.zeros(Pmax) #初始為0.0 
DS4=np.zeros(Pmax) #初始為0.0 
DS5=np.zeros(Pmax) #初始為0.0 
#宣告初始誤差E為0,只是for程式設計使用預設值
#E 為每次疊代後,期望值與實際輸出值的誤差
E =np.zeros(Pmax) #初始為0.0 
#為每次疊代後,誤差梯度
e3 =np.zeros(Pmax) #初始為0.0 
e4 =np.zeros(Pmax) #初始為0.0 
e5 =np.zeros(Pmax) #初始為0.0 
#宣告初始實際輸出值Y矩陣為0,只是for程式設計使用預設值
#第p次疊代實際輸出Y3,Y4,Y5
Y3 =np.zeros(Pmax) #初始為0.0
Y4 =np.zeros(Pmax) #初始為0.0
Y5 =np.zeros(Pmax) #初始為0.0
#Epoch ,疊代次數p
print("疊代次數|輸入x1|輸入x2|期望輸出Yd|實際輸出Y| 權重w35  | 權重w45 | 誤差E  | ")
for p in range(Pmax-1):  #from	 0,1... to Pmax
	#print("疊代次數:",p)
	#w13=w13[p]
	#w23=w23[p]
	#s3=s3[p]
	#w14=w14[p]
	#w24=w24[p]
	#s4=s4[p]
	Y3[p]=Decimal(sigmoid(x1[p]*w13[p]+x2[p]*w23[p]-s3[p]))
	#print(Y3[p])
	Y4[p]=Decimal(sigmoid(x1[p]*w14[p]+x2[p]*w24[p]-s4[p]))
	#print(Y4[p])
	Y5[p]=Decimal(sigmoid(Y3[p]*w35[p]+Y4[p]*w45[p]-s5[p]))
	#print(Y5[p])
	E[p]=Decimal(Y5d[p]-Y5[p])
	#print(E[p])
	e5[p]=Decimal(Y5[p])*Decimal(1-Y5[p])*Decimal(E[p])
	#print(e5[p])
	DW35[p]=Decimal(a)*Decimal(Y3[p])*Decimal(e5[p])
	#print(DW35[p])
	DW45[p]=Decimal(a)*Decimal(Y4[p])*Decimal(e5[p])
	#print(DW45[p])
	DS5[p]=Decimal(a)*Decimal(-1)*Decimal(e5[p])
	#print("DS5[p]=",DS5[p])
	e3[p]=Decimal(Y3[p])*Decimal(1-Y3[p])*Decimal(e5[p])*Decimal(w35[p])
	#print(e3[p])
	DW13[p]=Decimal(a)*Decimal(x1[p])*Decimal(e3[p])
	#print(DW13[p])
	DW23[p]=Decimal(a)*Decimal(x2[p])*Decimal(e3[p])
	#print(DW23[p])
	DS3[p]=Decimal(a)*Decimal('-1.00000')*Decimal(e3[p])
	#print("DS3[p]=",DS3[p])
	e4[p]=Decimal(Y4[p])*Decimal(1-Y4[p])*Decimal(e5[p])*Decimal(w45[p])
	#print(e4[p])
	DW14[p]=Decimal(a)*Decimal(x1[p])*Decimal(e4[p])
	#print(DW14[p])
	DW24[p]=Decimal(a)*Decimal(x2[p])*Decimal(e4[p])
	#print(DW24[p])
	DS4[p]=Decimal(a)*Decimal('-1.00000')*Decimal(e4[p])
	#print("DS4[p]=",DS4[p])
	
	w13[p+1]=Decimal(w13[p]+DW13[p])
	#print(w13[p])
	w14[p+1]=Decimal(w14[p]+DW14[p])
	#print(w14[p])
	w23[p+1]=Decimal(w23[p]+DW23[p])
	#print(w23[p])
	w24[p+1]=Decimal(w24[p]+DW24[p])
	#print(w24[p])
	w35[p+1]=Decimal(w35[p]+DW35[p])
	#print(w35[p])
	w45[p+1]=Decimal(w45[p]+DW45[p])
	#print(w45[p])
	s3[p+1]=Decimal(s3[p]+DS3[p])
	#print(s3[p+1])
	s4[p+1]=Decimal(s4[p]+DS4[p])
	#print(s4[p+1])
	s5[p+1]=Decimal(s5[p]+DS5[p])
	#print(s5[p+1])

	#print("疊代次數|輸入x1|輸入x2|期望輸出Yd|實際輸出Y| 權重w35  | 權重w45 | 誤差E  |  ")
	print('    {0:1d}      {1:1d}      {2:1d}       {3:1d}        {4:1.5f}       {5:1.5f}       {6:1.5f}     {7:1.5f}   '\
	.format(p, int(x1[p]),int(x2[p]),int(Y5d[p]),Y5[p],w35[p+1],w45[p+1],E[p]))
	if p>0 & int(p%4)==0: #每4次唯一週期,計算依次誤差平方和
		ee=EE(int(p-4)) 
		if ee<0.005:     #如果小於0.01 結束
			#print("誤差平方和=",ee)
			Pend=p
			break
		else:
			Pend=Pmax


print("訓練結果==========================")
ee=EE(int(Pend-4)) 
print("疊代次數=",Pend)
print("權重w13=",w13[p+1])
print("權重w14=",w14[p+1])
print("權重w23=",w23[p+1])
print("權重w24=",w24[p+1])
print("權重w35=",w35[p+1])
print("權重w45=",w45[p+1])
print("權重s3=",s3[p+1])
print("權重s4=",s4[p+1])
print("權重s5=",s5[p+1])
print("誤差平方和=",ee)

#====Prediction==========================
Xr1=[1.,0.,1.,0.]
Xr2=[1.,1.,0.,0.]
Yrt=[0.,1.,1.,0.]
Yr3=[0.,0.,0.,0.]
Yr4=[0.,0.,0.,0.]
Yr5=[0.,0.,0.,0.]
Er=[0.,0.,0.,0.]
eer=0
Wr13=w13[p+1]
Wr14=w14[p+1]
Wr23=w13[p+1]
Wr24=w14[p+1]
Wr35=w35[p+1]
Wr45=w45[p+1]
sr3=s3[p+1]
sr4=s4[p+1]
sr5=s5[p+1]
print("測試結果======================================================")
print("輸入X1|輸入X2 |期望輸出Yd| 實際輸出Y |  誤差E  | 誤差平方和EE|")
for j in range(4):  #from	 0,1... to Pmax
	Yr3[j]=sigmoid(Xr1[j]*Wr13+Xr2[j]*Wr23-sr3)
	Yr4[j]=sigmoid(Xr1[j]*Wr14+Xr2[j]*Wr24-sr4)
	Yr5[j]=sigmoid(Yr3[j]*Wr35+Yr4[j]*Wr45-sr5)
	Er[j]=Yrt[j]-Yr5[j]
	if j==3: #每4次唯一週期,計算依次誤差平方和
		eer= Decimal(math.pow(Er[0],2))+Decimal(math.pow(Er[1],2))+Decimal(math.pow(Er[2],2))+Decimal(math.pow(Er[3],2)) 
	print('   {0:1d}    {1:1d}          {2:1d}       {3:1.5f}     {4:1.5f}    {5:1.5f} '.format(int(Xr1[j]),int(Xr2[j]),int(Yrt[j]),Yr5[j],Er[j],eer))

#畫出激勵函數函數 Ysigmoid=1/(1+e^(-x)) ,
sx=np.linspace(-10.0, 10.0, 50)
sg=np.array([sigmoid(t) for t in sx])     

plt.subplot(2,2,1)				
plt.plot(sx, sg, 'b-',linewidth=2.0)			#紅色線寬可以改成2.0或其他數值
plt.axis([-10.0,10.0, -0.5,1.5])			#分別設定X,Y軸的最小最大值
plt.title(' Ysigmoid=1/(1+e^(-x)) ',fontsize=10)
plt.ylabel('Ysigmoid=1/(1+e^(-x))')
plt.xlabel('x')
plt.grid(True)

sp=np.arange(int(Pend/4))
se=np.array([EE(k) for k in range(0,Pend-4,4)])  
#if se.shape>sp.shape:
sp=sp[0:se.shape[0]]
#print("sp.shape=",sp.shape)
#print("se.shape=",se.shape)

plt.subplot(2,2,2)				
plt.plot(sp, se, 'r-',linewidth=2.0)			#紅色線寬可以改成2.0或其他數值
plt.axis([0,Pend/4, 0.0,1.1])			#分別設定X,Y軸的最小最大值
plt.title('Sum(E)^2 ',fontsize=10)
plt.ylabel('Sum(E)^2')
plt.xlabel('Period')
plt.grid(True)


Px=np.linspace(-10.0, 10.0, 100)
p23=np.array([Fx2(k,s3[p+1],w13[p+1],w23[p+1]) for k in Px])  
p24=np.array([Fx2(k,s4[p+1],w14[p+1],w24[p+1]) for k in Px])  
	
plt.subplot(2,2,3)	
plt.plot(Px, p23, 'b-',linewidth=2.0)		#紅色線寬可以改成2.0或其他數值
plt.plot(Px, p24, 'r-',linewidth=2.0)		#紅色線寬可以改成2.0或其他數值
plt.axis([-0.5,2.0,-0.5,2.0])   		#分別設定X,Y軸的最小最大值
plt.title(' X1 xor X2 Result: ',fontsize=10)
plt.ylabel('X2')
plt.xlabel('X1')
plt.grid(True)
plt.plot(0,0,'ro')
plt.text(0, 0, r'P0(0,0)')
plt.plot(0,1,'ro')
plt.text(0, 1, r'P1(0,1)')
plt.plot(1,1,'ro')
plt.text(1, 1, r'P3(1,1)')
plt.plot(1,0,'ro')
plt.text(1, 0, r'P4(1,0)')
plt.fill_between(Px, p23, p24, color='green', alpha='0.5')

plt.show()

