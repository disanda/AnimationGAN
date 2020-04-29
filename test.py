import torch
import model
import time
import torchvision

#model_dict=torch.load('/parameter.pkl')#这是一个字典,key是模型名字，value是模型参数
model_dict=torch.load('../model.ckpt',map_location=torch.device('cpu'))#这个是在cpu环境下加载所需额外参数
G = model.Generator_v1_1(100,12)
G.load_state_dict(model_dict['G'])
#G.eval()


#------------------info------------------
z_sample = torch.randn(100, 100)
c_sample = torch.zeros([100,10])
c_sample[:,6]=1 #全是6

z_sample = torch.randn(100,100)#每次生成不固定noise
temp_c = torch.linspace(-1, 1, 10)
c1 = torch.zeros([100,1])
c2 = torch.zeros([100,1])
c1[:,0]=temp_c[0]
for i2 in range(100):
	c1[i2]=temp_c[i2%10]
	#c2[i2]=temp_c[i2%10]
#c_con = torch.cat([c1,c2],-1)#[-1,2]
c_con = torch.cat([c2,c1],-1)#[-1,2]

c_all = torch.cat([c_sample,c_con],-1)#[-1,10+2]
x_f_sample = (G(z=z_sample, c=c_all) + 1) / 2.0
torchvision.utils.save_image(x_f_sample, '%s.jpg' % (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())), nrow=10)

# print('------------------')
# print(z_sample.shape)#[-1,100]
# print(c_sample.shape)#[-1,10]
# print(c_con.shape)#[-1,10]
# print(c_all.shape)#[-1,10]
# print('------------------')