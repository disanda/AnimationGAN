import torch
import model
import time
import torchvision


batch_size = 120
z_dim_num = 100
c_d_num = 20
c_c_num = 20
input_size = 64
img_channel = 3
sample_num =400

#model_dict=torch.load('/parameter.pkl')#这是一个字典,key是模型名字，value是模型参数
model_dict=torch.load('./info_output/shape-c_d20-c_c20(E-61).ckpt',map_location=torch.device('cpu'))#这个是在cpu环境下加载所需额外参数
G = model.generator_mwm(z_dim=z_dim_num, output_channel=img_channel, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)
G.load_state_dict(model_dict['G'])
#G.eval()


#----------------z,d,c-----------------
# 固定noise和cc，每c_d个变一次c_d
sample_z_d = torch.zeros((sample_num, z_dim_num))
temp = torch.zeros((c_d_num, 1))
for i in range(sample_num//c_d_num):
	sample_z_d[i * c_d_num] = torch.rand(1, z_dim_num)#为连续c_d个的样本赋值。
	for j in range(c_d_num):
		sample_z_d[i * c_d_num + j] = sample_z_d[i * c_d_num]#相同c_d的noize都相同

for i in range(c_d_num):
	temp[i, 0] = i #每一个标签
temp_d = torch.zeros((sample_num, 1))
for i in range(sample_num//c_d_num):
	temp_d[i * c_d_num: (i + 1) * c_d_num] = temp[i%c_d_num] #每c_d个的d一样
sample_d = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)
sample_c = torch.zeros((sample_num, c_c_num))

# 观察单一变量，固定其他变量
sample_z2 = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #每个样本的noize相同
sample_d2 = torch.zeros(sample_num, c_d_num)#[200,c_d]

temp_c = torch.linspace(-1, 1, c_d_num)		#c_d_num个范围在-1->1的等差数列
sample_c2 = torch.zeros((sample_num, c_c_num))#[200,c_c]

for i in range(sample_num//c_d_num):		#每c_d个noise,c_d相同,c_c不同
	#d_label = random.randint(0,c_d_num-1)
	d_label = i%c_d_num
	sample_d2[i*c_d_num:(i+1)*c_d_num, d_label] = 1
	sample_c2[i*c_d_num:(i+1)*c_d_num,i%c_c_num] = temp_c

temp = torch.zeros((c_d_num, 1))
temp_d = torch.zeros((sample_num, 1))
for i in range(sample_num):
	temp_d[i] = temp[i%c_d_num] #每c_d个的d一样
sample_d2_2 = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)

#顺序d
temp = torch.zeros((c_d_num, 1))
temp_d = torch.zeros((sample_num, 1))

for i in range(c_d_num):
	temp[i, 0] = i #每一个标签
for i in range(sample_num):
	temp_d[i] = temp[i%c_d_num] #每c_d个的d不一样

sample_d2 = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)


#--------------直观查一个变量
#----z
#sample_d 每行d一样
#sample_z_d 每行z一样
#sample_d2 顺序d,值为1,不含d内
#sample_c2 顺序d,含d内-1->1


sample_z = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #所有样本z相同
sample_c0 = torch.zeros((sample_num, c_c_num))#全部样本为0

sample_d2 = sample_d2*20
sample_c2 = sample_c2*20



with torch.no_grad():
	G.eval()
	# samples = G(sample_z, sample_c, sample_d)
	# samples = (samples + 1) / 2 #颜色更深[-1,1]->[0,1]
	# torchvision.utils.save_image(samples, 'c_d_%s.jpg' % (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())), nrow=20)
	# samples = G(sample_z2, sample_c2, sample_d2)
	# samples = (samples + 1) / 2 #颜色更深[-1,1]->[0,1]
	# torchvision.utils.save_image(samples, 'c_c_%s.jpg' % (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())), nrow=20)
	print(sample_z_d[0] == sample_z_d[21])
	#samples = G(sample_z_d,sample_d2,sample_c2)
	samples = G(sample_z_d,sample_d2,sample_d2)
	samples = (samples + 1) / 2 #颜色更深[--,1]->[0,1]
	torchvision.utils.save_image(samples, 'd2-d2-%s.jpg' % (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())), nrow=20)



# print('------------------')
# print(z_sample.shape)#[-1,100]
# print(c_sample.shape)#[-1,10]
# print(c_con.shape)#[-1,10]
# print(c_all.shape)#[-1,10]
# print('------------------')