
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as opt

'''============================part1 数据导入及可视化========================='''
data = scio.loadmat('ex8_movies.mat')
y = data['Y']
r = data['R']

#第一部电影《玩具总动员》的平均得分
print('Average rating for movie 1 (Toy Story): %f / 5\n\n'%np.mean(y[0, np.where(r[0, :]==1)]))

#可视化评分矩阵
#MATLAB中imagesc(A)将矩阵A中的元素数值按大小转化为不同颜色，并在坐标轴对应位置处以这种颜色染色
plt.figure(figsize=(5,9))
plt.imshow(y)
plt.colorbar() #加颜色条
plt.ylabel('Movies')
plt.xlabel('Users')

'''============================part2 协同过滤的代价函数=============================='''
#导入训练好的参数
parameters = scio.loadmat('ex8_movieParams.mat')
x = parameters['X'] #电影的特征矩阵
theta = parameters['Theta'] #用户的特征矩阵
num_users = parameters['num_users']
num_movies = parameters['num_movies']
num_features = parameters['num_features']

#减小数据集用来更快的测试代价函数的正确性
num_users = 4
num_movies = 5 
num_features = 3
X = x[0:num_movies, 0:num_features]
Theta = theta[0:num_users, 0:num_features]
Y = y[0:num_movies, 0:num_users]
R = r[0:num_movies, 0:num_users]

'''代价函数'''
#这里的params包含x和theta两个矩阵，它是一维的，因为要使用fmin_ncg函数的原因需要展开
def cofiCostFunc(params, y, r, num_users, num_movies, num_features, lmd=0):
    #reshape得到x和theta矩阵
    x = np.reshape(params[0:num_movies*num_features], (num_movies,num_features))
    theta = np.reshape(params[num_movies*num_features:], (num_users,num_features))
    #代价函数
    J = np.sum(np.square((x@theta.T-y))[np.where(r==1)])/2 
    reg = (lmd/2) * (np.sum(x*x) + np.sum(theta*theta))
    return J+reg

#展开参数
params = np.r_[X.flatten(), Theta.flatten()]
#测试代价函数是否正确
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lmd=0)
print('Cost at loaded parameters: %f \n(this value should be about 22.22)\n'% J)


'''============================part3 梯度函数与梯度检验=============================='''
'''梯度函数'''
def cofiGradFunc(params, y, r, num_users, num_movies, num_features, lmd=0):
    #reshape得到x和theta矩阵
    x = np.reshape(params[0:num_movies*num_features], (num_movies,num_features))
    theta = np.reshape(params[num_movies*num_features:], (num_users,num_features))
    #梯度函数
    x_grad = ((x@theta.T-y)*r)@theta + lmd*x #shape和x一样，为(num_movies,num_features)
    theta_grad = ((x@theta.T-y)*r).T@x + lmd*theta #shape和theta一样，为(num_users,num_features)
    grad = np.r_[x_grad.flatten(), theta_grad.flatten()]
    return grad

cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lmd=0).shape #(27,)

'''梯度检查'''
#数值梯度
def computeNumericalGradient(J, params):
    numgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)
    e =1e-4
    for p in range(len(params)):
        perturb[p] = e
        loss1 = J(params - perturb)
        loss2 = J(params + perturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

#重写一下代价函数，以免重复写其他参数（Y, R, num_users, num_movies, num_features, lmd=0）
def cost_func(t):
    return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lmd=0)

#计算数值梯度与理论梯度之间的差值
numgrad = computeNumericalGradient(cost_func, params)
grad = cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lmd=0)
diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
print('If your cost function implementation is correct, then \n' 
         'the relative difference will be small (less than 1e-9). \n' 
         '\nRelative Difference: %g\n'% diff)


'''============================part4 正则化的代价函数和梯度函数========================'''
#修改前面的两个函数，加上正则化的部分
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lmd=1.5)
print('Cost at loaded parameters (lambda = 1.5): %f \n(this value should be about 31.34)\n'% J)

#正则化之后的梯度检验
def cost_func(t):
    return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lmd=1.5)
#计算数值梯度与理论梯度之间的差值
numgrad = computeNumericalGradient(cost_func, params)
grad = cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lmd=1.5)
diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
print('If your cost function implementation is correct, then \n' 
         'the relative difference will be small (less than 1e-9). \n' 
         '\nRelative Difference: %g\n'% diff)


'''============================part5 新用户增加评分=============================='''      
#电影名称的文件里有一些电影好像是法文电影，所以编码上有些问题，会出现UnicodeDecodeError，所以进行以下步骤
#检查文件中哪些字符的编码转不过来
f = open('movie_ids.txt',"rb")#二进制格式读文件
lines = f.readlines()
for line in lines:
    try:
        line.decode('utf8') #解码
    except:
        #打印编码有问题的行
        print(str(line))
f.close
            
#修改那些好像是法文的字符，文件另存为movie_ids_mod.txt，再进行电影列表读入
t = open('movie_ids_mod.txt')
movie_list = []
for line in t:
    #先用strip去掉开头结尾的空格，然后用split对空格切片，选取从编号之后的所有字符串，再用jion空格连接成一个字符串
    movie_list.append(' '.join(line.strip().split(' ')[1:]))
t.close

len(movie_list) #得到了长度为1682的电影列表

#初始化我的评分
my_ratings = np.zeros((1682, 1))

#添加电影评分，注意这里的索引比作业中少1，从0开始的
my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

#查看评分以及对应的电影
print('\n\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s\n' %(my_ratings[i], movie_list[i]))


'''============================part6 训练协同过滤算法=============================='''
#y和r是part1中的矩阵，在此基础上加上my_ratings构成新的评分矩阵
Y = np.c_[my_ratings, y] #(1682, 944)
R = np.c_[(my_ratings != 0), r] #(1682, 944)

'''标准化'''
def normalizeRatings(y, r):
    #为了下面可以直接矩阵相减，将(1682,)reshape成(1682,1)
    mu = (np.sum(y, axis=1)/np.sum(r, axis=1)).reshape((len(y),1)) 
    y_norm = (y - mu)*r #未评分的依然为0
    return y_norm, mu

#标准化
Ynorm, Ymean = normalizeRatings(Y, R)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

#随机初始化参数
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.r_[X.flatten(), Theta.flatten()]

#训练模型
lmd = 10
#params = opt.fmin_ncg(f=cofiCostFunc, fprime=cofiGradFunc, x0=initial_parameters, args=(Y, R, num_users, num_movies, num_features, lmd), maxiter=100)
#上面的方法不知道为什么特别慢，跑不出来，但其实应该是一样的啊，不解~
res = opt.minimize(fun=cofiCostFunc,
                   x0=initial_parameters,
                   args=(Y, R, num_users, num_movies, num_features, lmd),
                   method='TNC',
                   jac=cofiGradFunc,
                   options={'maxiter': 100})

#得到模型参数
params = res.x
X = np.reshape(params[0:num_movies*num_features], (num_movies,num_features))
Theta = np.reshape(params[num_movies*num_features:], (num_users,num_features))


'''============================part7 预测评分并推荐=============================='''
#预测的评分矩阵
p = X @ Theta.T
#我的预测评分
my_predictions = (p[:,0].reshape(len(p),1) + Ymean).flatten() #为了矩阵相加，后面展开是为了打印方便

#为我推荐的10部电影
ix = np.argsort(my_predictions)[::-1] #逆序，由大到小得到索引
print('\nTop recommendations for you:\n')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s\n' %(my_predictions[j], movie_list[j]))

#我原来的评分
print('\nOriginal ratings provided:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s\n' %(my_ratings[i], movie_list[i]))


