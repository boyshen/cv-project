#!/usr/bin/python

import os
import cv2
import shutil
import numpy as np

from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# 异常数据保存目录
abnormal_dir = 'abnormal_picture'

# 定义狗的 decode 编码
dogs = [
    'n02085620', 'n02085782', 'n02085936', 'n02086079'
    , 'n02086240', 'n02086646', 'n02086910', 'n02087046'
    , 'n02087394', 'n02088094', 'n02088238', 'n02088364'
    , 'n02088466', 'n02088632', 'n02089078', 'n02089867'
    , 'n02089973', 'n02090379', 'n02090622', 'n02090721'
    , 'n02091032', 'n02091134', 'n02091244', 'n02091467'
    , 'n02091635', 'n02091831', 'n02092002', 'n02092339'
    , 'n02093256', 'n02093428', 'n02093647', 'n02093754'
    , 'n02093859', 'n02093991', 'n02094114', 'n02094258'
    , 'n02094433', 'n02095314', 'n02095570', 'n02095889'
    , 'n02096051', 'n02096177', 'n02096294', 'n02096437'
    , 'n02096585', 'n02097047', 'n02097130', 'n02097209'
    , 'n02097298', 'n02097474', 'n02097658', 'n02098105'
    , 'n02098286', 'n02098413', 'n02099267', 'n02099429'
    , 'n02099601', 'n02099712', 'n02099849', 'n02100236'
    , 'n02100583', 'n02100735', 'n02100877', 'n02101006'
    , 'n02101388', 'n02101556', 'n02102040', 'n02102177'
    , 'n02102318', 'n02102480', 'n02102973', 'n02104029'
    , 'n02104365', 'n02105056', 'n02105162', 'n02105251'
    , 'n02105412', 'n02105505', 'n02105641', 'n02105855'
    , 'n02106030', 'n02106166', 'n02106382', 'n02106550'
    , 'n02106662', 'n02107142', 'n02107312', 'n02107574'
    , 'n02107683', 'n02107908', 'n02108000', 'n02108089'
    , 'n02108422', 'n02108551', 'n02108915', 'n02109047'
    , 'n02109525', 'n02109961', 'n02110063', 'n02110185'
    , 'n02110341', 'n02110627', 'n02110806', 'n02110958'
    , 'n02111129', 'n02111277', 'n02111500', 'n02111889'
    , 'n02112018', 'n02112137', 'n02112350', 'n02112706'
    , 'n02113023', 'n02113186', 'n02113624', 'n02113712'
    , 'n02113799', 'n02113978']

# 定义猫的 decode 编码
cats = ['n02123045', 'n02123159', 'n02123394', 'n02123597'
    , 'n02124075', 'n02125311', 'n02127052']

def path_to_tensor(img_path):
    """
    将一张图片转换为张量
    :param img_path: 图片的绝对路径
    :return: 图片张量
    """
    img = image.load_img(img_path,target_size=(224,224))  #加载图片
    x = image.img_to_array(img)     #将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    return np.expand_dims(x,axis=0) #将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回

def paths_to_tensor(img_paths):
    """
    将多张图片转换为张量
    :param img_paths: 多张图片的路径
    :return: 图片张量
    """
    list_of_tensor = [path_to_tensor(img) for img in img_paths]
    return np.vstack(list_of_tensor)

def comparison_result(decodes):
    """
    :param decodes: 图片预测的decodes集
    :return: True or False .True 则说明图片是猫或狗
    """
    for decode in decodes:
        if decode in dogs:
            return True

        elif decode in cats:
            return True
    return False

def save_abnormal_picture(filename,lines):
    """
    保存异常图片信息
    :param filename: str,保存文件名
    :param file_list: list,写入信息列表
    :return: 无
    """

    # 如果目录不存在则创建
    if not os.path.exists(abnormal_dir):
        os.mkdir(abnormal_dir)
        print("create {} dir success".format(abnormal_dir))

    file_path = os.path.join(abnormal_dir,filename)
    with open(file_path,'w') as file:
        for line in lines:
            file.write(line+'\n')

    print("save file success! file Dir : {}".format(abnormal_dir))

def read_abnormal_file(file_path):
    """
    读文件记录
    :param file_path: str,文件名
    :return: list,读取文件的信息
    """
    read_list = list()

    if not os.path.exists(file_path):
        print("file not found ! Path : {}".format(file_path))
        return False

    with open(file_path,'r') as file:
        for line in file.readlines():
            line = line.strip("\n")
            read_list.append(line)

    return read_list

def check_abnormal_picture(model,img_dir,top_s,num=0,count=100):
    """
    检测数据中的异常图片
    :param model: 使用的模型，resnet or xception
    :param img_dir: str，图片目录
    :param top_s: int，top 数值
    :param num: int，如果只检测一部分数据，则在这里定义
    :param count: 输出间隔
    :return: 异常图片列表
    """
    abnormal_picture = list()
    images = os.listdir(img_dir) if num==0 else os.listdir(img_dir)[:num]

    for i, image in enumerate(images):
        img_path = os.path.join(img_dir, image)
        decodes = model.model_predict(top_s=top_s, img_path=img_path)
        if comparison_result(decodes=decodes) is False:
            abnormal_picture.append(image)

        if i % count == 0:
            print("current progress %d/%d, abnormal picture num: %d" \
                  % (i / count + 1, len(images) / count, len(abnormal_picture)))

    return abnormal_picture

def cv_read_image(img_path):
    """
    将图片转为可在matplotlib中显示
    :param img_path: 图片路径
    :return: 图片对象
    """
    # 加载图片
    img = cv2.imread(img_path)
    # 将opencv中的BGR、GRAY格式转换为RGB，使matplotlib中能正常显示opencv的图像 并返回
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def preview_pictures(image_names, img_path, img_num=0,line_num=6,fig_size=(20,10)):
    """
    预览图片
    :param image_names: list，图片列表
    :param img_path:str，图片所在目录
    :param img_num:int，预览图片数量
    :param line_num:int，每行显示预览的数量
    :param fig_size: tuple，大小
    :return:
    """
    images = list()
    for image in image_names:
        path = os.path.join(img_path, image)
        img = cv_read_image(path)
        images.append(img)

    # 如果预览图片的数量为0，则根据可预览的图片列表数量进行预览
    num = img_num if img_num != 0 else len(images)
    images = images[:num]
    print("preview picture num : %d" %num)

    fig = plt.figure(figsize=fig_size)
    for i, img in enumerate(images):
        ax = fig.add_subplot(num, line_num, i + 1, xticks=[], yticks=[])
        ax.imshow(img)
        ax.set_title(image_names[i])
    plt.show()

def create_symlink(img_dir,symlink_name,img_data):
    """
    创建软链接
    :param img_dir: str，图片路径(绝对路径)
    :param symlink_name: str，软链接目录名(绝对路径或相对路径)
    :param img_data: list，图片数据
    :return:
    """

    if not os.path.exists(img_dir):
        print("Image dir not found ! dir : {}".format(img_dir))
        return False

    if not os.path.exists(symlink_name):
        os.makedirs(symlink_name)
    else:
        for file in os.listdir(symlink_name):
            file_path = os.path.join(symlink_name,file)
            os.remove(file_path)

    for img in img_data:
        img_path = os.path.join(img_dir,img)
        img_symlink = os.path.join(symlink_name,img)

        if os.path.exists(img_symlink):
            continue

        os.symlink(img_path,img_symlink)

    print("create symlink {} success !".format(symlink_name))

def create_valid_datasets(train_dir,valid_dir,split=0.2):
    """
    创建验证数据集
    :param train_dir: str，训练数据集目录（绝对路径）
    :param valid_dir: str，验证数据集目录（绝对路径）,需要为空目录.
    :param split: float，比例
    :return: 无
    """

    if not os.path.exists(train_dir):
        print("{} dir not found ".format(train_dir))
        return False

    image_list = os.listdir(train_dir)

    # 验证目录不存在，则创建
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    # 目录存在，则判断目录下是否有文件，有文件则删除
    else:
        if os.listdir(valid_dir):
            for file_name in os.listdir(valid_dir):
                file_path = os.path.join(valid_dir,file_name)
                os.remove(file_path)

    # 对数据进行洗牌操作，打乱顺序
    np.random.shuffle(image_list)

    # 选取一定比例的数据集
    valid_data = image_list[:int(len(image_list) * split)]

    for img in valid_data:
        img_path = os.path.join(train_dir,img)
        shutil.move(img_path,valid_dir)

    print("create valid datasets success ! dir :{}".format(valid_dir))

def Data_Generator(directory,target_size,batch_size=32,is_train=False):
    """
    数据增强生成张量图像数据批次
    :param directory: str，数据目录.该目录下需要包含字目录，子目录下保存数据
    :param target_size: tuple，调整图片到指定尺寸
    :param batch_size: 一批数据的大小，默认是32
    :param is_train: 是否时train数据集.默认为False
    :return: 批量数据增强
    """

    if not os.path.exists(directory):
        print("{} dir not found !".format(directory))
        return False

    if is_train:
        data_generator = ImageDataGenerator(rescale=1.0/255,
                                            rotation_range=20,
                                            width_shift_range=0.02,
                                            height_shift_range=0.02,
                                            zoom_range=0.25,
                                            horizontal_flip=True)
    else:
        data_generator = ImageDataGenerator(rescale=1.0/255)

    generator = data_generator.flow_from_directory(directory,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       shuffle=False)

    return generator