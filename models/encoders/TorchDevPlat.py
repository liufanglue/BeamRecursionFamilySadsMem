#-*- coding:utf-8 -*-
# coding:unicode_escape

import os
import math
import torch
import xlwt
import random
import datetime
import pandas as pd
import numpy as np
import csv
import codecs
import torchvision
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms

from einops import rearrange
from torch.autograd import Function
import torch.optim.lr_scheduler as lr_scheduler
from sympy.abc import P
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
#from sklearn.model_selection import train_test_split

from models.encoders.S4DWrapper import S4DWrapper

INVALID_NUM = 0xFFFFFFFF

# CSV转Torch 数据类 begin

class TransCsvToTorch:
    def __init__(self, filePath, codeFormat, dateCol, dataStartRank, dataStartCol, isNeedRevert):
        #print(f"TransCsvToTorch enter filePath = {filePath}")
        if not os.path.exists(filePath):
            print(r'TransCsvToTorch path not exist')
            return
        dataTbl = pd.read_csv(filePath, encoding = codeFormat, sep = ',', header = None)
        # 将 '--'、空字符串和 NaN 替换为 NaN，以便进行填补
        dataTbl.replace(['--', '', np.nan], np.nan, inplace = True)
        # 使用上一行的有效值进行填补
        dataTbl.fillna(method = 'ffill', inplace = True)
        # 如果填补后仍然有NaN，可以再使用0进行填补
        dataTbl.fillna('0', inplace = True)
        if isNeedRevert == True:
            dataTbl = dataTbl.T
        dataSet = dataTbl.values
        dataSet = dataSet[dataStartRank : , : ]
        self.data = TransNumpyToFloatTorch(dataSet[ : , dataStartCol : ], True)
        dateList = [x[0] for x in dataSet[ : , dateCol : dateCol + 1].tolist()]
        self.date = torch.randn(self.data.shape[0])
        if dateCol != 0xFFFFFFFF:
            try:
                date = pd.to_datetime(dateList, format = "%Y-%m-%d").astype('int64').values / 1000000000
                torchDate = torch.IntTensor(date)
                self.date = torchDate
            except ValueError as e:
                dateList = [str(int(dateEle)) for dateEle in dateList]
                date = pd.to_datetime(dateList, format = "%Y%m%d").astype('int64').values / 1000000000
                torchDate = torch.IntTensor(date)
                self.date = torchDate

    def __len__(self):
        return [len(self.date), len(self.data)]

    def GetTorchData(self):
        #print(f"date = {self.date.shape}, data = {self.data.shape}")
        return [self.date, self.data]

    def PrintTorchData(self):
        print(f"data = {self.data}")
        print(f"date = {self.date}")

# CSV转Torch 数据类 end


# Torch转CSV 数据类 begin

class TransTorchToCsv:
    def __init__(self, filePath, torchData, codeFormat, csvDelimi):
        if os.path.exists(filePath):
            print(r'TransTorchToCsv path exist')
            return
        if len(torchData.shape) != 2:
            print(r'torchData shape = {torchData.shape}')
            return
        fileCsv = codecs.open(filePath, 'w+', codeFormat)  # 追加
        writer = csv.writer(fileCsv, delimiter = csvDelimi, quotechar = ' ', quoting = csv.QUOTE_MINIMAL)
        for rowData in torchData:
            writer.writerow(rowData.numpy())
        print("保存文件成功，处理结束")

# Torch转CSV 数据类 end

# BP 模型框架实现 begin

#调用方法如下例：
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载MNIST数据集
#trainData = torchvision.datasets.MNIST(root = './data', train = True, transform = torchvision.transforms.ToTensor(), download = True)
#verifyData = torchvision.datasets.MNIST(root = './data', train = False, transform = torchvision.transforms.ToTensor())
#verifyDataLoader = torch.utils.data.DataLoader(dataset = verifyData, batch_size = batchSize, shuffle = False)
# 实例化神经网络类
#model = HandleBpNetWorkProcess(taskName, trainData, layersDim, epochNum, learnRate, learnRateDecay, weightDecay, statPeriod, modulePath)
#model.VerifyBpNetWork(verifyDataLoader)

class BpNetWork(nn.Module):
    def __init__(self, taskName, layersDim):
        super(BpNetWork, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.device = device
        self.layers = nn.ModuleList([nn.Linear(layersDim[i - 1], size) for i, size in enumerate(layersDim) if i > 0]).to(self.device)
        #self.relu = nn.ReLU().to(self.device)
        self.leakRelu = nn.LeakyReLU(negative_slope = 0.01).to(self.device)
        self.loss = 1000
        self.varMinusRst = 1000

    def forward(self, inputData):
        outputData = inputData
        for layer in self.layers:
            #outputData = self.relu(layer(outputData)).to(self.device)
            outputData = self.leakRelu(layer(outputData)).to(self.device)
            #outputData = torch.sigmoid(layer(outputData)).to(self.device)
            #outputData = torch.tanh(layer(outputData)).to(self.device)
        return outputData

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def TrainNeuralNetWork(self, trainDataLoader, epochNum, statPeriod):
        # 训练模型
        totalStep = len(trainDataLoader)
        for epoch in range(epochNum):
            #print(type(trainDataLoader))
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.trainModule.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for i, (images, labels) in enumerate(trainDataLoader):
                # 将图像数据压缩为一维，并在GPU上运行
                #print(images.shape) = [100(batch), 1(labelDataDim), 28(trainRank), 28(trainCol)]
                images = images.reshape(-1, images.shape[2] * images.shape[3]).to(self.device)
                labels = labels.to(self.device)
                # 向前传递计算和计算损失
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                # 反向传播和优化
                self.optimizer.zero_grad()
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
                # 输出过程状态信息
                if (i + 1) % statPeriod == 0:
                    print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{self.loss}")
                    checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.trainModule.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                    self.varMinusRst = varMinusRst
                    self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break

    def VerifyBpNetWork(self, verifyDataLoader):
        # 测试模型, 切换模型到评估模式的函数
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in verifyDataLoader:
                images = images.reshape(-1, images.shape[2] * images.shape[3]).to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('测试集上的准确率为: {:.2f} %'.format(100 * correct / total))

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
        return output.cpu(), self.varMinusRst.item()

def SetTorchPrintPara():
    pd.set_option('display.max_rows', 10)  # 设置最大打印行数为10行
    pd.set_option('display.max_columns', 10)  # 设置最大打印列数为10列
    torch.set_printoptions(threshold = 10, linewidth = 10, edgeitems = 3)
    np.set_printoptions(threshold = 100, linewidth = 100, edgeitems = 10)

def HandleBpNetWorkProcess(taskName, trainData, layersDim, epochNum, learnRate, learnRateDecay, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化神经网络类
    model = BpNetWork(taskName, layersDim).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleBpNetWorkProcess path not exist, path = {modulePath}")
    # 定义损失函数和优化器
    #model.SetCriterion(nn.MSELoss())
    model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = (learnRate * (1 - learnRateDecay)), weight_decay = weightDecay))
    trainDataLoader = torch.utils.data.DataLoader(dataset = trainData, batch_size = batchSize, shuffle = True)
    # 使用DataLoader进行批量训练
    #parallelModel = nn.DataParallel(model) #针对多块GPU处理
    #model = parallelModel.module
    model.TrainNeuralNetWork(trainDataLoader, epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# BP 模型框架实现 end

# Simple RNN 模型框架实现 begin

#调用方法如下例：
#trainData = torch.randn(trainDataNum, timeStep, trainDataDim)
#labelData = torch.LongTensor(range(0, trainDataNum))
#verifyData = trainData
# 实例化神经网络类
#model = HandleSimpleRnnNetWorkProcess(taskName, True, False, trainData, trainDataDim, labelData, labelDataDim, hiddenDim, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
#model.GetModuleCalcRst(verifyData)

class SimpleRnnNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedHidden, isOutPut, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, oriTrainNum = None, targetTrainNum = None):
        super(SimpleRnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.isNeedHidden = isNeedHidden
        self.isOutPut = isOutPut
        self.trainDataDim = trainDataDim
        self.labelDataDim = labelDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.relu = nn.ReLU()
        self.loss = 1000
        self.varMinusRst = 1000
        self.trainModule = nn.RNN(trainDataDim, hiddenDim, batch_first = isBatchFirst).to(self.device)
        self.fullConnectLayer = nn.Linear(hiddenDim, labelDataDim).to(self.device)
        '''
        if (isOutPut) and (oriTrainNum is not None) and (targetTrainNum is not None):
            #print(f"SimpleRnnNetWork labelDataDim = {labelDataDim}, oriTrainNum = {oriTrainNum}, targetTrainNum = {targetTrainNum}")
            self.seq_transform = nn.Linear(labelDataDim * oriTrainNum, labelDataDim * targetTrainNum)  # 新增的线性层
            self.oriTrainNum = oriTrainNum
            self.targetTrainNum = targetTrainNum
        '''

    def forward(self, inputData):
        #print(f"SimpleRnnNetWork forward inputData = {inputData.shape}, trainDataDim = {self.trainDataDim}, hiddenDim = {self.hiddenDim}, labelDataDim = {self.labelDataDim}")
        h0 = torch.zeros(self.layerNum, inputData.size(0), self.hiddenDim).to(self.device)
        out, hn = self.trainModule(inputData, h0)
        hn = hn[-1, : , : ].unsqueeze(1)
        hn = self.relu(hn)
        hn = self.fullConnectLayer(hn)
        return hn

    '''
        #print(f"SimpleRnnNetWork forward inputData = {inputData.shape}, trainDataDim = {self.trainDataDim}, hiddenDim = {self.hiddenDim}, labelDataDim = {self.labelDataDim}")
        h0 = torch.zeros(1, inputData.size(0), self.hiddenDim).to(self.device)
        out, _ = self.trainModule(inputData, h0)
        out = F.silu(out)
        out = self.fullConnectLayer(out)
        out = F.silu(out)
        #print(f"SimpleRnnNetWork forward out = {out.shape}, self.isOutPut = {self.isOutPut}, self.labelDataDim = {self.labelDataDim}")
        if self.isOutPut and hasattr(self, 'seq_transform'):
            # 将 (batch, seq_len, output_dim) 转换为 (batch, seq_len * output_dim)
            out = out.view(self.batchSize, -1)
            out = self.seq_transform(out)
            # 再将 (batch, oriTrainNum * output_dim) 转换为 (batch, targetTrainNum, labelDataDim)
            out = out.view(self.batchSize, self.targetTrainNum, -1)
        #print(f"SimpleRnnNetWork forward self.isOutPut = {self.isOutPut}, self.labelDataDim = {self.labelDataDim}, inputData = {inputData.shape}, out = {out.shape}")
        return out
    '''

    #def state_dict(self, destination = None, prefix='', keep_vars = False):
    #    state_dict = super(SimpleRnnNetWork, self).state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
    #    if hasattr(self, 'seq_transform'):
    #        state_dict[prefix + 'seq_transform.weight'] = self.seq_transform.weight
    #        state_dict[prefix + 'seq_transform.bias'] = self.seq_transform.bias
    #    return state_dict

    #def load_state_dict(self, state_dict, strict = True):
    #    super(SimpleRnnNetWork, self).load_state_dict(state_dict = state_dict, strict = strict)
    #    if ('seq_transform.weight' in state_dict) and (hasattr(self, 'seq_transform')):
    #        self.seq_transform.weight = state_dict['seq_transform.weight']
    #        self.seq_transform.bias = state_dict['seq_transform.bias']

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = True, shuffle = True)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.trainModule.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                #print(f"TrainNeuralNetWork trainData = {trainData.shape}, hidden = {self.hidden.shape}")
                self.optimizer.zero_grad()
                if(self.isNeedHidden):
                    self.hidden = self.hidden.detach()  # 分离隐藏状态
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                #print(f"TrainNeuralNetWork output = {output}, labelData = {labelData}")
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.trainModule.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                #print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleSimpleRnnNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, hiddenDim, layerNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = SimpleRnnNetWork(taskName, isBatchFirst, isNeedHidden, isOutput, trainData.shape[2], labelData.shape[2], hiddenDim, layerNum, batchSize).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        print(f"taskName = {taskName}, modulePath = {modulePath}, model.loss = {model.loss}, model.varMinusRst.item() = {model.varMinusRst.item()}")
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleSimpleRnnNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float())
    #parallelModel = nn.DataParallel(model) #针对多块GPU处理
    #model = parallelModel.module
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Simple RNN 模型框架实现 end

# Res RNN 模型框架实现 begin

class ResidualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layerNum = 1, isBatchFirst = True, isBidirectional = False, directionNum = 1):
        super(ResidualRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers = layerNum, bidirectional = isBidirectional, batch_first = isBatchFirst)
        self.linear = nn.Linear(input_size, hidden_size * directionNum)

    def forward(self, x, h0):
        residual = self.linear(x)
        out, hn = self.rnn(x, h0)
        return out + residual, hn

class ResidualRnnNetwork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedHidden, isOutPut, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, oriTrainNum = None, targetTrainNum = None):
        super(ResidualRnnNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.isNeedHidden = isNeedHidden
        self.isOutPut = isOutPut
        self.trainDataDim = trainDataDim
        self.labelDataDim = labelDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.relu = nn.ReLU()
        self.loss = 1000
        self.varMinusRst = 1000
        self.isBidirectional = False
        self.directionNum = 1
        if self.isBidirectional:
            self.directionNum = 2
        self.oriTrainNum = 0
        self.targetTrainNum = 0

        self.trainModule = ResidualRNN(trainDataDim, hiddenDim, layerNum, isBatchFirst, self.isBidirectional, self.directionNum).to(self.device)
        self.fullConnectLayer = nn.Linear(hiddenDim * self.directionNum, labelDataDim).to(self.device)
        self.dropout = nn.Dropout(dropRate)
        '''
        if (isOutPut) and (oriTrainNum is not None) and (targetTrainNum is not None):
            #print(f"ResidualRnnNetwork labelDataDim = {labelDataDim}, oriTrainNum = {oriTrainNum}, targetTrainNum = {targetTrainNum}")
            self.seq_transform = nn.Linear(labelDataDim * oriTrainNum, labelDataDim * targetTrainNum)  # 新增的线性层
            self.oriTrainNum = oriTrainNum
            self.targetTrainNum = targetTrainNum
        '''

    def forward(self, inputData, maskData = None):
        #print(f"ResidualRnnNetwork forward inputData = {inputData.shape}, trainDataDim = {self.trainDataDim}, hiddenDim = {self.hiddenDim}, labelDataDim = {self.labelDataDim}")
        batchSize = inputData.size(0)
        h0 = torch.zeros(self.layerNum * self.directionNum, batchSize, self.hiddenDim).to(self.device) #h0 2, 256, 500, inputData 256, 4, 300
        out, hn = self.trainModule(inputData, h0) #out 256, 4, 500, hn 2, 256, 500
        out = self.dropout(self.relu(out))
        out = out[ : , -1, : ].unsqueeze(1)
        out = self.fullConnectLayer(out)
        return out
    '''
        if self.isOutPut and hasattr(self, 'seq_transform'):
            # 将 (batch, seq_len, output_dim) 转换为 (batch, seq_len * output_dim)
            #print(f"out = {out.shape}, labelDataDim = {self.labelDataDim}, oriTrainNum = {self.oriTrainNum}, targetTrainNum = {self.targetTrainNum}")
            out = out.view(batchSize, -1)
            out = self.seq_transform(out)
            # 再将 (batch, oriTrainNum * output_dim) 转换为 (batch, targetTrainNum, labelDataDim)
            out = out.view(batchSize, self.targetTrainNum, -1)
        #print(f"ResidualRnnNetwork forward self.isOutPut = {self.isOutPut}, self.labelDataDim = {self.labelDataDim}, inputData = {inputData.shape}, out = {out.shape}")
        return out
    '''

    #def state_dict(self, destination = None, prefix='', keep_vars = False):
    #    state_dict = super(ResidualRnnNetwork, self).state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
    #    if hasattr(self, 'seq_transform'):
    #        state_dict[prefix + 'seq_transform.weight'] = self.seq_transform.weight
    #        state_dict[prefix + 'seq_transform.bias'] = self.seq_transform.bias
    #    return state_dict

    #def load_state_dict(self, state_dict, strict = True):
    #    super(ResidualRnnNetwork, self).load_state_dict(state_dict = state_dict, strict = strict)
    #    if ('seq_transform.weight' in state_dict) and (hasattr(self, 'seq_transform')):
    #        self.seq_transform.weight = state_dict['seq_transform.weight']
    #        self.seq_transform.bias = state_dict['seq_transform.bias']

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = True, shuffle = True)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.trainModule.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                #print(f"TrainNeuralNetWork trainData = {trainData.shape}, hidden = {self.hidden.shape}")
                self.optimizer.zero_grad()
                if(self.isNeedHidden):
                    self.hidden = self.hidden.detach()  # 分离隐藏状态
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                #print(f"TrainNeuralNetWork output = {output}, labelData = {labelData}")
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 10)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.trainModule.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleResidualRnnNetworkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, hiddenDim, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = ResidualRnnNetwork(taskName, isBatchFirst, isNeedHidden, isOutput, trainData.shape[2], labelData.shape[2], hiddenDim, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        print(f"taskName = {taskName}, modulePath = {modulePath}, model.loss = {model.loss}, model.varMinusRst.item() = {model.varMinusRst.item()}")
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleResidualRnnNetworkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.AdamW(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float())
    #parallelModel = nn.DataParallel(model) #针对多块GPU处理
    #model = parallelModel.module
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Res RNN 模型框架实现 end

# Total RNN 模型框架实现 begin

#调用方法如下例：
#trainData = torch.randn(trainDataNum, timeStep, trainDataDim)
#labelData = torch.LongTensor(range(0, trainDataNum))
#verifyData = trainData
# 实例化神经网络类
#model = HandleTotalRnnNetWorkProcess(taskName, True, trainData, trainDataDim, labelData, labelDataDim, hiddenDim, dropOut, epochNum, learnRate, weightDecay, statPeriod, modulePath)
#model.GetModuleCalcRst(verifyData)

class TotalRnnNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum):
        super(TotalRnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.labelDataDim = labelDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.Wxh = nn.ParameterList([nn.Parameter(torch.randn(trainDataDim if i == 0 else hiddenDim, hiddenDim)) for i in range(layerNum)])
        self.Whh = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.bh = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim)) for _ in range(layerNum)])
        self.loss = 1000
        self.varMinusRst = 1000
        if (labelDataDim != 0):
            self.fullConnectLayer = nn.Linear(hiddenDim, labelDataDim)  # 添加全连接层
        self.init_weights()
        self.to(self.device)  # 这将把所有的参数都移动到 self.device

    def init_weights(self):
        for i in range(self.layerNum):
            nn.init.xavier_uniform_(self.Wxh[i])
            nn.init.xavier_uniform_(self.Whh[i])
            nn.init.xavier_uniform_(self.bh[i])

    def forward(self, inputData, hidden=None):
        # print(f"TotalRnnNetWork inputData = {inputData.shape}, inputMask = {inputMask.shape}")
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
        else:
            hiddenStates = hidden
        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            inputStep = inputData[i]
            # 对于每一层 RNN，更新 hiddenStates和cellStates
            new_hiddenStates = []
            for layer in range(self.layerNum):
                hidden = hiddenStates[layer]
                new_hidden = torch.tanh(inputStep @ self.Wxh[layer] + hidden @ self.Whh[layer] + self.bh[layer])
                new_hiddenStates.append(new_hidden)
                inputStep = new_hidden  # 为下一层更新inputStep
            hiddenStates = new_hiddenStates  # 更新所有层的状态
            all_outputs.append(new_hiddenStates[-1])  # 记录最后一层的输出
        out = torch.stack(all_outputs, dim=0)
        hn = torch.stack(hiddenStates, dim=0)  # Shape: [num_layers, batch_size, hiddenDim]
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
        if (self.labelDataDim != 0):
            hn = self.fullConnectLayer(hn)
        # print(f"TotalRnnNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        # return out, hn
        return hn

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = inputData.shape[0], drop_last = True, shuffle = True)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                #print(f"TrainNeuralNetWork trainData = {trainData.shape}, hidden = {self.hidden.shape}")
                self.optimizer.zero_grad()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                #print(f"TrainNeuralNetWork output = {output}, labelData = {labelData}")
                loss.backward(retain_graph = True)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
        return output.cpu(), self.varMinusRst.item()

def HandleTotalRnnNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TotalRnnNetWork(taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, 1).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleTotalRnnNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float())
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Total RNN 模型框架实现 end

# Simple LSTM 模型框架实现 begin

#调用方法如下例：
# 定义 LSTM 模型并放到 GPU 上
#model = HandleSimpleLstmNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
#model.GetModuleCalcRst(verifyData)

class SimpleLstmNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedHidden, isOutPut, isBiDirectional, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate):
        super(SimpleLstmNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.isNeedHidden = isNeedHidden
        self.isOutPut = isOutPut
        self.trainDataDim = trainDataDim
        self.labelDataDim = labelDataDim
        self.hiddenDim = hiddenDim
        self.isBidirectional = isBiDirectional
        self.directionNum = 1
        if self.isBidirectional:
            self.directionNum = 2
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.loss = 1000
        self.varMinusRst = 1000

        self.trainModule = nn.LSTM(hiddenDim, hiddenDim, layerNum, bidirectional = self.isBidirectional, batch_first = isBatchFirst).to(self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropRate)
        self.input2HiddenDim = nn.Linear(trainDataDim, hiddenDim)
        self.direction2One = nn.Linear(self.layerNum * self.directionNum, 1)
        self.biHidden2HiddenDim = nn.Linear(self.directionNum * self.hiddenDim, self.hiddenDim)

    def forward(self, inputData, inputMask):
        #print(f"SimpleLstmNetWork inputData = {inputData.shape}")
        if (inputData.shape[2] == self.trainDataDim):
            inputData = self.input2HiddenDim(inputData)
        '''
        oriDataNum = 0
        if (inputData.shape[0] < 2 * self.batchSize):
            oriDataNum = inputData.shape[0]
            padInputData = torch.zeros(self.batchSize * 2 - inputData.shape[0], inputData.shape[1], inputData.shape[2]).to(self.device)
            inputData = AddDataToTorch(inputData, padInputData, 0)
        '''

        #print(f"SimpleLstmNetWork forward inputData = {inputData.shape}, trainDataDim = {self.trainDataDim}, hiddenDim = {self.hiddenDim}, labelDataDim = {self.labelDataDim}")
        batchSize = inputData.shape[0]
        h0 = torch.zeros(self.layerNum * self.directionNum, batchSize, self.trainModule.hidden_size).to(self.device)
        c0 = torch.zeros(self.layerNum * self.directionNum, batchSize, self.trainModule.hidden_size).to(self.device)
        out, (hn, cn) = self.trainModule(inputData, (h0, c0))

        hn = self.dropout(hn)
        cn = self.dropout(cn)
        out = self.dropout(out)
        if self.isBatchFirst:
            #out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
            cn = cn.transpose(0, 1)
        if (self.layerNum > 1) or (self.directionNum > 1):
            hn = hn.transpose(1, 2)
            hn = self.direction2One(hn)
            hn = hn.transpose(1, 2)

            cn = cn.transpose(1, 2)
            cn = self.direction2One(cn)
            cn = cn.transpose(1, 2)

        out = self.biHidden2HiddenDim(out)
        '''
        if (oriDataNum != 0):
            out = out[0 : oriDataNum, : , : ]
            hn = hn[0 : oriDataNum, : , : ]
            cn = cn[0 : oriDataNum, : , : ]
        '''
        #print(f"SimpleLstmNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        return out, (hn, cn)

    #def state_dict(self, destination = None, prefix='', keep_vars = False):
    #    state_dict = super(SimpleLstmNetWork, self).state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
    #    if hasattr(self, 'seq_transform'):
    #        state_dict[prefix + 'seq_transform.weight'] = self.seq_transform.weight
    #        state_dict[prefix + 'seq_transform.bias'] = self.seq_transform.bias
    #    return state_dict

    #def load_state_dict(self, state_dict, strict = True):
    #    super(SimpleLstmNetWork, self).load_state_dict(state_dict = state_dict, strict = strict)
    #    if ('seq_transform.weight' in state_dict) and (hasattr(self, 'seq_transform')):
    #        self.seq_transform.weight = state_dict['seq_transform.weight']
    #        self.seq_transform.bias = state_dict['seq_transform.bias']

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData, isNeedShuffle = False):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = False, shuffle = isNeedShuffle)
        #print(f"SetTrainDataInfo Dataset length: {len(self.dataloader.dataset)}")

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        isDataEmpty = True
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.trainModule.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                isDataEmpty = False
                #print(f"TrainNeuralNetWork trainData = {trainData.shape}, hiddenDim = {self.hiddenDim}")
                self.optimizer.zero_grad()
                if(self.isNeedHidden):
                    self.hidden = self.hidden.detach()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                #print(f"TrainNeuralNetWork output = {output}, labelData = {labelData}, loss = {loss}")
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
            if ((epoch + 1) % statPeriod == 0) and (not isDataEmpty):
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.trainModule.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                #print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")
        if not isDataEmpty:
            return self.loss
        return 0

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output, self.varMinusRst.item()

def HandleSimpleLstmNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, hiddenDim, layerNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"HandleSimpleLstmNetWorkProcess trainData = {trainData.shape}, labelData = {labelData.shape}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = SimpleLstmNetWork(taskName, isBatchFirst, isNeedHidden, isOutput, trainData.shape[2], labelData.shape[2], hiddenDim, layerNum, batchSize, trainData.shape[1], labelData.shape[1]).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        print(f"HandleSimpleLstmNetWorkProcess taskName = {taskName}, modulePath = {modulePath}, model.loss = {model.loss}, model.varMinusRst = {model.varMinusRst}")
        if (model.loss < 30) and (model.varMinusRst.item() < 100):
            return model
    print(f"HandleSimpleLstmNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float(), True)
    #parallelModel = nn.DataParallel(model)
    #model = parallelModel.module
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Simple LSTM 模型框架实现 end

# Total LSTM 模型框架实现 begin

#调用方法如下例：
#model = HandleTotalLstmNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, headNum, dropOut, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 定义 LSTM 模型并放到 GPU 上
#model.GetModuleCalcRst(verifyData)

class TotalLstmNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isOutput, isBidirectional, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, oriTrainNum = None, targetTrainNum = None):
        super(TotalLstmNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.isBidirectional = isBidirectional
        self.directions = 2 if isBidirectional else 1
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(trainDataDim if i == 0 else hiddenDim, hiddenDim)) for i in range(layerNum * self.directions)])
        self.Wf = nn.ParameterList([nn.Parameter(torch.randn(trainDataDim if i == 0 else hiddenDim, hiddenDim)) for i in range(layerNum * self.directions)])
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(trainDataDim if i == 0 else hiddenDim, hiddenDim)) for i in range(layerNum * self.directions)])
        self.Wg = nn.ParameterList([nn.Parameter(torch.randn(trainDataDim if i == 0 else hiddenDim, hiddenDim)) for i in range(layerNum * self.directions)])
        self.Ui = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum * self.directions)])
        self.Uf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum * self.directions)])
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum * self.directions)])
        self.Ug = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum * self.directions)])

        self.loss = 1000
        self.varMinusRst = 1000
        self.fullConnectLayer = nn.Linear(hiddenDim * self.directions, labelDataDim)
        self.dropout = nn.Dropout(dropRate)

        self.oriTrainNum = oriTrainNum
        self.targetTrainNum = targetTrainNum

        if (isOutput) and (oriTrainNum is not None) and (targetTrainNum is not None):
            #print(f"TotalLstmNetWork labelDataDim = {labelDataDim}, oriTrainNum = {oriTrainNum}, targetTrainNum = {targetTrainNum}")
            self.seq_transform = nn.Linear(labelDataDim * oriTrainNum * self.directions, labelDataDim * targetTrainNum)  # 新增的线性层
            self.oriTrainNum = oriTrainNum
            self.targetTrainNum = targetTrainNum

        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        for i in range(self.layerNum * self.directions):
            nn.init.xavier_uniform_(self.Wi[i])
            nn.init.xavier_uniform_(self.Wf[i])
            nn.init.xavier_uniform_(self.Wo[i])
            nn.init.xavier_uniform_(self.Wg[i])
            nn.init.xavier_uniform_(self.Ui[i])
            nn.init.xavier_uniform_(self.Uf[i])
            nn.init.xavier_uniform_(self.Uo[i])
            nn.init.xavier_uniform_(self.Ug[i])

    def forward(self, inputData, hidden = None):
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum * self.directions)]
            cellStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum * self.directions)]
        else:
            hiddenStates, cellStates = hidden

        # 正向和反向的输入数据
        forward_input = inputData
        if (self.isBidirectional):
            backward_input = inputData.flip(0)

        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            forward_step = forward_input[i]
            if (self.isBidirectional):
                backward_step = backward_input[i]

            # 对于每一层 LSTM，更新正向和反向的 hiddenStates 和 cellStates
            for layer in range(self.layerNum):
                # 正向
                forward_prev_hidden = hiddenStates[layer]
                forward_prev_cell = cellStates[layer]
                forward_cell_states = torch.sigmoid(forward_step @ self.Wi[layer] + forward_prev_hidden @ self.Ui[layer])
                forward_forget_gate = torch.sigmoid(forward_step @ self.Wf[layer] + forward_prev_hidden @ self.Uf[layer])
                forward_output_gate = torch.sigmoid(forward_step @ self.Wo[layer] + forward_prev_hidden @ self.Uo[layer])
                forward_choice_gate = torch.tanh(forward_step @ self.Wg[layer] + forward_prev_hidden @ self.Ug[layer])
                cellStates[layer] = forward_forget_gate * forward_prev_cell + forward_cell_states * forward_choice_gate
                hiddenStates[layer] = forward_output_gate * torch.tanh(cellStates[layer])
                forward_step = hiddenStates[layer]

                if (self.isBidirectional):
                    # 反向
                    backward_layer_index = layer + self.layerNum
                    backward_prev_hidden = hiddenStates[backward_layer_index]
                    backward_prev_cell = cellStates[backward_layer_index]
                    backward_cell_states = torch.sigmoid(backward_step @ self.Wi[backward_layer_index] + backward_prev_hidden @ self.Ui[backward_layer_index])
                    backward_forget_gate = torch.sigmoid(backward_step @ self.Wf[backward_layer_index] + backward_prev_hidden @ self.Uf[backward_layer_index])
                    backward_output_gate = torch.sigmoid(backward_step @ self.Wo[backward_layer_index] + backward_prev_hidden @ self.Uo[backward_layer_index])
                    backward_choice_gate = torch.tanh(backward_step @ self.Wg[backward_layer_index] + backward_prev_hidden @ self.Ug[backward_layer_index])
                    cellStates[backward_layer_index] = backward_forget_gate * backward_prev_cell + backward_cell_states * backward_choice_gate
                    hiddenStates[backward_layer_index] = backward_output_gate * torch.tanh(cellStates[backward_layer_index])
                    backward_step = hiddenStates[backward_layer_index]

            # 将正向和反向的输出拼接在一起
            if (self.isBidirectional):
                final_output = torch.cat((forward_step, backward_step), dim = 1)
            else:
                final_output = forward_step
            all_outputs.append(final_output)
            
        out = torch.stack(all_outputs, dim = 0)
        hn = torch.stack(hiddenStates, dim = 0)  # Shape: [num_layers * directions, batch_size, hiddenDim]
        cn = torch.stack(cellStates, dim = 0)  # Shape: [num_layers * directions, batch_size, hiddenDim]
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
            cn = cn.transpose(0, 1)
        out = self.dropout(out)
        if self.isOutput and hasattr(self, 'seq_transform'):
            # 将 (batch, seq_len, output_dim) 转换为 (batch, seq_len * output_dim)
            #print(f"TotalLstmNetWork out = {out.shape}, labelDataDim = {self.labelDataDim}, oriTrainNum = {self.oriTrainNum}, targetTrainNum = {self.targetTrainNum}")
            out = out.view(batchSize, -1)
            out = self.seq_transform(out)
            out = out.view(batchSize, self.targetTrainNum, -1)
        hn = self.fullConnectLayer(hn)
        #print(f"TotalLstmNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        return out, (hn, cn)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        # 将数据转换为 PyTorch Dataset 对象并放到 GPU 上
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        # 加载数据并自动进行批处理
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = False)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        # 模型训练
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        #模型预测
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleTotalLstmNetWorkProcess(taskName, isBatchFirst, isOutput, isBidirectional, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HandleTotalLstmNetWorkProcess trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = TotalLstmNetWork(taskName, isBatchFirst, isOutput, isBidirectional, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleTotalLstmNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Total LSTM 模型框架实现 end

# Mask Res 模型框架实现 begin

#调用方法如下例：
#model = HandleMaskResNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, headNum, dropOut, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 定义 MaskRes 模型并放到 GPU 上
#model.GetModuleCalcRst(verifyData)

class MaskResNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate):
        super(MaskResNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.trainDataDim = trainDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wg = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ui = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ug = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.loss = 1000
        self.varMinusRst = 1000

        self.transInputDataDim = nn.Linear(trainDataDim, hiddenDim)
        self.transDataDim = nn.Linear(hiddenDim, labelDataDim)
        self.transDataNum = nn.Linear(layerNum, 1)
        self.dropout = nn.Dropout(dropRate)
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        for i in range(self.layerNum):
            nn.init.xavier_uniform_(self.Wi[i])
            nn.init.xavier_uniform_(self.Wf[i])
            nn.init.xavier_uniform_(self.Wo[i])
            nn.init.xavier_uniform_(self.Wg[i])
            nn.init.xavier_uniform_(self.Ui[i])
            nn.init.xavier_uniform_(self.Uf[i])
            nn.init.xavier_uniform_(self.Uo[i])
            nn.init.xavier_uniform_(self.Ug[i])

    def forward(self, inputData, inputMask, hidden = None):
        #print(f"MaskResNetWork inputData = {inputData.shape}, inputMask = {inputMask.shape}, self.batchSize = {self.batchSize}")
        if (inputData.shape[2] == self.trainDataDim):
            inputData = self.transInputDataDim(inputData)
        #'''
        oriDataNum = 0
        if (inputData.shape[0] < self.batchSize * 2) or (inputMask.shape[0] < self.batchSize * 2):
            oriDataNum = inputData.shape[0]
            padInputData = torch.zeros(self.batchSize * 2 - inputData.shape[0], inputData.shape[1], inputData.shape[2]).to(self.device)
            padInputMask = torch.zeros(self.batchSize * 2 - inputMask.shape[0], inputMask.shape[1]).to(self.device)
            inputData = AddDataToTorch(inputData, padInputData, 0)
            inputMask = AddDataToTorch(inputMask, padInputMask, 0)
        #'''
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
            inputMask = inputMask.transpose(0, 1)
        reInputMask = torch.ones(inputMask.shape[0], inputMask.shape[1]).to(self.device) - inputMask
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
            cellStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
        else:
            hiddenStates, cellStates = hidden
        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            inputStep = inputData[i]
            maskStep = inputMask[i]
            reMaskStep = reInputMask[i]
            maskStep = maskStep.unsqueeze(1).expand(maskStep.shape[0], self.hiddenDim)
            #print(f"MaskResNetWork inputStep = {inputStep.shape}, maskStep = {maskStep.shape}")
            reverseMaskStep = reMaskStep.unsqueeze(1).expand(reMaskStep.shape[0], self.hiddenDim)
            # 对于每一层 LSTM，更新 hiddenStates 和 cellStates
            for layer in range(self.layerNum):
                curCellStates = nn.functional.gelu(inputStep @ self.Wi[layer] + hiddenStates[layer] @ self.Ui[layer])
                #outputGate = nn.functional.gelu(inputStep @ self.Wo[layer] + hiddenStates[layer] @ self.Uo[layer])
                cellStates[layer] = reverseMaskStep * cellStates[layer] + maskStep * curCellStates
                #hiddenStates[layer] = nn.functional.gelu(outputGate * cellStates[layer])
                hiddenStates[layer] = nn.functional.gelu(cellStates[layer])
                #hiddenStates[layer] = cellStates[layer]
                # 下一层的输入是当前层的输出
                inputStep = hiddenStates[layer]
            # 记录每个时间步的输出（使用最后一层的输出作为标准输出）
            final_output = hiddenStates[-1]
            all_outputs.append(final_output)
        out = torch.stack(all_outputs, dim = 0)
        hn = torch.stack(hiddenStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        cn = torch.stack(cellStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        hn = self.dropout(hn)
        out = self.dropout(out)
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
            cn = cn.transpose(0, 1)
        if (self.layerNum > 1):
            hn = hn.transpose(1, 2)
            hn = self.transDataNum(hn)
            hn = hn.transpose(1, 2)
        #out = self.transDataDim(out)
        #hn = self.transDataDim(hn)
        #cn = self.transDataDim(cn)
        #'''
        if (oriDataNum != 0):
            out = out[0 : oriDataNum, : , : ]
            hn = hn[0 : oriDataNum, : , : ]
            cn = cn[0 : oriDataNum, : , : ]
        #'''
        #print(f"MaskResNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        return out, (hn, cn)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        # 将数据转换为 PyTorch Dataset 对象并放到 GPU 上
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        # 加载数据并自动进行批处理
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = False)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        # 模型训练
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        #模型预测
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleMaskResNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = MaskResNetWork(taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleMaskResNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Mask Res 模型框架实现 end

# Mask LSTM 模型框架实现 begin

#调用方法如下例：
#model = HandleMaskLstmNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, headNum, dropOut, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 定义 MaskLstm 模型并放到 GPU 上
#model.GetModuleCalcRst(verifyData)

class MaskLstmNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate):
        super(MaskLstmNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.trainDataDim = trainDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wg = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ui = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ug = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.loss = 1000
        self.varMinusRst = 1000

        self.relu = nn.ReLU()
        self.leakRelu = nn.LeakyReLU(negative_slope = 0.01).to(self.device)
        self.transMask = nn.Linear(batchSize * 2, batchSize * 2)
        self.transInputDataDim = nn.Linear(trainDataDim, hiddenDim)
        self.transDataDim = nn.Linear(hiddenDim, labelDataDim)
        self.transDataNum = nn.Linear(layerNum, 1)
        self.dropout = nn.Dropout(dropRate)
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        for i in range(self.layerNum):
            nn.init.xavier_uniform_(self.Wi[i])
            nn.init.xavier_uniform_(self.Wf[i])
            nn.init.xavier_uniform_(self.Wo[i])
            nn.init.xavier_uniform_(self.Wg[i])
            nn.init.xavier_uniform_(self.Ui[i])
            nn.init.xavier_uniform_(self.Uf[i])
            nn.init.xavier_uniform_(self.Uo[i])
            nn.init.xavier_uniform_(self.Ug[i])

    def forward(self, inputData, inputMask, hidden = None):
        #print(f"MaskLstmNetWork inputData = {inputData.shape}, inputMask = {inputMask.shape}")
        if (inputData.shape[2] == self.trainDataDim):
            inputData = self.transInputDataDim(inputData)
        #'''
        oriDataNum = 0
        if (inputData.shape[0] < 2 * self.batchSize) or (inputMask.shape[0] < 2 * self.batchSize):
            oriDataNum = inputData.shape[0]
            padInputData = torch.zeros(self.batchSize * 2 - inputData.shape[0], inputData.shape[1], inputData.shape[2]).to(self.device)
            padInputMask = torch.zeros(self.batchSize * 2 - inputMask.shape[0], inputMask.shape[1]).to(self.device)
            inputData = AddDataToTorch(inputData, padInputData, 0)
            inputMask = AddDataToTorch(inputMask, padInputMask, 0)
        #'''
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
            inputMask = inputMask.transpose(0, 1)
        #inputMask = torch.sigmoid(self.transMask(inputMask))
        reInputMask = torch.ones(inputMask.shape[0], inputMask.shape[1]).to(self.device) - inputMask
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
            cellStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
        else:
            hiddenStates, cellStates = hidden
        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            inputStep = inputData[i]
            maskStep = inputMask[i]
            reMaskStep = reInputMask[i]
            maskStep = maskStep.unsqueeze(1).expand(maskStep.shape[0], self.hiddenDim)
            #print(f"MaskLstmNetWork maskStep = {maskStep}")
            reverseMaskStep = reMaskStep.unsqueeze(1).expand(reMaskStep.shape[0], self.hiddenDim)
            # 对于每一层 LSTM，更新 hiddenStates 和 cellStates
            for layer in range(self.layerNum):
                prev_hidden = hiddenStates[layer]
                #print(f"inputStep = {inputStep.shape}, prev_hidden = {prev_hidden.shape}, self.Wi[layer] = {self.Wi[layer].shape}, self.Ui[layer] = {self.Ui[layer].shape}")
                # nn.functional.gelu
                curCellStates = torch.tanh(inputStep @ self.Wi[layer] + prev_hidden @ self.Ui[layer])
                forgetGate = torch.sigmoid(inputStep @ self.Wf[layer] + prev_hidden @ self.Uf[layer])
                choiceGate = torch.sigmoid(inputStep @ self.Wg[layer] + prev_hidden @ self.Ug[layer])
                #outputGate = torch.sigmoid(inputStep @ self.Wo[layer] + prev_hidden @ self.Uo[layer])
                curCellStates = forgetGate * prev_hidden + curCellStates * choiceGate
                #curCellStates = outputGate * self.leakRelu(curCellStates)
                hiddenStates[layer] = torch.tanh(reverseMaskStep * hiddenStates[layer] + maskStep * curCellStates)
                # 下一层的输入是当前层的输出
                inputStep = hiddenStates[layer]
            # 记录每个时间步的输出（使用最后一层的输出作为标准输出）
            final_output = hiddenStates[-1]
            all_outputs.append(final_output)
        out = torch.stack(all_outputs, dim = 0)
        hn = torch.stack(hiddenStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        cn = torch.stack(cellStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        hn = self.dropout(hn)
        out = self.dropout(out)
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
            cn = cn.transpose(0, 1)
        if (self.layerNum > 1):
            hn = hn.transpose(1, 2)
            hn = self.transDataNum(hn)
            hn = hn.transpose(1, 2)
        #out = self.transDataDim(out)
        #hn = self.transDataDim(hn)
        #cn = self.transDataDim(cn)
        #'''
        if (oriDataNum != 0):
            out = out[0 : oriDataNum, : , : ]
            hn = hn[0 : oriDataNum, : , : ]
            cn = cn[0 : oriDataNum, : , : ]
        #'''
        #print(f"MaskLstmNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        return out, (hn, cn)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        # 将数据转换为 PyTorch Dataset 对象并放到 GPU 上
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        # 加载数据并自动进行批处理
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = False)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        # 模型训练
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        #模型预测
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleMaskLstmNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = MaskLstmNetWork(taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleMaskLstmNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Mask LSTM 模型框架实现 end

# Mem Rnn 模型框架实现 begin

#调用方法如下例：
#model = HandleMemRnnNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, headNum, dropOut, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 定义 MemRnn 模型并放到 GPU 上
#model.GetModuleCalcRst(verifyData)

class MemRnnNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedMaskMem, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate):
        super(MemRnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.trainDataDim = trainDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)]).to(self.device)
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)]).to(self.device)
        self.Ui = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)]).to(self.device)
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)]).to(self.device)
        self.loss = 1000
        self.varMinusRst = 1000

        #self.reslinear = nn.Linear(hiddenDim, hiddenDim)
        self.inputToHiddenDim = nn.Linear(trainDataDim, hiddenDim)
        #self.transDataDim = nn.Linear(hiddenDim, labelDataDim)
        self.transDataNum = nn.Linear(layerNum, 1)
        self.memMaskWi = None
        self.memMaskUi = None
        self.dropout = nn.Dropout(dropRate)
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        for i in range(self.layerNum):
            nn.init.xavier_uniform_(self.Wi[i])
            nn.init.xavier_uniform_(self.Wo[i])
            nn.init.xavier_uniform_(self.Ui[i])
            nn.init.xavier_uniform_(self.Uo[i])

    def forward(self, inputData, inputMask, hidden = None):
        #print(f"MemRnnNetWork inputData = {inputData.shape}, self.trainDataDim = {self.trainDataDim}, self.hiddenDim = {self.hiddenDim}, self.batchSize = {self.batchSize}")
        if (inputData.shape[2] == self.trainDataDim):
            inputData = self.inputToHiddenDim(inputData)
        '''
        oriDataNum = 0
        if (inputData.shape[0] < self.batchSize * 2):
            oriDataNum = inputData.shape[0]
            padInputData = torch.zeros(self.batchSize * 2 - inputData.shape[0], inputData.shape[1], inputData.shape[2]).to(self.device)
            inputData = AddDataToTorch(inputData, padInputData, 0)
        '''
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
            #inputMask = inputMask.transpose(0, 1)
        #inputDataRes = self.reslinear(inputData)
        #reInputMask = torch.ones(inputMask.shape[0], inputMask.shape[1]).to(self.device) - inputMask
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
            #cellStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
        else:
            hiddenStates = hidden
        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            inputStep = inputData[i]
            #inputRes = inputDataRes[i]
            #maskStep = inputMask[i]
            #reMaskStep = reInputMask[i]
            #maskStep = maskStep.unsqueeze(1).expand(maskStep.shape[0], self.hiddenDim)
            #reverseMaskStep = reMaskStep.unsqueeze(1).expand(reMaskStep.shape[0], self.hiddenDim)
            #print(f"MemRnnNetWork inputStep = {inputStep.shape}")
            # 对于每一层 LSTM，更新 hiddenStates
            for layer in range(self.layerNum):
                if (self.memMaskWi is None) or ((self.memMaskUi is None)):
                    self.memMaskWi = torch.randn(inputStep.shape[0], 1).to(self.device)
                    self.memMaskUi = torch.randn(inputStep.shape[0], 1).to(self.device)
                else:
                    if (self.memMaskWi.shape[0] == self.memMaskUi.shape[0]):
                        if (self.memMaskWi.shape[0] < inputStep.shape[0]) and (inputStep.shape[0] % self.memMaskWi.shape[0] == 0):
                            self.memMaskWi = self.memMaskWi.repeat(int(inputStep.shape[0] / self.memMaskWi.shape[0]), 1)
                            self.memMaskUi = self.memMaskUi.repeat(int(inputStep.shape[0] / self.memMaskUi.shape[0]), 1)
                        elif (self.memMaskWi.shape[0] > inputStep.shape[0]) and (self.memMaskWi.shape[0] % inputStep.shape[0] == 0):
                            self.memMaskWi = self.memMaskWi[0 : inputStep.shape[0], : ]
                            self.memMaskUi = self.memMaskUi[0: inputStep.shape[0], :]
                        elif (self.memMaskWi.shape[0] != inputStep.shape[0]):
                            print(f"MemRnnNetWork inputStep = {inputStep.shape}, self.memMaskWi = {self.memMaskWi.shape}, self.memMaskUi = {self.memMaskUi.shape}")
                    else:
                        print(f"MemRnnNetWork inputStep = {inputStep.shape}, self.memMaskWi = {self.memMaskWi.shape}, self.memMaskUi = {self.memMaskUi.shape}")
                activeMemMaskWi = torch.sigmoid((inputStep.transpose(0, 1) @ self.memMaskWi).squeeze(1))
                activeMemMaskUi = torch.sigmoid((inputStep.transpose(0, 1) @ self.memMaskUi).squeeze(1))
                curCellStates = nn.functional.gelu(inputStep @ (activeMemMaskWi * self.Wi[layer]) + hiddenStates[layer] @ (activeMemMaskUi * self.Ui[layer]))
                curCellStates = torch.clamp(curCellStates, min = -1, max = 1)
                #cellStates[layer] = reverseMaskStep * cellStates[layer] + maskStep * curCellStates
                #outputGate = nn.functional.gelu(inputStep @ self.Wo[layer] + hiddenStates[layer] @ self.Uo[layer])
                #hiddenStates[layer] = nn.functional.gelu(outputGate * cellStates[layer])
                #hiddenStates[layer] = nn.functional.gelu(cellStates[layer])
                hiddenStates[layer] = nn.functional.gelu(curCellStates)
                # 下一层的输入是当前层的输出
                inputStep = hiddenStates[layer]
            # 记录每个时间步的输出（使用最后一层的输出作为标准输出）
            final_output = hiddenStates[-1]
            #final_output = final_output + inputRes
            all_outputs.append(final_output)
        out = torch.stack(all_outputs, dim = 0)
        hn = torch.stack(hiddenStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        hn = self.dropout(hn)
        out = self.dropout(out)
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
        if (self.layerNum > 1):
            hn = hn.transpose(1, 2)
            hn = self.transDataNum(hn)
            hn = hn.transpose(1, 2)
        #out = self.transDataDim(out)
        #hn = self.transDataDim(hn)
        #cn = self.transDataDim(cn)
        '''
        if (oriDataNum != 0):
            out = out[0 : oriDataNum, : , : ]
            hn = hn[0 : oriDataNum, : , : ]
        '''
        #print(f"MemRnnNetWork out = {out.shape}, hn = {hn.shape}")
        return out, (hn, None)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        # 将数据转换为 PyTorch Dataset 对象并放到 GPU 上
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        # 加载数据并自动进行批处理
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = False)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        # 模型训练
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                output = self.forward(trainData, None)
                loss = self.criterion(output, labelData)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        #模型预测
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleMemRnnNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = MemRnnNetWork(taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleMemRnnNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Mem Rnn 模型框架实现 end

# Mem Lstm 模型框架实现 begin

#调用方法如下例：
#model = HandleMemLstmNetWorkProcess(taskName, True, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, headNum, dropOut, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 定义 MemLstm 模型并放到 GPU 上
#model.GetModuleCalcRst(verifyData)

class MemLstmNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedMaskMem, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate):
        super(MemLstmNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.trainDataDim = trainDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Wg = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ui = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uf = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.Ug = nn.ParameterList([nn.Parameter(torch.randn(hiddenDim, hiddenDim)) for _ in range(layerNum)])
        self.loss = 1000
        self.varMinusRst = 1000

        self.relu = nn.ReLU()
        #self.reslinear = nn.Linear(hiddenDim, hiddenDim)
        self.inputToHiddenDim = nn.Linear(trainDataDim, hiddenDim)
        # self.transDataDim = nn.Linear(hiddenDim, labelDataDim)
        self.transDataNum = nn.Linear(layerNum, 1)
        self.memMask = None
        self.dropout = nn.Dropout(dropRate)
        self.init_weights()
        self.to(self.device)
        #self.transInputDataDim = nn.Linear(trainDataDim, hiddenDim)
        #self.transDataDim = nn.Linear(hiddenDim, labelDataDim)

    def init_weights(self):
        for i in range(self.layerNum):
            nn.init.xavier_uniform_(self.Wi[i])
            nn.init.xavier_uniform_(self.Wf[i])
            nn.init.xavier_uniform_(self.Wo[i])
            nn.init.xavier_uniform_(self.Wg[i])
            nn.init.xavier_uniform_(self.Ui[i])
            nn.init.xavier_uniform_(self.Uf[i])
            nn.init.xavier_uniform_(self.Uo[i])
            nn.init.xavier_uniform_(self.Ug[i])

    def forward(self, inputData, inputMask, hidden = None):
        #print(f"MemLstmNetWork inputData = {inputData.shape}, self.trainDataDim = {self.trainDataDim}, self.hiddenDim = {self.hiddenDim}, self.batchSize = {self.batchSize}")
        if (inputData.shape[2] == self.trainDataDim):
            inputData = self.inputToHiddenDim(inputData)
        '''
        oriDataNum = 0
        if (inputData.shape[0] < self.batchSize * 2):
            oriDataNum = inputData.shape[0]
            padInputData = torch.zeros(self.batchSize * 2 - inputData.shape[0], inputData.shape[1], inputData.shape[2]).to(self.device)
            inputData = AddDataToTorch(inputData, padInputData, 0)
        '''
        if self.isBatchFirst:
            inputData = inputData.transpose(0, 1)  # [seq_len, batch_size, input_size]
            #inputMask = inputMask.transpose(0, 1)
        #inputDataRes = self.reslinear(inputData)
        #reInputMask = torch.ones(inputMask.shape[0], inputMask.shape[1]).to(self.device) - inputMask
        seqLength, batchSize, _ = inputData.size()
        if hidden is None:
            hiddenStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
            cellStates = [torch.zeros(batchSize, self.hiddenDim).to(self.device) for _ in range(self.layerNum)]
        else:
            hiddenStates, cellStates = hidden
        all_outputs = []  # 用于存储每个时间步的输出
        for i in range(seqLength):
            inputStep = inputData[i]
            #inputRes = inputDataRes[i]
            #maskStep = inputMask[i]
            #reMaskStep = reInputMask[i]
            #maskStep = maskStep.unsqueeze(1).expand(maskStep.shape[0], self.hiddenDim)
            #reverseMaskStep = reMaskStep.unsqueeze(1).expand(reMaskStep.shape[0], self.hiddenDim)
            #print(f"MemLstmNetWork inputStep = {inputStep.shape}")
            # 对于每一层 LSTM，更新 hiddenStates 和 cellStates
            for layer in range(self.layerNum):
                if (self.memMask is None):
                    self.memMask = torch.randn(inputStep.shape[0], 1).to(self.device)
                else:
                    if (self.memMask.shape[0] < inputStep.shape[0]) and (inputStep.shape[0] % self.memMask.shape[0] == 0):
                        self.memMask = self.memMask.repeat(int(inputStep.shape[0] / self.memMask.shape[0]), 1)
                    elif (self.memMask.shape[0] > inputStep.shape[0]) and (self.memMask.shape[0] % inputStep.shape[0] == 0):
                        self.memMask = self.memMask[0 : inputStep.shape[0], : ]
                    elif (self.memMask.shape[0] != inputStep.shape[0]):
                        print(f"MemLstmNetWork inputStep = {inputStep.shape}, self.memMask = {self.memMask.shape}")
                activeMemMask = torch.sigmoid((inputStep.transpose(0, 1) @ self.memMask).squeeze(1))

                prev_hidden = hiddenStates[layer]
                #print(f"inputStep = {inputStep.shape}, prev_hidden = {prev_hidden.shape}, self.Wi[layer] = {self.Wi[layer].shape}, self.Ui[layer] = {self.Ui[layer].shape}")
                curCellStates = nn.functional.gelu(inputStep @ (activeMemMask * self.Wi[layer]) + prev_hidden @ (activeMemMask * self.Ui[layer]))
                curCellStates = torch.clamp(curCellStates, min = -1, max = 1)
                forgetGate = nn.functional.gelu(inputStep @ (activeMemMask * self.Wf[layer]) + prev_hidden @ (activeMemMask * self.Uf[layer]))
                forgetGate = torch.clamp(forgetGate, min = -1, max = 1)
                choiceGate = nn.functional.gelu(inputStep @ (activeMemMask * self.Wg[layer]) + prev_hidden @ (activeMemMask * self.Ug[layer]))
                choiceGate = torch.clamp(choiceGate, min = -1, max = 1)
                outputGate = nn.functional.gelu(inputStep @ (activeMemMask * self.Wo[layer]) + prev_hidden @ (activeMemMask * self.Uo[layer]))
                outputGate = torch.clamp(outputGate, min = -1, max = 1)
                cellStates[layer] = forgetGate * prev_hidden + curCellStates * choiceGate
                hiddenStates[layer] = outputGate * self.relu(cellStates[layer])
                # 下一层的输入是当前层的输出
                inputStep = hiddenStates[layer]
            # 记录每个时间步的输出（使用最后一层的输出作为标准输出）
            final_output = hiddenStates[-1]
            # final_output = final_output + inputRes
            all_outputs.append(final_output)
        out = torch.stack(all_outputs, dim = 0)
        hn = torch.stack(hiddenStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        cn = torch.stack(cellStates, dim = 0)  # Shape: [num_layers, batch_size, hiddenDim]
        hn = self.dropout(hn)
        out = self.dropout(out)
        if self.isBatchFirst:
            out = out.transpose(0, 1)
            hn = hn.transpose(0, 1)
            cn = cn.transpose(0, 1)
        if (self.layerNum > 1):
            hn = hn.transpose(1, 2)
            hn = self.transDataNum(hn)
            hn = hn.transpose(1, 2)
        #out = self.transDataDim(out)
        #hn = self.transDataDim(hn)
        #cn = self.transDataDim(cn)
        '''
        if (oriDataNum != 0):
            out = out[0 : oriDataNum, : , : ]
            hn = hn[0 : oriDataNum, : , : ]
            cn = cn[0 : oriDataNum, : , : ]
        '''
        #print(f"MemLstmNetWork out = {out.shape}, hn = {hn.shape}, cn = {cn.shape}")
        return out, (hn, cn)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        # 将数据转换为 PyTorch Dataset 对象并放到 GPU 上
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        # 加载数据并自动进行批处理
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = False)

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        # 模型训练
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                output = self.forward(trainData, None)
                loss = self.criterion(output, labelData)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break
        #print(f"TrainNeuralNetWork false stop, varThreshold = {varThreshold}")

    def GetModuleCalcRst(self, verifyData):
        #模型预测
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleMemLstmNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, trainDataDim = {trainDataDim}, labelDataDim = {labelDataDim}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = MemLstmNetWork(taskName, isBatchFirst, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, dropRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleMemLstmNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    model.TrainNeuralNetWork(epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Mem Lstm 模型框架实现 end

# Snn 模型框架实现 begin

#调用方法如下例：
# 创建数据集和数据加载器
#trainData = torch.randn(trainDataNum, timeStep, trainDataDim)
#labelData = torch.LongTensor(range(0, trainDataNum))
#model = HandleSnnNetWorkProcess(taskName, isBatchFirst, trainData, labelData, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, varThreshold, modulePath)
# 使用模型进行预测
#verifyData = trainData
#model.GetModuleCalcRst(verifyData)

class SnnNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedHidden, isOutPut, trainDataDim, labelDataDim, hiddenDim, layerNum, batchSize, pruneRate, oriTrainNum = None, targetTrainNum = None):
        super(SnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.isNeedHidden = isNeedHidden
        self.isOutPut = isOutPut
        self.trainDataDim = trainDataDim
        self.labelDataDim = labelDataDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.batchSize = batchSize
        self.relu = nn.ReLU()
        self.Wi = nn.Parameter(torch.randn(trainDataDim, hiddenDim))
        self.Wf = nn.Parameter(torch.randn(trainDataDim, hiddenDim))
        self.Wo = nn.Parameter(torch.randn(trainDataDim, hiddenDim))
        self.Wg = nn.Parameter(torch.randn(trainDataDim, hiddenDim))
        self.Ui = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        self.Uf = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        self.Uo = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        self.Ug = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        self.dataFreq = nn.Parameter(torch.randn(trainDataDim, trainDataDim))
        self.dataPhasePos = nn.Parameter(torch.randn(trainDataDim, trainDataDim))
        #self.hiddenFreq = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        #self.hiddenPhasePos = nn.Parameter(torch.randn(hiddenDim, hiddenDim))
        self.loss = 1000
        self.varMinusRst = 1000
        self.pruneRate = pruneRate
        self.fullConnectLayer = nn.Linear(hiddenDim, labelDataDim).to(self.device)
        self.init_weights()
        self.to(self.device)
        if (isOutPut) and (oriTrainNum is not None) and (targetTrainNum is not None):
            #print(f"SnnNetWork labelDataDim = {labelDataDim}, oriTrainNum = {oriTrainNum}, targetTrainNum = {targetTrainNum}")
            self.seq_transform = nn.Linear(labelDataDim * oriTrainNum, labelDataDim * targetTrainNum)  # 新增的线性层
            self.oriTrainNum = oriTrainNum
            self.targetTrainNum = targetTrainNum

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.Wf)
        nn.init.xavier_uniform_(self.Wo)
        nn.init.xavier_uniform_(self.Wg)
        nn.init.xavier_uniform_(self.Ui)
        nn.init.xavier_uniform_(self.Uf)
        nn.init.xavier_uniform_(self.Uo)
        nn.init.xavier_uniform_(self.Ug)

    def forward(self, inputData):
        if not IsTorchDataListEmpty([self.dataFreq, self.dataPhasePos]):
            if (self.isBatchFirst):
                # Transpose the input so that batchSize is the second dimension
                inputData = inputData.transpose(0, 1)  # New shape: [trainDataNum, batchSize, trainDataDim]
            trainDataNum, batchSize, _ = inputData.size()
            hiddenStates = torch.zeros(batchSize, self.hiddenDim).to(self.device)
            cellStates = torch.zeros(batchSize, self.hiddenDim).to(self.device)
            dataTimeMatrix = torch.sin(self.dataFreq * time.time() + self.dataPhasePos)
            #hiddenTimeMatrix = torch.sin(self.hiddenFreq * time.time() + self.hiddenPhasePos)
            hiddenStatesRst = None
            for i in range(trainDataNum):
                inputStep = inputData[i]
                curCellStates = torch.sigmoid(inputStep @ dataTimeMatrix @ self.Wi + hiddenStates @ self.Ui)
                forgetGate = torch.sigmoid(inputStep @ dataTimeMatrix @ self.Wf + hiddenStates @ self.Uf)
                outputGate = torch.sigmoid(inputStep @ dataTimeMatrix @ self.Wo + hiddenStates @ self.Uo)
                choiceGate = torch.tanh(inputStep @ dataTimeMatrix @ self.Wg + hiddenStates @ self.Ug)
                cellStates = forgetGate * cellStates + curCellStates * choiceGate
                hiddenStates = outputGate * torch.tanh(cellStates)
                tempHiddenStates = hiddenStates.unsqueeze(1)
                hiddenStatesRst = AddDataToTorch(hiddenStatesRst, tempHiddenStates, 1)
            out = F.silu(hiddenStatesRst)
            out = self.fullConnectLayer(out)
            out = F.silu(out)
            #print(f"SnnNetWork forward out = {out.shape}, self.isOutPut = {self.isOutPut}, self.labelDataDim = {self.labelDataDim}")
            if self.isOutPut and hasattr(self, 'seq_transform'):
                # 将 (batch, seq_len, output_dim) 转换为 (batch, seq_len * output_dim)
                out = out.view(self.batchSize, -1)
                out = self.seq_transform(out)
                # 再将 (batch, oriTrainNum * output_dim) 转换为 (batch, targetTrainNum, labelDataDim)
                out = out.view(self.batchSize, self.targetTrainNum, -1)
            #print(f"SnnNetWork forward self.isOutPut = {self.isOutPut}, self.labelDataDim = {self.labelDataDim}, inputData = {inputData.shape}, out = {out.shape}")
            return out
        else:
            return None

    #def state_dict(self, destination = None, prefix='', keep_vars = False):
    #    state_dict = super(SnnNetWork, self).state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
    #    if hasattr(self, 'seq_transform'):
    #        state_dict[prefix + 'seq_transform.weight'] = self.seq_transform.weight
    #        state_dict[prefix + 'seq_transform.bias'] = self.seq_transform.bias
    #    return state_dict

    #def load_state_dict(self, state_dict, strict = True):
    #    super(SnnNetWork, self).load_state_dict(state_dict = state_dict, strict = strict)
    #    if ('seq_transform.weight' in state_dict) and (hasattr(self, 'seq_transform')):
    #        self.seq_transform.weight = state_dict['seq_transform.weight']
    #        self.seq_transform.bias = state_dict['seq_transform.bias']

    def SetModel(self, model):
        self.model = model

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = True, shuffle = True)

    def TrainNeuralNetWork(self, epochNum, statPeriod, varThreshold):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                if(self.isNeedHidden):
                    self.hidden = self.hidden.detach()
                output = self.forward(trainData)
                loss = self.criterion(output, labelData)
                #print(f"TrainNeuralNetWork output = {output}, labelData = {labelData}")
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                #print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                self.varMinusRst = varMinusRst
                self.loss = loss.item()
                #PruneTrainModulePara(self, ['Wi', 'Wf', 'Wo', 'Wg', 'Ui', 'Uf', 'Uo', 'Ug', 'dataFreq', 'dataPhasePos', 'hiddenFreq', 'hiddenPhasePos'], self.pruneRate)
                #PruneTrainModulePara(self, ['Wi', 'Wf', 'Wo', 'Wg', 'Ui', 'Uf', 'Uo', 'Ug', 'dataFreq', 'dataPhasePos'], self.pruneRate)
            #if checkRst:
            #    print(f"TrainNeuralNetWork stop, varThreshold = {varThreshold}")
            #    break

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu(), self.varMinusRst.item()

def HandleSnnNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, hiddenDim, layerNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, varThreshold, pruneRate, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = SnnNetWork(taskName, isBatchFirst, isNeedHidden, isOutput, trainData.shape[2], labelData.shape[2], hiddenDim, layerNum, batchSize, pruneRate).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        print(f"taskName = {taskName}, modulePath = {modulePath}, model.loss = {model.loss}, model.varMinusRst.item() = {model.varMinusRst.item()}")
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleSnnNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float())
    #parallelModel = nn.DataParallel(model)
    #model = parallelModel.module
    model.TrainNeuralNetWork(epochNum, statPeriod, varThreshold)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Snn 模型框架实现 end

# RRNN 模型框架实现 begin

#调用方法如下例：
# 创建数据集和数据加载器
#MAX_SEQ_LEN = 3
#SPLIT_PART_NUM = 5
#CROSS_HANDLE_LENGTH = 1 / 3
#modulePath = './modulePara'
#trainData = torch.randn(TRAIN_BATCH_SIZE, TRAIN_DATA_NUM, TRAIN_DATA_DIM)
#labelData = torch.randn(TRAIN_BATCH_SIZE, LABEL_DATA_DIM)
#model = HandleRecurrentRnnNetWorkProcess("asdfasd", True, False, trainData, labelData, TRAIN_HIDDEN_DIM, MAX_SEQ_LEN, SPLIT_PART_NUM, CROSS_HANDLE_LENGTH, TRAIN_BATCH_SIZE, TRAIN_EPOCH_NUM, TRAIN_LEARN_RATE, TRAIN_WEIGHT_DECAY, TRAIN_STAT_PERIOD, modulePath)
# 使用模型进行预测
#verifyData = trainData
#model.GetModuleCalcRst(verifyData)

#'''

#from torch.utils.tensorboard import SummaryWriter
from models.encoders.RecurrentGRC import RecurrentGRC
from models.encoders.RecurrentGRCX import RecurrentGRCX
from fairseq.modules import (
    SequenceNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout

class RecurrentRnnNetWork(nn.Module):
    def __init__(self, taskName, isBatchFirst, isNeedHidden, isNeedMaskMem, isBidirectional, trainDataDim, trainDataNum, labelDataDim, hiddenDim, layerNum, maxSeqLen, splitPartNum, crossLenRate, maxLevelNum, manualSeed, batchSize, resDropRate, learnRate, weightDecay):
        super(RecurrentRnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.isBatchFirst = isBatchFirst
        self.isNeedHidden = isNeedHidden
        self.isNeedMaskMem = isNeedMaskMem
        self.isBidirectional = isBidirectional
        self.hidden_dim = hiddenDim
        self.trainDataNum = trainDataNum
        self.trainDataDim = trainDataDim
        self.labelDataDim = labelDataDim
        self.mergeHiddenDim = hiddenDim
        self.maxSeqLen = maxSeqLen
        self.splitPartNum = splitPartNum
        self.crossLenRate = crossLenRate
        self.maxLevelNum = maxLevelNum
        self.manualSeed = manualSeed
        self.batchSize = batchSize
        self.resDropRate = resDropRate
        self.learnRate = learnRate
        self.weightDecay = weightDecay

        self.layerNum = layerNum
        self.rootLevel = 0
        self.outputLevel = 1

        '''
        self.headNum = 20
        if (self.headNum != 0):
            self.multiheadAttention = MultiHeadAttention(self.mergeHiddenDim, self.headNum, resDropRate)
        '''

        '''
        self.mergeModuleDict = nn.ModuleDict()
        for i in range(int(self.maxLevelNum)):
            self.mergeModuleDict[str(i)] = MaskResNetWork(self.taskName, self.isBatchFirst, self.trainDataDim, self.labelDataDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        '''

        '''
        if (isNeedMaskMem):
            #self.handleModule = MemRnnNetWork(self.taskName, self.isBatchFirst, self.isNeedMaskMem, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
            self.handleModule = MemLstmNetWork(self.taskName, self.isBatchFirst, self.isNeedMaskMem, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        else:
            self.handleModule = SimpleLstmNetWork(self.taskName, self.isBatchFirst, False, False, self.isBidirectional, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        '''

        '''
        if (isNeedMaskMem):
            self.handleModule = SimpleLstmNetWork(self.taskName, self.isBatchFirst, False, False, True, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        else:
            self.handleModule = SimpleLstmNetWork(self.taskName, self.isBatchFirst, False, False, True, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        '''

        #'''
        if (isNeedMaskMem):
            #self.handleModule = MaskResNetWork(self.taskName, self.isBatchFirst, self.trainDataDim, self.labelDataDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
            self.handleModule = MaskLstmNetWork(self.taskName, self.isBatchFirst, self.trainDataDim, self.labelDataDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        else:
            #self.handleModule = MaskResNetWork(self.taskName, self.isBatchFirst, self.trainDataDim, self.labelDataDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
            self.handleModule = MaskLstmNetWork(self.taskName, self.isBatchFirst, self.trainDataDim, self.labelDataDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)
        #'''

        self.outputModule = SimpleLstmNetWork(self.taskName, self.isBatchFirst, False, False, self.isBidirectional, self.trainDataDim, self.mergeHiddenDim, self.mergeHiddenDim, self.layerNum, self.batchSize, self.resDropRate).to(self.device)

        #self.outResLinear = nn.Linear(self.trainDataDim, self.trainDataDim)
        self.outResLinear = BpNetWork(self.taskName, [self.trainDataDim, self.mergeHiddenDim, self.labelDataDim])
        self.attentionFc = nn.Linear(self.mergeHiddenDim, self.mergeHiddenDim)
        self.biHiddenToHiddenDim = nn.Linear(self.mergeHiddenDim * 2, self.mergeHiddenDim)
        self.hidden2LabelDim = nn.Linear(self.mergeHiddenDim, self.labelDataDim)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #torch.manual_seed(self.manualSeed)
        #manualSeed = int(random.random() * 10)
        #torch.manual_seed(manualSeed)
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}, self.batchSize = {self.batchSize}")
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = False, shuffle = False)

    def GetSubTrainData(self, startPos, endPos):
        finalData = None
        finalMaskData = None
        for trainData, labelData in self.dataloader:
            if (startPos < 0) or (endPos <= 0) or (startPos >= trainData.shape[1]):
                print(f"GetSubTrainData startPos <= 0 or endPos <= 0, startPos = {startPos}, endPos = {endPos}, trainData = {trainData.shape}")
                return finalData, finalMaskData
            if (endPos > trainData.shape[1]):
                endPos = trainData.shape[1]
            tempData = trainData[ : , startPos : endPos, : ]
            tempMaskData = labelData[ : , startPos : endPos]
            finalData = AddDataToTorch(finalData, tempData, 0)
            finalMaskData = AddDataToTorch(finalMaskData, tempMaskData, 0)
        return finalData, finalMaskData

    def SaveModuleWeight(self, modulePath):
        print(f"SaveModuleWeight modulePath = {modulePath}")
        torch.save(self, modulePath)

    def LoadModuleWeight(self, modulePath):
        print(f"LoadModuleWeight modulePath = {modulePath}")
        self = torch.load(modulePath)
        return self

    def forward(self, input, input_mask, isOutput = False):
        #print(f"RecurrentRnnNetWork input = {input.shape}, input_mask = {input_mask.shape}, self.isNeedMaskMem = {self.isNeedMaskMem}")
        #self.batchSize = input.shape[0]

        '''
        #reverse input process
        rInput = torch.flip(input, dims=[1])
        rInput_mask = torch.flip(input_mask, dims=[1])
        input = AddDataToTorch(input, rInput, 1)
        input_mask = AddDataToTorch(input_mask, rInput_mask, 1)
        '''

        #'''
        #ori data
        sequence = input
        self.SetTrainDataInfo(input, input_mask)
        #'''

        '''
        if (self.headNum != 0):
            attentionOut = self.multiheadAttention(sequence, sequence, sequence)
            #print(f"RecurrentRnnNetWork sequence = {sequence.shape}, attentionOut = {attentionOut.shape}")
            sequence = self.attentionFc(attentionOut)
        '''

        '''
        #no mask
        sequence = GenerateNormalizeRst(input)
        self.SetTrainDataInfo(sequence, input_mask)
        '''

        '''
        #ori mask method
        if(input.shape[1] > self.trainDataNum):
            print(f"input.shape[1] = {input.shape[1]}")
        if (input.shape[1] < self.trainDataNum):
            padInput = torch.randn(input.shape[0], self.trainDataNum - input.shape[1], input.shape[2]).to(self.device)
            padMask = torch.zeros(input_mask.shape[0], self.trainDataNum - input_mask.shape[1]).to(self.device)
            input = AddDataToTorch(input, padInput, 1)
            input_mask = AddDataToTorch(input_mask, padMask, 1)
        if (input.shape[1] > self.trainDataNum):
            input = input[ : , : self.trainDataNum, : ]
            input_mask = input_mask[ : , : self.trainDataNum]
        input = GenerateNormalizeMaskRst(input, input_mask)
        self.SetTrainDataInfo(input, input_mask)
        '''

        '''
        #Total LSTM mask method
        input = GenerateNormalizeMaskRst(input, input_mask)
        self.SetTrainDataInfo(input, input_mask)
        '''

        if (sequence.shape[0] <= self.batchSize):
            oriBatchSize = sequence.shape[0]
            sequence = AddDataToTorch(sequence, torch.zeros(self.batchSize - sequence.shape[0], sequence.shape[1], sequence.shape[2]).to(self.device), 0)
            input_mask = AddDataToTorch(input_mask, torch.zeros(self.batchSize - input_mask.shape[0], input_mask.shape[1]).to(self.device), 0)
            sequence = self.TwoTowerForardProcess(sequence, input_mask, isOutput)
            oriOutput = self.hidden2LabelDim(sequence)[0 : oriBatchSize, : , : ]
        else:
            #print(f"RecurrentRnnNetWork > self.batchSize = {self.batchSize}, sequence = {sequence.shape}, input_mask = {input_mask.shape}")
            startPos = 0
            endPos = self.batchSize
            oriOutput = None
            while (startPos < sequence.shape[0]):
                tempSeq = sequence[startPos : endPos, : , : ]
                tempSeqMask = input_mask[startPos : endPos, : ]
                tempOutput = self.TwoTowerForardProcess(tempSeq, tempSeqMask, isOutput)
                oriOutput = AddDataToTorch(oriOutput, tempOutput, 0)
                startPos = endPos
                endPos = endPos + self.batchSize
                if (endPos > sequence.shape[0]):
                    endPos = sequence.shape[0]
            oriOutput = self.hidden2LabelDim(oriOutput)
            #print(f"RecurrentRnnNetWork oriOutput = {oriOutput.shape}")
        oriOutput = torch.clamp(oriOutput, min = -1, max = 1)

        outputRes = self.outResLinear(input)
        output = oriOutput + torch.mean(outputRes, dim = 1).unsqueeze(1)
        output = torch.clamp(output, min = -1, max = 1)
        #make_dot(output, params=dict(self.named_parameters())).render("GetSplitIndexList", format="png")
        #print(f"RecurrentRnnNetWork output = {output.shape}, self.isNeedMaskMem = {self.isNeedMaskMem}")

        return {"sequence": input,
        "global_state": output,
        "input_mask": input_mask,
        "aux_loss": None}

    def TwoTowerForardProcess(self, sequence, seqMask, isOutput):
        #print(f"TwoTowerForardProcess in sequence = {sequence.shape}, seqMask = {seqMask.shape}, isOutput = {isOutput}")
        #maxSeqLen 10; splitPartNum 20;lra sequence[32, 882, 128]
        isCalRrnn = False
        batchSize, trainDataNum, trainDataDim = sequence.shape #batchSize 32;trainDataNum 882;trainDataDim 128
        oriBatchSize = batchSize

        while (sequence.shape[1] > self.maxSeqLen):
            batchSize, trainDataNum, trainDataDim = sequence.size() #sequence[32, 882, 128]
            if (trainDataNum % self.splitPartNum != 0):
                dataNumErr = ((trainDataNum // self.splitPartNum) * self.splitPartNum) + self.splitPartNum - trainDataNum #dataNumErr 18;trainDataNum 882
                trainDataNum = trainDataNum + dataNumErr #trainDataNum 900
                padData = torch.zeros(batchSize, dataNumErr, trainDataDim).to(self.device) #padData [32, 18, 128]
                padMask = torch.zeros(batchSize, dataNumErr).to(self.device)
                sequence = AddDataToTorch(sequence, padData, 1) #sequence [32, 900, 128]
                seqMask = AddDataToTorch(seqMask, padMask, 1)
                assert sequence.size() == (batchSize, trainDataNum, trainDataDim)
            newTrainDataNum = trainDataNum // self.splitPartNum #trainDataNum 900;newTrainDataNum 45
            sequence = sequence.view(batchSize, self.splitPartNum, newTrainDataNum, trainDataDim)  # sequence[32, 20, 45, 128]
            sequence = sequence.reshape(batchSize * self.splitPartNum, newTrainDataNum, trainDataDim)  # sequence[640, 45, 128]
            seqMask = seqMask.view(batchSize, self.splitPartNum, newTrainDataNum)
            seqMask = seqMask.reshape(batchSize * self.splitPartNum, newTrainDataNum)

        #sequence[12800, 3, 128]
        #print(f"TwoTowerForardProcess one sequence = {sequence.shape}")

        batchSize, trainDataNum, trainDataDim = sequence.size()
        self.trainDataNum = int(batchSize / oriBatchSize * trainDataNum)
        while (oriBatchSize != batchSize) or (not isCalRrnn):
            isCalRrnn = True
            #print(f"TwoTowerForardProcess sequence = {sequence.shape}")

            crossLen = int(self.crossLenRate * sequence.shape[1])
            if (crossLen > 0):
                #print(f"TwoTowerForardProcess tempSeq = {sequence.shape}, tempSeqMask = {seqMask.shape}")
                tempSeq = AddDataToTorch(sequence, sequence[ : , 0 : crossLen, : ], 1)
                tempSeq = AddDataToTorch(sequence[ : , sequence.shape[1] - crossLen - 1 : , : ], tempSeq, 1)
                tempSeqMask = AddDataToTorch(seqMask, seqMask[ : , 0 : crossLen], 1)
                tempSeqMask = AddDataToTorch(seqMask[ : , seqMask.shape[1] - crossLen - 1 : ], tempSeqMask, 1)
            else:
                tempSeq = sequence
                tempSeqMask = seqMask

            if (isOutput) and (tempSeq.shape[0] == oriBatchSize):
                out, (hn, _) = self.outputModule(tempSeq, tempSeqMask)
                sequence = hn
            else:
                out, (hn, _) = self.handleModule(tempSeq, tempSeqMask)  # sequence[12800, 3, 128]
                if (crossLen > 0):
                    # print(f"TwoTowerForardProcess crossLen = {crossLen}, out before = {out.shape}")
                    out = out[ : , crossLen : out.shape[1] - crossLen - 1, : ]
                    # print(f"TwoTowerForardProcess crossLen = {crossLen}, out after = {out.shape}")
                sequence = out
            sequence = torch.clamp(sequence, min = -1, max = 1)

            #print(f"TwoTowerForardProcess out = {out.shape}, hn = {hn.shape}, self.isBidirectional = {self.isBidirectional}, isOutput = {isOutput}")
            batchSize, trainDataNum, trainDataDim = sequence.size()  # sequence[12800, 1, 200]
            if (batchSize > oriBatchSize):
                assert (batchSize % self.splitPartNum == 0)
                sequence = sequence.view(int(batchSize / self.splitPartNum), self.splitPartNum, trainDataNum, trainDataDim)
                sequence = sequence.reshape(int(batchSize / self.splitPartNum), self.splitPartNum * trainDataNum, trainDataDim)
                seqMask = seqMask.view(int(batchSize / self.splitPartNum), self.splitPartNum, seqMask.shape[1])
                seqMask = seqMask.reshape(int(batchSize / self.splitPartNum), self.splitPartNum * seqMask.shape[2])
        #print(f"TwoTowerForardProcess out sequence = {sequence.shape}")
        return sequence

    def HandleSubSeqProcess(self, startPos, endPos, level):
        #with torch.autograd.detect_anomaly():
        trainDataNum = endPos - startPos
        #print(f"HandleSubSeqProcess level = {level}, trainDataNum = {trainDataNum}, startPos = {startPos}, endPos = {endPos}, self.splitPartNum = {self.splitPartNum}")
        if (trainDataNum > self.maxSeqLen) and (level < self.maxLevelNum):
            totalCalcSeq = None
            totalMask = None
            partLength = trainDataNum / self.splitPartNum
            extensionLength = partLength * (self.crossLenRate)
            for i in range(self.splitPartNum):
                segmentStart = startPos + i * partLength
                segmentEnd = startPos + (i + 1) * partLength
                extendedStart = math.ceil((segmentStart - extensionLength) if (i > 0) else segmentStart)
                extendedEnd = math.ceil(segmentEnd + extensionLength if (i < self.splitPartNum - 1) else segmentEnd)
                #print(f"HandleSubSeqProcess level = {level}, i = {i}, startPos = {startPos}, endPos = {endPos}, extendedStart = {extendedStart}, extendedEnd = {extendedEnd}")
                if (extendedStart < extendedEnd) and (extendedEnd <= endPos):
                    partCalcSeq = self.HandleSubSeqProcess(extendedStart, extendedEnd, level + 1)
                    #print(f"HandleSubSeqProcess merge level = {level}, i = {i}, startPos = {startPos}, endPos = {endPos}, extendedStart = {extendedStart}, extendedEnd = {extendedEnd}, offset = {extendedEnd - extendedStart}")
                    _, tempMaskData = self.GetSubTrainData(extendedStart, extendedEnd)
                    totalCalcSeq = AddDataToTorch(totalCalcSeq, partCalcSeq, addDim = 1)
                    #totalMask = AddDataToTorch(totalMask, tempMaskData[ : , -1].unsqueeze(1), addDim = 1)
            #moduleLevel = self.GetModuleLevel(totalCalcSeq.shape[1])
            totalMask = torch.ones(totalCalcSeq.shape[0], totalCalcSeq.shape[1]).to(self.device)
            #out = self.GetSubSeqModuleCalcRst(moduleLevel, totalCalcSeq, totalMask)
            #out = self.GetSubSeqModuleCalcRst(level, totalCalcSeq, totalMask)
            out = self.GetSubSeqModuleCalcRst(self.rootLevel, totalCalcSeq, totalMask)
            #print(f"HandleSubSeqProcess level = {level}, out = {out.shape}")
            return out
        else:
            #print(f"GetSubTrainData startPos = {startPos}, endPos = {endPos}")
            tempTrainData, tempMaskData = self.GetSubTrainData(startPos, endPos)
            #print(f"GetSubTrainData level = {level}, trainData = {trainData.shape}")
            #out = self.GetSubSeqModuleCalcRst(level, tempTrainData, tempMaskData)
            out = self.GetSubSeqModuleCalcRst(self.rootLevel, tempTrainData, tempMaskData)
            return out

    def GetModuleLevel(self, trainDataLen):
        splitLen = (1 / self.splitPartNum) * (1 + self.crossLenRate * 2)
        #print(f"GetModuleLevel self.trainDataNum = {self.trainDataNum}, trainDataLen = {trainDataLen}, splitLen = {splitLen}")
        #print(f"GetModuleLevel math.log(self.trainDataNum) = {math.log(self.trainDataNum)}, math.log(trainDataLen) = {math.log(trainDataLen)}, math.log(splitLen) = {math.log(splitLen)}")
        moduleId = int(math.log(trainDataLen) / math.log(splitLen)) - int(math.log(self.trainDataNum) / math.log(splitLen))
        if moduleId < 0:
            moduleId = 0
        moduleId = int(moduleId)
        #print(f"GetModuleLevel self.trainDataNum = {self.trainDataNum}, trainDataLen = {trainDataLen}, splitLen = {splitLen}, moduleId = {moduleId}")
        return moduleId

    def GetSubSeqModuleCalcRst(self, level, trainData, maskData):
        if (trainData is not None) and (level < self.maxLevelNum):
            #print(f"GetSubSeqModuleCalcRst moduleIndex = {moduleIndex}, levelIndex = {levelIndex}, self.trainDataNum = {self.trainDataNum}, trainData = {trainData.shape}")
            #print(f"GetSubSeqModuleCalcRst levelIndex = {levelIndex}, trainData = {trainData.shape}, self.labelDataNum = {self.labelDataNum}")
            #PrintModuleParaShape(handleSubSeqModule)
            #out, (hn, cn) = (self.mergeModuleDict[str(level)])(trainData, maskData)
            out, (hn, cn) = (self.handleModule)(trainData, maskData)
            #print(f"GetSubSeqModuleCalcRst levelIndex = {levelIndex}, trainData = {trainData.shape}, out = {out.shape} GetSubSeqModuleCalcRst stop end")
            return out
        print(f"GetSubSeqModuleCalcRst level = {level} error")
        return None

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        trainLoss = 0
        startTime = datetime.now()
        for epoch in range(epochNum):
            #print(f"TrainNeuralNetWork epoch = {epoch}, epochNum = {epochNum}")
            for trainData, labelData in self.dataloader:
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                if (len(trainData.shape) == 3) and ((len(labelData.shape) == 2) or (len(labelData.shape) == 3)):
                    output = self.forward(trainData)
                    loss = self.criterion(output, labelData)
                    loss.backward(retain_graph = True)
                    # 生成计算图并保存为PDF文件
                    #print(f"TrainNeuralNetWork named_parameters = {dict(self.named_parameters())}")
                    #dot = make_dot(loss, params = dict(self.named_parameters()))
                    #dot.render(f"/home/ubuntu/BlackOp/1.pdf")
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 10)
                    #for level, model in self.mergeModuleDict.items():
                    #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
                    #self.AddNoisyToDictGrad(self.mergeModuleDict)
                    #mergeParamDict = self.GetModuleDictParameters('merge', self.mergeModuleDict)
                    #self.PrintDictGrad('merge', mergeParamDict)
                    #print(f"mergeParamDict = {mergeParamDict}")
                    self.optimizer.step()
                    trainLoss = loss.item()
                    #if self.scheduler:
                    #    self.scheduler.step()
                    #print(f"RecurrentRnnNetWork TrainNeuralNetWork loss.item() = {loss.item()}")
            if (epoch + 1) % statPeriod == 0:
                #print(f"RecurrentRnnNetWork TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], trainLoss:{trainLoss}")
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], trainLoss: {trainLoss}")
                #print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                self.loss = loss.item()
                #pass
        endTime = datetime.now()
        print(f"time = {endTime - startTime}")
        return trainLoss

    def AddNoisyToDictGrad(self, dictVal):
        for level, model in dictVal.items():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * 1e-3
                    param.grad = param.grad + noise

    def GetModuleDictParameters(self, dictName, dictVal):
        allParams = {}
        for level, model in dictVal.items():
            for name, param in model.named_parameters():
                allParams[f"{dictName}Module_{level}_{name}"] = param
        return allParams

    def PrintDictGrad(self, dictName, paramDict):
        for key, parameters in paramDict.items():
            print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, parameters.shape = {parameters.shape}")
            if parameters.grad is not None:
                #grad = torch.autograd.grad(loss, parameters, retain_graph = True, allow_unused = True)
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, gradmean = {parameters.grad.mean()}, gradvar = {parameters.grad.var()}, max = {parameters.grad.max()}, min = {parameters.grad.min()}")
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, mean = {parameters.mean()}, var = {parameters.var()}, max = {parameters.max()}, min = {parameters.min()}")
            else:
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key} grad is None")

    def GetModuleCalcRst(self, verifyData, maskData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            falseLabel = torch.zeros(self.batchSize, self.labelDataNum, self.labelDataDim)
            self.SetTrainDataInfo(verifyData, falseLabel)
            output = self.forward(verifyData, maskData)
            #print(f"GetModuleCalcRst verifyData = {verifyData}, output = {output}")
            return output

def HandleRecurrentRnnNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isNeedMaskMem, isBidirectional, trainData, labelData, hiddenDim, maxSeqLen, splitPartNum, crossLenRate, batchSize, resDropRate, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    model = RecurrentRnnNetWork(taskName, isBatchFirst, isNeedHidden, isNeedMaskMem, isBidirectional, trainData.shape[2], labelData.shape[1], labelData.shape[2], hiddenDim, maxSeqLen, splitPartNum, crossLenRate, batchSize, resDropRate, learnRate, weightDecay)
    if os.path.exists(modulePath):
        model = model.LoadModuleWeight(modulePath)
        return model
    else:
        model.SetCriterion(nn.MSELoss())
        #model.SetCriterion(nn.CrossEntropyLoss())
        model.SetTrainDataInfo(trainData, labelData)
        model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
        #parallelModel = nn.DataParallel(model) #针对多块GPU处理
        #model = parallelModel.module
        startTime = datetime.now()
        model.TrainNeuralNetWork(epochNum, statPeriod)
        endTime = datetime.now()
        model.SaveModuleWeight(modulePath)
        print(f"HandleRecurrentRnnNetWorkProcess time = {endTime - startTime}")
        return model
#'''

# RRNN 模型框架实现 end

# Sads 模型框架实现 begin

#调用方法如下例：
# 创建数据集和数据加载器
#trainData = torch.randn(trainDataNum, timeStep, trainDataDim)
#labelData = torch.LongTensor(range(0, trainDataNum))
#model = HandleSadsNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, lobeLabelDim, hiddenDim, cacheSize, maxSeqLen, splitPartNum, crossLenRate, batchSize, epochNum, resDropRate, learnRate, weightDecay, statPeriod, varThreshold, pruneRate, modulePath)
# 使用模型进行预测
#verifyData = trainData
#model.GetModuleCalcRst(verifyData)

class TransDataDim(nn.Module):
    def __init__(self, inputDataDim, outputDataDim):
        super(TransDataDim, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.outputDataDim = outputDataDim
        self.linear = nn.Linear(inputDataDim, outputDataDim)

    def forward(self, inputData):
        # 调整形状为 [batchSize * inputDataNum, inputDataDim]
        #print(f"inputData = {inputData.shape}, continue = {inputData.is_contiguous()}")
        batchSize, inputDataNum, inputDataDim = inputData.shape
        inputData = inputData.view(-1, inputDataDim)
        # 应用线性层
        outputData = self.linear(inputData)
        # 调整形状为 [batchSize, inputDataNum, outputDataDim]
        outputData = outputData.view(batchSize, inputDataNum, self.outputDataDim)
        return outputData

class CircularBuffer:
    """A simple circular buffer with a fixed size to store temporal and parietal outputs."""
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.buffer = []

    def append(self, item):
        """Append new item into buffer, remove oldest if buffer is full."""
        if len(self.buffer) >= self.maxSize:
            self.buffer.pop(0)  # Remove the oldest item
        #print(f"item = {item.shape}")
        self.buffer.append(item)

    def get(self):
        """Return the buffer as a stacked tensor: [batchSize, buffDataNum, trainDataDim]."""
        #for data in self.buffer:
        #    print(f"data = {data.shape}")
        return torch.cat(self.buffer, dim = 1)  # [batchSize, buffDataNum, trainDataDim]

    def clear(self):
        self.buffer = []

from itertools import chain

class SadsNetWork(nn.Module):
    def __init__(self, config):
        super(SadsNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.taskName = config["taskName"]
        self.isBatchFirst = config["isBatchFirst"]
        self.isNeedHidden = config["isNeedHidden"]
        self.trainDataNum = config["trainDataNum"]
        self.trainDataDim = config["trainDataDim"]
        self.lobeLabelDim = config["lobeLabelDim"]
        self.labelDataDim = config["labelDataDim"]
        self.hiddenDim = config["hiddenDim"]
        self.layerNum = config["layerNum"]
        self.maxSeqLen = config["maxSeqLen"]
        self.splitPartNum = config["splitPartNum"]
        self.crossLenRate = config["crossLenRate"]
        self.maxLevelNum = config["maxLevelNum"]
        self.cacheSize = config["cacheSize"]
        self.maxLoopCount = config["maxLoopCount"]

        self.manualSeed = config["manualSeed"]
        self.batchSize = config["batchSize"]
        self.resDropRate = config["resDropRate"]
        self.learnRate = config["learnRate"]
        self.weightDecay = config["weightDecay"]

        self.loss = torch.tensor([[0.0]], requires_grad = True)
        self.lossAdjustNum = 2
        self.varMinusRst = 1000

        '''
        self.headNum = 20
        if (self.headNum != 0):
            self.multiheadAttention = MultiHeadAttention(self.hiddenDim, self.headNum, self.resDropRate)
        '''

        #'''
        #HGRC mask func
        self.RNN = S4DWrapper(config)
        self.initial_transform = nn.Linear(self.hiddenDim, self.hiddenDim)
        if config and "rvnn_norm" in config:
            self.norm = config["rvnn_norm"]
        else:
            self.norm = "layer"
        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(self.hiddenDim)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(self.hiddenDim)
        #'''

        #额叶
        self.frontalLobe = RecurrentRnnNetWork(self.taskName, self.isBatchFirst, self.isNeedHidden, False, True, self.lobeLabelDim, self.trainDataNum, self.lobeLabelDim, self.hiddenDim, self.layerNum, self.maxSeqLen, self.splitPartNum, self.crossLenRate, self.maxLevelNum, self.manualSeed, self.batchSize, self.resDropRate, self.learnRate, self.weightDecay).to(self.device)
        #颞叶
        self.temporalLobe = RecurrentRnnNetWork(self.taskName, self.isBatchFirst, self.isNeedHidden, True, False, self.lobeLabelDim, self.trainDataNum, self.lobeLabelDim, self.hiddenDim, self.layerNum, self.maxSeqLen, self.splitPartNum, self.crossLenRate, self.maxLevelNum, self.manualSeed, self.batchSize, self.resDropRate, self.learnRate, self.weightDecay).to(self.device)

        self.attentionFc = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.trainToLobeDim = TransDataDim(self.trainDataDim, self.lobeLabelDim).to(self.device)
        self.lobeToLabelDim = TransDataDim(self.lobeLabelDim, self.labelDataDim).to(self.device)
        #self.summarizeCacheNum = TransDataDim(self.cacheSize, 1).to(self.device)
        self.summarizeCacheNum = BpNetWork(self.taskName, [self.cacheSize, self.hiddenDim, 1])
        #self.outResLinear = nn.Linear(self.lobeLabelDim, self.labelDataDim)
        self.outResLinear = BpNetWork(self.taskName, [self.lobeLabelDim, self.hiddenDim, self.labelDataDim])

    def SetModel(self, model):
        self.model = model

    #'''
    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)
    #'''

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def SetTrainDataInfo(self, inputData, labelData):
        #print(f"SetTrainDataInfo inputData = {inputData.shape}, labelData = {labelData.shape}")
        manualSeed = int(random.random() * 10)
        torch.manual_seed(manualSeed)
        #print(f"SetRnnTrainData inputData = {inputData.shape}, labelData = {labelData.shape}")
        data = TensorDataset(inputData.detach().to(self.device), labelData.detach().to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, drop_last = False, shuffle = False)

    def SaveModuleWeight(self, modulePath):
        print(f"SaveModuleWeight modulePath = {modulePath}")
        torch.save(self, modulePath)

    def LoadModuleWeight(self, modulePath):
        print(f"LoadModuleWeight modulePath = {modulePath}")
        self = torch.load(modulePath)
        return self

    #def state_dict(self, destination = None, prefix='', keep_vars = False):
    #    state_dict = super(SadsNetWork, self).state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
    #    if hasattr(self, 'seq_transform'):
    #        state_dict[prefix + 'seq_transform.weight'] = self.seq_transform.weight
    #        state_dict[prefix + 'seq_transform.bias'] = self.seq_transform.bias
    #    return state_dict

    #def load_state_dict(self, state_dict, strict = True):
    #    super(SadsNetWork, self).load_state_dict(state_dict = state_dict, strict = strict)
    #    if ('seq_transform.weight' in state_dict) and (hasattr(self, 'seq_transform')):
    #        self.seq_transform.weight = state_dict['seq_transform.weight']
    #        self.seq_transform.bias = state_dict['seq_transform.bias']

    def forward(self, inputData, inputMask):
        inputData = inputData.to(self.device)
        inputMask = inputMask.to(self.device)

        #if (inputData.shape[0] > self.batchSize):
        #    print(f"SadsNetWork self.batchSize = {self.batchSize}, inputData = {inputData.shape}, inputMask = {inputMask.shape}")

        '''
        #no mask
        inputData = GenerateNormalizeRst(inputData)
        self.SetTrainDataInfo(inputData, inputMask)
        '''

        #'''
        #HGRC mask method
        inputData = self.RNN(inputData, inputMask)["sequence"]
        inputData = self.normalize(self.initial_transform(inputData))
        self.SetTrainDataInfo(inputData, inputMask)
        #'''

        '''
        if (self.headNum != 0):
            attentionOut = self.multiheadAttention(inputData, inputData, inputData)
            #print(f"RecurrentRnnNetWork sequence = {sequence.shape}, attentionOut = {attentionOut.shape}")
            inputData = self.attentionFc(attentionOut)
        '''

        #print(f"SadsNetWork inputData = {inputData.shape}")
        inputData = self.trainToLobeDim(inputData)
        #print(f"SadsNetWork inputData = {inputData.shape}")

        # 循环器输出
        #self.loopCount = int(self.loss.item() * self.lossAdjustNum)
        #self.loopCount = min(self.loopCount, self.maxLoopCount)
        self.loopCount = self.maxLoopCount
        #print(f"SadsNetWork self.loss = {self.loss}, self.loopCount = {self.loopCount}")

        #print(f"SadsNetWork self.loss = {self.loss}, loopCount = {loopCount}")
        # 创建缓存
        cacheBuff = CircularBuffer(self.cacheSize)
        #print(f"SadsNetWork self.loopCount = {self.loopCount}, self.cacheSize = {self.cacheSize}")

        # 额叶处理
        frontalOutput = self.frontalLobe(inputData, inputMask)['global_state']  # [batchSize, 1, hiddenDim]
        #print(f"SadsNetWork frontalOutput = {frontalOutput.shape}")
        cacheBuff.append(frontalOutput.unsqueeze(1))

        for index in range(self.loopCount):
            #print(f"index = {index}")
            # 颞叶处理
            if (index == 0):
                temporalOutput = self.temporalLobe(inputData, inputMask)['global_state']
            else:
                temporalOutput = self.GetCacheLobeCalRst(cacheBuff, self.temporalLobe)
            #print(f"SadsNetWork temporalOutput = {temporalOutput.shape}")
            cacheBuff.append(temporalOutput.unsqueeze(1))

            # 额叶处理
            if (index != self.loopCount - 1):
                frontalOutput = self.GetCacheLobeCalRst(cacheBuff, self.frontalLobe)
                #print(f"SadsNetWork frontalOutput = {frontalOutput.shape}")
                cacheBuff.append(frontalOutput.unsqueeze(1))

        # 额叶最后一次处理
        outputData = self.GetCacheLobeCalRst(cacheBuff, self.frontalLobe, True)
        #print(f"SadsNetWork outputData = {outputData.shape}")
        #outputData = self.lobeToLabelDim(outputData)
        outputRes = self.outResLinear(inputData)
        #print(f"SadsNetWork outputData = {outputData.shape}, outputRes = {outputRes.shape}")
        outputData = self.lobeToLabelDim(outputData) + torch.mean(outputRes, dim = 1).unsqueeze(1)
        #cacheBuff.clear()
        #print(f"SadsNetWork outputData = {outputData.shape}" + '\n')

        #print(f"SadsNetWork forward self.splitModuleDict end named_parameters = {dict(self.named_parameters())}")
        #make_dot(outputData, params=dict(self.named_parameters())).render("/home/ubuntu/BlackOp/outputData", format="png")
        #print(f"SadsNetWork forward outputData.requires_grad: {outputData.requires_grad}, outputData.grad_fn: {outputData.grad_fn}")
        #print(f"SadsNetWork outputData = {outputData.shape}")
        #dot = make_dot(outputData, params = dict(self.named_parameters()))
        #dot.render(f"/home/ubuntu/BlackOp/outputData_Rst")

        #print(f"SadsNetWork outputData = {outputData.is_contiguous()}")
        #outputData = outputData.permute(0, 2, 1).contiguous()
        #print(f"SadsNetWork outputData = {outputData.is_contiguous()}")
        #print(f"SadsNetWork outputData = {outputData.shape}")
        #outputData = self.lobeToLabelNum(outputData).permute(0, 2, 1)
        #print(f"SadsNetWork outputData = {outputData.shape}")

        #make_dot(outputData, params=dict(self.named_parameters())).render("GetSplitIndexList", format="png")
        #assert outputData.size() == (self.batchSize, self.labelDataNum, self.labelDataDim)
        global_state = outputData[ : , -1 , : ]

        return {"sequence": inputData,
        "global_state": global_state,
        "input_mask": inputMask,
        "aux_loss": None}

    def GetCacheLobeCalRst(self, cacheBuff, lobeModule, isOutput = False):
        inputData = cacheBuff.get()
        #print(f"GetCacheLobeCalRst inputData = {inputData.shape}")
        outputData = None
        if len(inputData.shape) == 4:
            inputData = inputData.permute(1, 0, 2, 3)
            for i in range(0, inputData.shape[0]):
                #print(f"inputData.requires_grad = {inputData.requires_grad}")
                sampleInput = inputData[i, : , : , : ]
                #dot = make_dot(sampleInput, params=dict(self.named_parameters()))
                #dot.render(f"/home/ubuntu/BlackOp/sampleInput{i}_Rst")
                #print(f"GetCacheLobeCalRst sampleInput = {sampleInput.shape}")
                sampleMask = torch.ones(sampleInput.shape[0], sampleInput.shape[1]).to(self.device)
                sampleOutput = lobeModule(sampleInput, sampleMask, isOutput)['global_state']
                #print(f"sampleInput = {sampleInput.shape}, sampleOutput = {sampleOutput.shape}")
                #print(f"sampleOutput.requires_grad = {sampleOutput.requires_grad}")
                #dot = make_dot(sampleOutput, params=dict(self.named_parameters()))
                #dot.render(f"/home/ubuntu/BlackOp/sampleOutput{i}_Rst")
                #print(f"GetCacheLobeCalRst sampleOutput = {sampleOutput.shape}")
                sampleOutput = sampleOutput.unsqueeze(0)
                #print(f"GetCacheLobeCalRst sampleOutput = {sampleOutput.shape}")
                outputData = AddDataToTorch(outputData, sampleOutput, 0)
                #print(f"GetCacheLobeCalRst outputData = {outputData.shape}")
            batchSize = outputData.shape[1]
            trainDataNum = outputData.shape[2]
            trainDataDim = outputData.shape[3]
            padOutputData = torch.zeros(self.cacheSize - outputData.shape[0], outputData.shape[1], outputData.shape[2], outputData.shape[3]).to(self.device)
            outputData = AddDataToTorch(outputData, padOutputData, 0)
            outputData = outputData.permute(1, 2, 3, 0)
            outputData = outputData.view(outputData.shape[0] * outputData.shape[1], outputData.shape[2], outputData.shape[3])
            outputData = self.summarizeCacheNum(outputData)
            outputData = outputData.squeeze(2)
            outputData = outputData.view(batchSize, trainDataNum, trainDataDim)
        else:
            print(f"GetCacheLobeCalRst inputData error, inputData = {inputData.shape}")
        return outputData

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        trainLoss = 0
        startTime = datetime.now()
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                if hasattr(self.frontalLobe, 'optimizer'):
                    self.frontalLobe.optimizer.zero_grad()
                if hasattr(self.temporalLobe, 'optimizer'):
                    self.temporalLobe.optimizer.zero_grad()
                if (len(trainData.shape) == 3) and (len(labelData.shape) == 3):
                    output = self.forward(trainData)
                    loss = self.criterion(output, labelData)
                    self.loss = loss
                    #torch.autograd.set_detect_anomaly(True)
                    loss.backward(retain_graph = True)

                    '''
                    mergeParamDict = self.frontalLobe.GetModuleDictParameters('frontalLobe', self.frontalLobe.mergeModuleDict)
                    self.frontalLobe.PrintDictGrad('frontalLobe', mergeParamDict)
                    mergeParamDict = self.temporalLobe.GetModuleDictParameters('temporalLobe', self.temporalLobe.mergeModuleDict)
                    self.temporalLobe.PrintDictGrad('temporalLobe', mergeParamDict)
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            print(f"TrainNeuralNetWork key = {name}, param = {param.grad.mean()}, gradvar = {param.grad.var()}, max = {param.grad.max()}, min = {param.grad.min()}")
                        else:
                            print(f"TrainNeuralNetWork key = {name}, param grad none")
                    '''

                    #torch.autograd.set_detect_anomaly(True)
                    # 生成计算图并保存为PDF文件
                    #print(f"TrainNeuralNetWork named_parameters = {dict(self.named_parameters())}")
                    #dot = make_dot(loss, params = dict(self.named_parameters()))
                    #dot.render(f"/home/ubuntu/BlackOp/loss_Rst")
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 10)
                    #torch.nn.utils.clip_grad_norm_(chain(self.parameters(), self.frontalLobe.parameters(), self.temporalLobe.parameters()), max_norm = 10)
                    self.optimizer.step()
                    trainLoss = loss.item()
                    #if self.scheduler:
                    #    self.scheduler.step()
                    #print(f"TrainNeuralNetWork loss.item() = {loss.item()}")
            if (epoch + 1) % statPeriod == 0:
                print(f"TrainNeuralNetWork taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], loss:{trainLoss}")
                #checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                #print(f"TrainNeuralNetWork TrainNeuralNetWork checkRst = {checkRst}, varMinusRst = {varMinusRst}")
                #self.varMinusRst = varMinusRst
                #self.loss = loss
        endTime = datetime.now()
        print(f"time = {endTime - startTime}")
        return trainLoss

    def PrintDictGrad(self, dictName, paramDict):
        for key, parameters in paramDict.items():
            print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, parameters.shape = {parameters.shape}")
            if parameters.grad is not None:
                #grad = torch.autograd.grad(loss, parameters, retain_graph = True, allow_unused = True)
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, gradmean = {parameters.grad.mean()}, gradvar = {parameters.grad.var()}, max = {parameters.grad.max()}, min = {parameters.grad.min()}")
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key}, mean = {parameters.mean()}, var = {parameters.var()}, max = {parameters.max()}, min = {parameters.min()}")
            else:
                print(f"TrainNeuralNetWork dictName = {dictName}, key = {key} grad is None")

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyData = verifyData.to(self.device)
            output = self.forward(verifyData)
            return output.cpu()

def HandleSadsNetWorkProcess(taskName, isBatchFirst, isNeedHidden, isOutput, trainData, labelData, lobeLabelDim, hiddenDim, cacheSize, maxSeqLen, splitPartNum, crossLenRate, batchSize, epochNum, resDropRate, learnRate, weightDecay, statPeriod, varThreshold, pruneRate, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"trainData = {trainData.shape}, labelData = {labelData.shape}, hiddenDim = {hiddenDim}, batchSize = {batchSize}")
    model = SadsNetWork(taskName, isBatchFirst, isNeedHidden, trainData.shape[1], trainData.shape[2], labelData.shape[1], lobeLabelDim, labelData.shape[2], hiddenDim, cacheSize, maxSeqLen, splitPartNum, crossLenRate, batchSize, resDropRate, learnRate, weightDecay).to(device)
    if os.path.exists(modulePath):
        model.SaveModuleWeight(modulePath)
        print(f"taskName = {taskName}, modulePath = {modulePath}, model.loss = {model.loss.item()}, model.varMinusRst.item() = {model.varMinusRst.item()}")
        if (model.loss.item() < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleSadsNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData.float())
    #parallelModel = nn.DataParallel(model)
    #model = parallelModel.module
    model.TrainNeuralNetWork(epochNum, statPeriod, varThreshold)
    model.SaveModuleWeight(modulePath)
    return model

# Sads 模型框架实现 end

'''
# Temp HGRC begin

from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.encoders.S4DWrapper import S4DWrapper
from models.encoders.OrderedMemory import OrderedMemory
from models.encoders.RecurrentGRCX import RecurrentGRCX

class RecurrentRnnNetWork(nn.Module):
    def __init__(self, config):
        super(RecurrentRnnNetWork, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.model_chunk_size = config["model_chunk_size"]
        self.small_d = 64
        self.chunk_size = 30

        self.RNN = S4DWrapper(config)
        self.initial_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        if config and "rvnn_norm" in config:
            self.norm = config["rvnn_norm"]
        else:
            self.norm = "layer"

        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(self.hidden_dim)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(self.hidden_dim)

        self.GRC = RecurrentGRCX(config)

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)


    def forward(self, input, input_mask):

        sequence = self.RNN(input, input_mask)["sequence"]
        osequence = sequence.clone()
        oinput_mask = input_mask.clone()

        sequence = self.normalize(self.initial_transform(sequence))
        N, S, D = sequence.size()
        if not self.config["chunk_mode_inference"] and not self.training:
            self.chunk_size = S
        else:
            self.chunk_size = self.model_chunk_size

        while S > 1:
            N, S, D = sequence.size()
            if S >= (self.chunk_size + self.chunk_size // 2):
                if S % self.chunk_size != 0:
                    e = ((S // self.chunk_size) * self.chunk_size) + self.chunk_size - S
                    S = S + e
                    pad = T.zeros(N, e, D).float().to(sequence.device)
                    input_mask = T.cat([input_mask, T.zeros(N, e).float().to(sequence.device)], dim=-1)
                    sequence = T.cat([sequence, pad], dim=-2)
                    assert sequence.size() == (N, S, D)
                    assert input_mask.size() == (N, S)
                S1 = S // self.chunk_size
                chunk_size = self.chunk_size
            else:
                S1 = 1
                chunk_size = S
            sequence = sequence.view(N, S1, chunk_size, D)
            sequence = sequence.view(N * S1, chunk_size, D)

            input_mask = input_mask.view(N, S1, chunk_size)
            input_mask = input_mask.view(N * S1, chunk_size)

            N0 = N
            N, S, D = sequence.size()
            assert N == N0 * S1

            sequence = self.GRC(sequence, input_mask)["global_state"]
            assert sequence.size() == (N, D)
            sequence = sequence.view(N0, S1, D)
            input_mask = input_mask.view(N0, S1, chunk_size)[:, :, 0]
            S = S1
            N = N0

        assert sequence.size() == (N, 1, D)
        global_state = sequence.squeeze(1)

        return {"sequence": osequence,
                "global_state": global_state,
                "input_mask": oinput_mask,
                "aux_loss": None}

# Temp HGRC end
'''

# Mamba 模型框架实现 begin

class RMSNorm(nn.Module):
    def __init__(self, trainDataDim: int, eps: float = 1e-5, device: str ='cuda'):
         super().__init__()
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.eps = eps
         self.weight = nn.Parameter(torch.ones(trainDataDim, device = self.device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) * self.weight
        return output

class S6(nn.Module):
    def __init__(self, trainDataNum, trainDataDim, hiddenDim, batchSize):
        super(S6, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainDataNum = trainDataNum
        self.trainDataDim = trainDataDim
        self.hiddenDim = hiddenDim
        self.fc1 = nn.Linear(trainDataDim, trainDataDim, device = self.device)
        self.fc2 = nn.Linear(trainDataDim, hiddenDim, device = self.device)
        self.fc3 = nn.Linear(trainDataDim, hiddenDim, device = self.device)
        self.A = nn.Parameter(F.normalize(torch.ones(trainDataDim, hiddenDim, device = self.device), p = 2, dim = -1))
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(batchSize, self.trainDataNum, self.hiddenDim, device = self.device)
        self.C = torch.zeros(batchSize, self.trainDataNum, self.hiddenDim, device = self.device)
        self.delta = torch.zeros(batchSize, self.trainDataNum, self.trainDataDim, device = self.device)
        self.dA = torch.zeros(batchSize, self.trainDataNum, self.trainDataDim, self.hiddenDim, device = self.device)
        self.dB = torch.zeros(batchSize, self.trainDataNum, self.trainDataDim, self.hiddenDim, device = self.device)
        # h [batchSize, trainDataNum, trainDataDim, hiddenDim]
        self.h = torch.zeros(batchSize, self.trainDataNum, self.trainDataDim, self.hiddenDim, device = self.device)
        self.y = torch.zeros(batchSize, self.trainDataNum, self.trainDataDim, device = self.device)

    def Discretization(self):
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        self.Discretization()
        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            global curBatchSize
            curBatchSize = x.shape[0]
            if (curBatchSize > 0):
                if self.h.shape[0] != curBatchSize:
                    different_batch_size = True
                    h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[ : curBatchSize, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB
                else:
                    different_batch_size = False
                    h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB
                # y [batchSize, trainDataNum, trainDataDim]
                self.y = torch.einsum('bln,bldn->bld', self.C, h_new)
                global temp_buffer
                temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
                return self.y
            else:
                print(f"S6 curBatchSize = {curBatchSize}, x = {x.shape}")
        else:
            # h [batchSize, trainDataNum, trainDataDim, hiddenDim]
            h = torch.zeros(x.size(0), self.trainDataNum, self.trainDataDim, self.hiddenDim, device = x.device)
            y = torch.zeros_like(x)
            h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB
            # y [batchSize, trainDataNum, trainDataDim]
            y = torch.einsum('bln,bldn->bld', self.C, h)
            return y

class MambaBlock(nn.Module):
    def __init__(self, trainDataNum, trainDataDim, hiddenDim, batchSize, kernelSize, padNum):
        super(MambaBlock, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inp_proj = nn.Linear(trainDataDim, 2 * trainDataDim, device = self.device)
        self.out_proj = nn.Linear(2 * trainDataDim, trainDataDim, device = self.device)
        # For residual skip connection
        self.D = nn.Linear(trainDataDim, 2 * trainDataDim, device = self.device)
        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True
        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)
        self.S6 = S6(trainDataNum, 2 * trainDataDim, hiddenDim, batchSize)
        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(trainDataNum, trainDataNum, kernel_size = kernelSize, padding = padNum, device = self.device)
        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2 * trainDataDim, 2 * trainDataDim, device = self.device)
        # rmsnorm
        self.norm = RMSNorm(trainDataDim, device = self.device)

    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, trainDataNum, 2 * trainDataDim])
        x_conv.shape = torch.Size([batch_size, trainDataNum, 2 * trainDataDim])
        x_conv_act.shape = torch.Size([batch_size, trainDataNum, 2 * trainDataDim])
        """
        # Refer to Figure 3 in the MAMBA paper
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)
        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)
        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)
        return x_out

class MambaNetWork(nn.Module):
    def __init__(self, config):
        super(MambaNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.taskName = config["taskName"]
        self.batchSize = config["batchSize"]
        self.trainDataNum = config["trainDataNum"]
        self.trainDataDim = config["trainDataDim"]
        self.labelDataDim = config["labelDataDim"]
        self.hiddenDim = config["hiddenDim"]
        self.layerNum = config["layerNum"]
        self.resDropRate = config["resDropRate"]
        self.kernelSize = config["kernelSize"]
        self.padNum = config["padNum"]
        self.manualSeed = config["manualSeed"]

        global curBatchSize
        curBatchSize = 0
        global DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM
        DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0
        self.mambaBlock1 = MambaBlock(self.trainDataNum, self.trainDataDim, self.hiddenDim, self.batchSize, self.kernelSize, self.padNum)
        self.mambaBlock2 = MambaBlock(self.trainDataNum, self.trainDataDim, self.hiddenDim, self.batchSize, self.kernelSize, self.padNum)
        self.mambaBlock3 = MambaBlock(self.trainDataNum, self.trainDataDim, self.hiddenDim, self.batchSize, self.kernelSize, self.padNum)
        self.mambaTrainDataNum2One = nn.Linear(self.trainDataNum, 1)
        self.summerizeModule = SimpleLstmNetWork(self.taskName, True, False, False, False, self.trainDataDim, self.labelDataDim, self.hiddenDim, self.layerNum, self.batchSize, self.resDropRate)
        self.hidden2LabelDim = nn.Linear(self.hiddenDim, self.labelDataDim)

    def forward(self, sequence, input_mask):
        #print(f"MambaNetWork sequence = {sequence.shape}, input_mask = {input_mask.shape}")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        #'''
        #no mask
        inputData = sequence
        sequence = GenerateNormalizeRst(sequence)
        #'''

        mambaOutput = None
        while (sequence.shape[1] > self.trainDataNum):
            tempSeq = sequence[ : , 0 : self.trainDataNum, : ]
            sequence = sequence[ : , self.trainDataNum : , : ]
            #print(f"MambaNetWork tempSeq = {tempSeq.shape}")
            tempMambaOutput = self.HandleMambaBlock(tempSeq)
            #print(f"MambaNetWork tempMambaOutput = {tempMambaOutput.shape}")
            mambaOutput = AddDataToTorch(mambaOutput, tempMambaOutput, 1)
        if (sequence.shape[1] <= self.trainDataNum):
            sequence = self.pad_sequences_3d(sequence, self.trainDataNum)
            #print(f"MambaNetWork pad sequence = {sequence.shape}")
            tempMambaOutput = self.HandleMambaBlock(sequence)
            #print(f"MambaNetWork pad tempMambaOutput = {tempMambaOutput.shape}")
            mambaOutput = AddDataToTorch(mambaOutput, tempMambaOutput, 1)
        _, (finalOutput, _) = self.summerizeModule(mambaOutput, None)
        finalOutput = self.hidden2LabelDim(finalOutput)
        finalOutput = finalOutput.squeeze(1)

        return {"sequence": inputData,
        "global_state": finalOutput,
        "input_mask": input_mask,
        "aux_loss": None}

    def pad_sequences_3d(self, sequences, max_len = None, pad_value = 0):
        # Assuming sequences is a tensor of shape (batch_size, seq_len, feature_size)
        batch_size, seq_len, feature_size = sequences.shape
        if max_len is None:
            max_len = seq_len + 1
        # Initialize padded_sequences with the pad_value
        padded_sequences = torch.full((batch_size, max_len, feature_size), fill_value = pad_value, dtype = sequences.dtype, device = sequences.device)
        # Pad each sequence to the max_len
        padded_sequences[ : , : seq_len, : ] = sequences
        return padded_sequences
    def HandleMambaBlock(self, sequence):
        sequence = self.mambaBlock1(sequence)
        sequence = self.mambaBlock2(sequence)
        sequence = self.mambaBlock3(sequence)
        sequence = sequence.transpose(1, 2)
        sequence = self.mambaTrainDataNum2One(sequence)
        sequence = sequence.transpose(1, 2)
        return sequence

    def SetTrainDataInfo(self, inputData, labelData):
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = True)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def TrainNeuralNetWork(self, epochNum, statPeriod, maxGradNorm):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(trainData)
                loss = self.criterion(outputs.view(-1, self.labelDataDim), labelData.view(-1))
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                for name, param in model.named_parameters():
                    if 'out_proj.bias' not in name:
                        # clip weights but not bias for out_proj
                        torch.nn.utils.clip_grad_norm_(param, max_norm = maxGradNorm)
                if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
                    model.S6.h[ : curBatchSize, ...].copy_(temp_buffer)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()

def HandleMambaNetWorkProcess(taskName, trainData, labelData, hiddenDim, maxGradNorm, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    model = MambaNetWork(taskName, trainData.shape[2], labelData.shape[2], hiddenDim, batchSize)
    if os.path.exists(modulePath):
        model.LoadModuleWeight(modulePath)
        return model
    else:
        model.SetCriterion(nn.MSELoss())
        #model.SetCriterion(nn.CrossEntropyLoss())
        model.SetTrainDataInfo(trainData, labelData)
        model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
        #parallelModel = nn.DataParallel(model) #针对多块GPU处理
        #model = parallelModel.module
        startTime = datetime.now()
        model.TrainNeuralNetWork(epochNum, statPeriod, maxGradNorm)
        endTime = datetime.now()
        model.SaveModuleWeight(modulePath)
        print(f"HandleMambaNetWorkProcess time = {endTime - startTime}")
        return model

# Mamba 模型框架实现 end

# Transformer 模型框架实现 begin

#调用方法如下例：
# 创建数据集和数据加载器
#trainData = torch.randint(minTrainData, maxTrainData, (trainDataNum, trainDataDim), dtype = torch.int64)
#labelData = torch.randint(minTrainData, maxTrainData, (trainDataNum, trainDataDim), dtype = torch.int64)
#model = HandleTransformerNetWorkProcess(taskName, trainData, trainDataDim, labelData, labelDataDim, hiddenDim, layerNum, headNum, maxEncodeLen, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
# 使用模型进行预测
#testData = torch.randint(minTrainData, maxTrainData, (verifyDataNum, trainDataDim), dtype = torch.int64)
#model.GetModuleCalcRst(testData)

class TransformerNetWork(nn.Module):
    def __init__(self, taskName, trainDataDim, hiddenDim, labelDataDim, layerNum, headNum, maxEncodeLen, batchSize):
        super(TransformerNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.embedding = nn.Embedding(trainDataDim, hiddenDim).to(self.device)
        self.positionalEncode = PositionalEncoding(hiddenDim, maxEncodeLen).to(self.device)
        self.encoderLayer = nn.TransformerEncoderLayer(hiddenDim, headNum).to(self.device)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, layerNum).to(self.device)
        self.fullConnectLayer = nn.Linear(hiddenDim, labelDataDim).to(self.device)
        self.maxEncodeLen = maxEncodeLen
        self.labelDataDim = labelDataDim
        self.batchSize = batchSize
        self.loss = 1000
        self.varMinusRst = 1000

    def forward(self, x):
        x = self.embedding(x.to(self.device))
        x = self.positionalEncode(x)
        x = self.encoder(x)
        x = self.fullConnectLayer(x)
        return x

    def SetTrainDataInfo(self, inputData, labelData):
        data = TensorDataset(inputData.to(self.device), labelData.to(self.device))
        self.dataloader = DataLoader(data, batch_size = self.batchSize, shuffle = True)

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def TrainNeuralNetWork(self, epochNum, statPeriod):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for trainData, labelData in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(trainData)
                loss = self.criterion(outputs.view(-1, self.labelDataDim), labelData.view(-1))
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
            if (epoch + 1) % statPeriod == 0:
                print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                self.varMinusRst = varMinusRst
                self.loss = loss.item()

    def GetModuleCalcRst(self, verifyData):
        with torch.no_grad():
            verifyRst = self.forward(verifyData)
            predictedLabel = torch.argmax(verifyRst, dim = 2)
            print("Test Input:", verifyData.cpu())
            print("Predicted Label:", predictedLabel.cpu())

class PositionalEncoding(nn.Module):
    def __init__(self, hiddenDim, maxEncodeLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxEncodeLen, hiddenDim)
        position = torch.arange(0, maxEncodeLen).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, hiddenDim, 2) * -(math.log(10000.0) / hiddenDim))
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

def HandleTransformerNetWorkProcess(taskName, trainData, trainDataDim, labelData, labelDataDim, hiddenDim, layerNum, headNum, maxEncodeLen, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型和优化器
    model = TransformerNetWork(taskName, trainDataDim, hiddenDim, labelDataDim, layerNum, headNum, maxEncodeLen, batchSize).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleTransformerNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay))
    model.to(device)
    model.SetTrainDataInfo(trainData, labelData)
    # 使用Transformer模型进行训练
    model.TrainNeuralNetWork(epochNum, statPeriod)
    # 保存Transformer模型
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# Transformer 模型框架实现 end

# CNN 模型框架实现 begin

#调用方法如下例：
# 加载训练数据
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#trainDataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)
#trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True)
#testDataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform, download=True)
#testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle = False)
# 进行预测
#model = HandleCnnNetWorkProcess(taskName, trainDataset, trainMomentum, convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
#model.VerifyCnnNetWork(testLoader)

class CnnNetWork(nn.Module):
    def __init__(self, taskName, convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum):
        super(CnnNetWork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskName = taskName
        self.conv1 = nn.Conv2d(3, 16, kernel_size = convKernelSize, stride = convStride, padding = padNum).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.maxpool = nn.MaxPool2d(kernel_size = maxPoolKernelSize, stride = maxPoolStride).to(self.device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = convKernelSize, stride = convStride, padding = padNum).to(self.device)
         # 假设输出类别为10
        self.fullConnectLayer = nn.Linear(32 * 8 * 8, 10).to(self.device)
        self.loss = 1000
        self.varMinusRst = 1000

    def forward(self, inputData):
        inputData = self.conv1(inputData.to(self.device))
        inputData = self.relu(inputData)
        inputData = self.maxpool(inputData)
        inputData = self.conv2(inputData)
        inputData = self.relu(inputData)
        inputData = self.maxpool(inputData)
        inputData = inputData.view(inputData.size(0), -1)
        inputData = self.fullConnectLayer(inputData)
        return inputData

    def SetCriterion(self, func):
        self.criterion = func

    def SetOptimizer(self, func):
        self.optimizer = func

    def TrainNeuralNetWork(self, trainDataLoader, epochNum, statPeriod):
        for epoch in range(epochNum):
            initParam = {name: torch.zeros_like(param, device = self.device) for name, param in self.named_parameters()}
            lastAverage = {name: value.to(self.device) for name, value in initParam.items()}
            lastVar = {name: value.to(self.device) for name, value in initParam.items()}
            for i, (inputs, labels) in enumerate(trainDataLoader, 0):
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                #torch.autograd.set_detect_anomaly(True)
                loss.backward()
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.step()
                if (i + 1) % statPeriod == 0:
                    print(f"taskName = {self.taskName}, Epoch[{epoch + 1}/{epochNum}], self.loss:{self.loss}")
                    checkRst, lastAverage, lastVar, varMinusRst = IsParaVarBounded(self.taskName, dict(self.named_parameters()), lastAverage, lastVar, epoch, 0, self.device)
                    self.varMinusRst = varMinusRst
                    self.loss = loss.item()

    def VerifyCnnNetWork(self, verifyDataLoader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in verifyDataLoader:
                outputs = model(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.to(self.device).size(0)
                correct += (predicted == labels.to(self.device)).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")

def HandleCnnNetWorkProcess(taskName, trainDataset, trainMomentum, convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义损失函数和优化器
    model = CnnNetWork(taskName, convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum).to(device)
    if os.path.exists(modulePath):
        checkpoint = torch.load(modulePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.varMinusRst = checkpoint['varMinusRst']
        model.loss = checkpoint['loss']
        if (model.loss < 30) and (model.varMinusRst.item() < 20):
            return model
    print(f"HandleCnnNetWorkProcess path not exist, path = {modulePath}")
    model.SetCriterion(nn.MSELoss())
    #model.SetCriterion(nn.CrossEntropyLoss())
    model.SetOptimizer(optim.SGD(model.parameters(), lr = learnRate, momentum = trainMomentum, weight_decay = weightDecay))
    model.to(device)
    # 进行训练
    model.TrainNeuralNetWork(trainDataLoader, epochNum, statPeriod)
    torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, modulePath)
    return model

# CNN 模型框架实现 end

# MultiHeadAttention 模型框架实现 begin

class MultiHeadAttention(nn.Module):
    def __init__(self, hiddenDim, headNum, dropOutRate = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hiddenDim % headNum == 0, "hiddenDim must be divisible by headNum"
        self.hiddenDim = hiddenDim
        self.headNum = headNum
        self.headDim = hiddenDim // headNum
        self.query = nn.Linear(hiddenDim, hiddenDim)
        self.key = nn.Linear(hiddenDim, hiddenDim)
        self.value = nn.Linear(hiddenDim, hiddenDim)
        self.dropout = nn.Dropout(p = dropOutRate)
        self.fullConnectLayer = nn.Linear(hiddenDim, hiddenDim)

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        # Linear projection and split into heads
        query = self.query(query).view(batch_size, -1, self.headNum, self.headDim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.headNum, self.headDim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.headNum, self.headDim).transpose(1, 2)
        # Scaled Dot-Product Attention
        energy = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.headDim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(energy, dim = -1)
        attention = self.dropout(attention)
        # Combine the attention heads
        x = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.hiddenDim)
        out = self.fullConnectLayer(x)
        return out

# MultiHeadAttention 模型框架实现 end

def PrintTorchDataInfo(descStr, torchData):
    if (IsTorchDataEmpty(torchData)):
        print(f"{descStr}, torchData is None")
    else:
        print(f"{descStr}, torchData = {torchData}, shape = {torchData.shape}")

def PrintModuleParaShape(module):
    for name, param in module.named_parameters():
        print(f"{name}: {param.size()}")

def MergeTorchDataOutSizeUpLess(firstVec, firstData, secondVec, secondData, matchDim):
    if not CheckMatchTorchDataOutFormat(firstVec, firstData, secondVec, secondData, matchDim):
        return
    firstIndex = 0
    mergeData = torch.Tensor()
    for firstDataEle in torch.unbind(firstData, matchDim):
        secondIndex = FindLessValAfterStartIdxInSizeUpVec(firstVec[firstIndex], secondVec, secondVec.shape[0], secondIndex)
        if secondIndex != INVALID_NUM:
            tmpData = torch.cat((firstDataEle.unsqueeze(matchDim), secondData[secondIndex].unsqueeze(matchDim)), dim = 1)
        else:
            tmpData = torch.cat((firstDataEle.unsqueeze(matchDim), torch.zeros_like(secondData[0])).unsqueeze(matchDim), dim = 1)
        mergeData = AddDataToTorch(mergeData, tmpData, matchDim)
        firstIndex = firstIndex + 1
    return mergeData

def MergeTorchDataOutSizeDownLess(firstVec, firstData, secondVec, secondData, matchDim):
    if not CheckMatchTorchDataOutFormat(firstVec, firstData, secondVec, secondData, matchDim):
        print('MergeTorchDataOutSizeDownLess not CheckMatchTorchDataOutFormat')
        return
    firstIndex = 0
    mergeData = torch.Tensor()
    for firstDataEle in torch.unbind(firstData, matchDim):
        secondIndex = FindLessValAfterStartIdxInSizeDownVec(firstVec[firstIndex], secondVec, secondVec.shape[0])
        if secondIndex != INVALID_NUM:
            tmpData = torch.cat((firstDataEle.unsqueeze(matchDim), secondData[secondIndex].unsqueeze(matchDim)), dim = 1)
        else:
            tmpData = torch.cat((firstDataEle.unsqueeze(matchDim), torch.zeros_like(secondData[0])).unsqueeze(matchDim), dim = 1)
        mergeData = AddDataToTorch(mergeData, tmpData, matchDim)
        firstIndex = firstIndex + 1
    return mergeData

def CheckMatchTorchDataOutFormat(firstVec, firstData, secondVec, secondData, matchDim):
    if (not IsVector(firstVec)) or (not IsVector(secondVec)):
        print(f"CheckMatchTorchDataOutFormat firstVec.shape = {firstVec.shape}, secondVec.shape = {secondVec.shape}")
        return False
    if firstVec.shape[matchDim] != firstData.shape[matchDim] or secondVec.shape[matchDim] != secondData.shape[matchDim]:
        print(f"CheckMatchTorchDataOutFormat firstVec.shape[matchDim] = {firstVec.shape[matchDim]}, firstData.shape[matchDim] = {firstData.shape[matchDim]}")
        print(f"CheckMatchTorchDataOutFormat secondVec.shape[matchDim] = {secondVec.shape[matchDim]}, secondData.shape[matchDim] = {secondData.shape[matchDim]}")
        return False
    return True

def TransNumpyDataToFloat(numPyData):
    convertedData = []
    for row in numPyData:
        newRow = []
        for item in row:
            try:
                newRow.append(float(item))
            except ValueError:
                newRow.append(0.0)
        convertedData.append(newRow)
    npArray = np.array(convertedData)
    return npArray

def TransNumpyDataToInt(numPyData):
    convertedData = []
    for row in numPyData:
        newRow = []
        for item in row:
            try:
                newRow.append(int(item))
            except ValueError:
                newRow.append(0)
        convertedData.append(newRow)
    npArray = np.array(convertedData)
    return npArray

def TransNumpyToFloatTorch(numPyData, isTransStrToNum):
    if isTransStrToNum:
        npArray = TransNumpyDataToFloat(numPyData)
    tensorData = torch.from_numpy(npArray)
    return tensorData.float()

def TransNumpyToIntTorch(numPyData, isTransStrToNum):
    if isTransStrToNum:
        npArray = TransNumpyDataToInt(numPyData)
    tensorData = torch.from_numpy(npArray)
    return tensorData.int()

def IsVector(torchData):
    if len(torchData.shape) != 1:
        print(f"IsVector len(torchData.shape) = {len(torchData.shape)}")
        return False
    if torchData.shape[0] == 0:
        print(f"IsVector torchData.shape = {torchData.shape}")
        return False
    return True

def IsMatrix(torchData):
    if len(torchData.shape) != 2:
        print(f"IsMatrix len(torchData.shape) = {len(torchData.shape)}")
        return False
    if torchData.shape[0] == 0 or torchData.shape[1] == 0:
        print(f"IsMatrix torchData.shape[0] = {torchData.shape[0]}, torchData.shape[1] = {torchData.shape[1]}")
        return False
    return True

def AddDataToTorch(oriData, newData, addDim):
    if oriData is None:
        return newData
    if newData is None:
        return oriData
    #print(f"oriData.shape = {oriData.shape}, newData.shape = {newData.shape}, addDim = {addDim}")
    if (len(oriData.shape) > addDim) and (len(newData.shape) > addDim):
        return torch.cat((oriData, newData), dim = addDim)
    else:
        return torch.cat((oriData, newData), dim = 0)

def GenerateRandInputLabelData(inputDataNum, inputDataDim, timeStep, labelDataDim):
    inputData = torch.randn(inputDataNum, timeStep, inputDataDim)
    labelData = torch.randn(inputDataNum, labelDataDim)
    return [inputData, labelData]

def IsTorchDataEmpty(torchData):
    if (not torchData is None) and (torchData.numel() != 0):
        return False
    return True

def IsTorchDataListEmpty(torchDataList):
    for torchData in torchDataList:
        if IsTorchDataEmpty(torchData):
            return True
    return False

def IsTorchDataValid(torchData):
    if IsTorchDataEmpty(torchData):
        return False
    if torch.isnan(tensor).any():
        return False
    return True

def SelectTargetValByCol(torchData, colIndex, targetVal):
    if len(torchData.shape) < 2:
        print(f"SelectTargetValByCol torchData.shape = {torchData.shape}")
        return torch.empty((0, torchData.size(1)))
    indexList = torch.where(torchData[:, colIndex] == targetVal)[colIndex]
    # 打印选择出的行数据
    rst = torch.empty((0, torchData.size(1)))
    for index in indexList:
        rowData = torchData[index]
        rst = torch.cat((rst, rowData.unsqueeze(0)), dim = 0)
    #print(f"SelectTargetValByCol rst = {rst}")
    return rst

def SaveSpecDecimalNumSize(data, size):
    return (data * torch.pow(torch.tensor(10), size)).round() / (torch.pow(torch.tensor(10), size))

def GenerateNormalizeRst(data):
    if len(data.shape) != 3:
        return data
    #dataMean = torch.mean(data, dim = (0, 1), keepdim = True)
    #dataStd = torch.std(data, dim = (0, 1), keepdim = True)
    dataMean = torch.mean(data, dim = tuple(range(data.dim())), keepdim = True)
    dataStd = torch.std(data, dim = tuple(range(data.dim())), keepdim = True)
    dataStd = torch.clamp(dataStd, min = 1e-8)
    normalizeRst = (data - dataMean) / dataStd
    return normalizeRst

def GenerateNormalizeMaskRst(data, mask):
    # 如果数据为空，返回空张量
    if len(data.shape) == 0:
        return torch.empty((0,))
    if mask is None:
        mask = torch.ones_like(data)
    # 将 mask 扩展维度，以便与 data 的形状对齐
    mask = mask.unsqueeze(-1)
    # 将 mask 作用到数据上，屏蔽无效部分
    masked_data = data * mask  # 利用广播机制，将 mask 的无效位置对应的数据置为 0
    # 计算有效元素的数量
    valid_count = mask.sum(dim=1, keepdim=True)  # 在 dataNum 维度上求和
    # 避免所有 mask 为 0 的情况，确保最小为 1 以避免除以 0
    valid_count = torch.clamp(valid_count, min=1)
    # 计算 batch 内有效数据的加权均值
    data_sum = masked_data.sum(dim=1, keepdim=True)
    data_mean = data_sum / valid_count
    # 计算 batch 内有效数据的加权标准差
    variance_sum = ((masked_data - data_mean) ** 2).sum(dim=1, keepdim=True)
    data_std = torch.sqrt(variance_sum / valid_count)
    # 避免标准差为 0 的情况
    data_std = torch.clamp(data_std, min=1e-8)
    # 标准化处理，并在最后对无效数据的位置保持为 0
    normalize_rst = (data - data_mean) / data_std
    normalize_rst = normalize_rst * mask  # 保持无效部分为 0

    return normalize_rst

def IsParaVarBounded(taskName, paraDict, lastAverage, lastVar, epoch, varThreshold, device):
    #print(f"IsParaVarBounded taskName = {taskName}, paraDict = {type(paraDict)}, lastAverage = {type(lastAverage)}, lastVar = {type(lastVar)}")
    curAverage = CalParaAverage(paraDict, lastAverage, epoch, device)
    curVar = CalParaVar(paraDict, lastAverage, lastVar, epoch, device)
    #print(f"IsParaVarBounded taskName = {taskName}, epoch = {epoch}, IsParaVarBounded var = {curVar}")
    lastVarSum = 0
    curVarSum = 0
    for lastName, lastParaMatrix in lastVar.items():
        lastVarSum = lastVarSum + torch.sum(lastParaMatrix)
    for curName, curParaMatrix in curVar.items():
        curVarSum = curVarSum + torch.sum(curParaMatrix)
    #if abs(curVarSum - lastVarSum) > varThreshold:
    #    print(f"IsParaVarBounded taskName = {taskName}, epoch = {epoch}, totalSumBias = {abs(curVarSum - lastVarSum)}")
    #    return False, curAverage, curVar, curVarSum - lastVarSum
    return True, curAverage, curVar, curVarSum - lastVarSum

def CalParaAverage(paraDict, lastAverage, epoch, device):
    if epoch > 0:
        curAverage = {name: eleVal1.to(device) - eleVal2.to(device) for (name, eleVal1), (_, eleVal2) in zip(paraDict.items(), lastAverage.items())}
        curAverage = {name: (1 / epoch) * eleVal.to(device) for name, eleVal in curAverage.items()}
        curAverage = {name: eleVal1.to(device) + eleVal2.to(device) for (name, eleVal1), (_, eleVal2) in zip(lastAverage.items(), curAverage.items())}
    else:
        curAverage = paraDict
    return curAverage

def CalParaVar(paraDict, lastAverage, lastVar, epoch, device):
    if epoch > 0:
        curVar = {name: ((epoch - 1) / epoch) * eleVal.to(device) for name, eleVal in lastVar.items()}
        var1 = {name: eleVal1.to(device) - eleVal2.to(device) for (name, eleVal1), (_, eleVal2) in zip(paraDict.items(), lastAverage.items())}
        var1 = {name: torch.square(eleVal.to(device)) for name, eleVal in var1.items()}
        var1 = {name: ((epoch - 1) / (epoch * epoch)) * eleVal.to(device) for name, eleVal in var1.items()}
        curVar = {name: eleVal1.to(device) + eleVal2.to(device) for (name, eleVal1), (_, eleVal2) in zip(curVar.items(), var1.items())}
    else:
        paraName = list(paraDict)
        paramDim = [(name, param.size()) for name, param in paraDict.items()]
        curVar = {name: torch.zeros(size).to(device) for name, size in paramDim}
    return curVar

def CatVecToTorchData(torchData, vec, catDim):
    torchVec = torch.unsqueeze(maskRst, dim = 1)
    torchData = torch.cat([torchData, torchVec], dim = catDim)
    return torchData

def PruneTrainModulePara(trainModule, paraNameList, pruneRate):
    # 获取模型的所有参数名称
    paraToPrune = [(trainModule, paraName) for paraName in paraNameList]
    # Execute L1 norm unstructured pruning with a pruning rate of 0.4 (retain 60% of weights)
    prune.global_unstructured(paraToPrune, pruning_method = prune.L1Unstructured, amount = pruneRate)
    # Make the pruning permanent if needed
    for module, name in paraToPrune:
        prune.remove(module, name)
    # Print pruned parameters to check the result after pruning
    #for name in paraNameList:
    #    param = getattr(trainModule, name)
    #    print(param)

def RollTorchData(torchData, rollNum):
    # 假定 torchData 是一个已经加载和处理的二维 PyTorch 张量
    rowNum = torchData.shape[0]  # 获取张量的行数
    # 将所有的行向下循环移动 rollNum 位
    # 创建一个索引列表，表明每行新的位置
    indexList = torch.cat((torch.arange(rollNum, rowNum), torch.arange(0, rollNum)))
    # 使用这个索引列表来重新对张量进行索引，实现行的重新排列
    torchData = torchData[indexList]
    return torchData

def AddOptimizerToModule(oriModule, model, learnRate, weightDecay, isNeedAnneal = False, scheduler_type = 'cosine', min_lr = 1e-8, max_lr = 1e-4, T_max = 10):
    # 初始化优化器
    if not hasattr(oriModule, 'optimizer'):
        oriModule.optimizer = torch.optim.Adam(model.parameters(), lr = learnRate, weight_decay = weightDecay)
    else:
        oriModule.optimizer.add_param_group({'params': model.parameters(), 'weight_decay': weightDecay})
    if isNeedAnneal:
        # 初始化调度器
        if scheduler_type == 'cosine':
            # Cosine Annealing
            oriModule.scheduler = lr_scheduler.CosineAnnealingLR(oriModule.optimizer, T_max = T_max, eta_min = min_lr)
        elif scheduler_type == 'step':
            # StepLR
            oriModule.scheduler = lr_scheduler.StepLR(oriModule.optimizer, step_size = T_max // 2, gamma = 0.1)
        elif scheduler_type == 'exponential':
            # ExponentialLR
            oriModule.scheduler = lr_scheduler.ExponentialLR(oriModule.optimizer, gamma = 0.95)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return oriModule.optimizer, oriModule.scheduler
    else:
        return oriModule.optimizer, None

def SaveModuleDictPara(module, moduleDict, moduleName, modulePath):
    for key, model in moduleDict.items():
        savePath = os.path.join(modulePath, moduleName + f"{key}.pth")
        print(f"SaveModuleDictPara Saved weights for key {key} at {savePath}")
        if hasattr(module, 'varMinusRst') and hasattr(module, 'loss'):
            torch.save({'model_state_dict': model.state_dict(), 'varMinusRst': model.varMinusRst, 'loss': model.loss}, savePath)
        else:
            torch.save({'model_state_dict': model.state_dict()}, savePath)

def GumbelSoftmax(inputData, tau = 1, hard = False, eps = 1e-10, dim = -1):
    # 生成Gumbel噪声
    gumbels = -torch.empty_like(inputData).exponential_().log()
    gumbels = (inputData + gumbels) / tau
    ySoft = gumbels.softmax(dim)
    if hard:
        # Straight through.
        index = ySoft.max(dim, keepdim=True)[1]
        yHard = torch.zeros_like(inputData).scatter_(dim, index, 1.0)
        ret = yHard - ySoft.detach() + ySoft
        #print(f"GumbelSoftmax ret = {ret}, yHard = {yHard}, ySoft = {ySoft}")
    else:
        ret = ySoft
        #print(f"GumbelSoftmax ret = {ret}, ySoft = {ySoft}")
    return ret

def DifferentiableTopk(inputData, topNum, device):
    # 使用Gumbel-Softmax来近似top-k选择
    logits = torch.log(torch.abs(inputData) + 1e-20)  # 避免log(0)
    gumbelOut = GumbelSoftmax(logits, tau=0.1, hard = False)
    # 选择top-k个元素
    print(f"DifferentiableTopk gumbelOut = {gumbelOut}, topNum = {topNum}")
    _, topIndices = torch.topk(gumbelOut, topNum)
    # 创建一个mask
    mask = torch.zeros_like(inputData).to(device)
    mask.scatter_(0, topIndices, 1)
    # 应用mask到原始张量
    result = inputData * mask
    return result, topIndices

def SortTorchArray(torchArray):
    torchArrayLen = torchArray.shape[0]
    for i in range(torchArrayLen):
        for j in range(0, torchArrayLen - i - 1):
            if torchArray[j] > torchArray[j + 1]:
                # 使用加法和乘法来交换元素，这些操作支持自动微分
                torchArray[j] = torchArray[j] + torchArray[j + 1]
                torchArray[j + 1] = torchArray[j] - torchArray[j + 1]
                torchArray[j] = torchArray[j] - torchArray[j + 1]
    return torchArray


def DifferentiableSort(x, temperature=1.0):
    n = x.size(0)
    onehot = torch.eye(n, device=x.device)
    softmax = F.softmax(x.unsqueeze(-1) / temperature, dim=0)
    sorted_x = torch.mm(onehot, softmax).sum(dim=1)
    indices = sorted_x.argsort(descending=True)
    return indices

if __name__ == "__main__":
    # 定义超参的大小
    learnRate = 0.01
    weightDecay = 0.001
    batchSize = 64
    statPeriod = 100
    epochNum = 5
    trainMomentum = 0.9
    convKernelSize = 3
    convStride = 1
    maxPoolKernelSize = 2
    maxPoolStride = 2
    padNum = 1
    modulePath = "cnn_model.pth"

    # 加载训练数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainDataset = torchvision.datasets.CIFAR10(root='./data', train = True, transform = transform, download = True)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True)
    testDataset = torchvision.datasets.CIFAR10(root='./data', train = False, transform = transform, download=True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle = False)
    # 进行预测
    HandleCnnNetWorkProcess(trainDataset, trainMomentum, convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum, batchSize, epochNum, learnRate, weightDecay, statPeriod, modulePath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnNetWork(convKernelSize, convStride, maxPoolKernelSize, maxPoolStride, padNum).to(device)
    model.load_state_dict(torch.load(modulePath))
    model.VerifyCnnNetWork(testLoader)
