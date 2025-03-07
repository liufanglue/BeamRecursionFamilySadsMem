#-*- coding:utf-8 -*-
# coding:unicode_escape

import os
import sys
import random
import shutil
from pathlib import Path
from datetime import *
import time
import glob
import json
import math
import decimal
from enum import Enum
import numpy as np
import torch
import heapq

INVALID_NUM = 0xFFFFFFFF
SECOND_NUM_PER_MINUTE = 60
SECOND_NUM_PER_HOUR = 60 * SECOND_NUM_PER_MINUTE
SECOND_NUM_PER_DAY = 24 * SECOND_NUM_PER_HOUR
SECOND_NUM_PER_MONTH = 30 * SECOND_NUM_PER_DAY
SECOND_NUM_PER_YEAR = 365 * SECOND_NUM_PER_DAY
DAY_NUM_PER_MONTH = 30
DAY_NUM_PER_YEAR = 365
MONTH_NUM_PER_YEAR = 12

# 固定大小堆数据类 begin
class FixedSizeHeap:
    def __init__(self, max_size, key):
        self.max_size = max_size
        self.heap = []
        self.key = key

    def PushEle(self, item):
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            min_item = heapq.heappop(self.heap)
            if (self.key)(item, min_item):
                heapq.heappush(self.heap, item)
            else:
                heapq.heappush(self.heap, min_item)

    def PopEle(self):
        return heapq.heappop(self.heap)

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str(self.heap)

    def GetSizeDownEleList(self):
        return heapq.nlargest(len(self.heap), self.heap)

    def GetSizeUpEleList(self):
        return heapq.nsmallest(len(self.heap), self.heap)

# 固定大小堆数据类 end

def GetCurrentSecondNum():
    currentTime = time.time()
    return int(currentTime)

def GetCurrentDaySecondNum():
    currentTime = time.time()
    return int(currentTime) - (int(currentTime) % SECOND_NUM_PER_DAY)

def GetDayOffsetSecondNum(dayOffset):
    currentTime = datetime.now()  # 获取当前时间
    timeOffsetRst = currentTime + timedelta(days = dayOffset)  # 获取前一个月的时间
    return int(timeOffsetRst.timestamp())  # 获取前一个月同日期的秒数

def GetDayOffsetDaySecondNum(dayOffset):
    currentTime = datetime.now()
    timeOffsetRst = currentTime + timedelta(days = dayOffset)
    #print(f"GetDayOffsetDaySecondNum timeOffsetRst = {timeOffsetRst}")
    return int(timeOffsetRst.timestamp()) - (int(timeOffsetRst.timestamp()) % SECOND_NUM_PER_DAY)

def TransTimeOffsetToStr(dateOffSet):
    startDate = datetime.today() + timedelta(dateOffSet)
    return '{0:04}{1:02}{2:02}'.format(startDate.year, startDate.month, startDate.day)

def IsWinSys():
    if sys.platform.startswith('win'):
        return True
    return False

def IsFileExist(path):
    return os.path.isdir(path)

def IsDirExist(filePath):
    return os.path.exists(filePath)

def RemoveAllFile(filePath):
    files = os.listdir(filePath)
    for fileName in files:
        filePathName = filePath + fileName
        os.remove(filePathName)

def CopyAllFile(dstFilePath, oriFilePath):
    oriFiles = os.listdir(oriFilePath)
    for oriFileName in oriFiles:
        oriFilePathName = os.path.join(oriFilePath, oriFileName)
        dstFilePathName = os.path.join(dstFilePath, oriFileName)
        AdditionalWriteCsvFile(dstFilePathName, oriFilePathName)

def MergeCsvFile(dstFilePathName, oriFilePathName):
    with open(dstFilePathName, 'a') as dstFile:
        with open(oriFilePathName, 'r', encoding = 'utf_8_sig') as oriFile:
            pos = 0;
            lines = oriFile.readlines()
            for line in lines:
                if pos != 0:
                    dstFile.write(line)
                else:
                    pos = pos + 1

def MakeFilePath(filePath, rootPath = ""):
    if not os.path.exists(filePath):
        try:
            directoryPath = Path(filePath)
            directoryPath.mkdir(parents = True, exist_ok = True)
            print(f"Directory '{directoryPath}' created successfully.")
        except OSError as error:
            print(f"Error creating directory '{directory_path}': {error}")

def FindLessValAfterStartIdxInSizeUpVec(val, vec, vecSize, startIndex):
    for i in range(startIndex, vecSize):
        if vec[i] <= val:
            return i
    return INVALID_NUM

def FindLessValAfterStartIdxInSizeDownVec(val, vec, vecSize):
    for i in range(0, vecSize):
        #print(f"i = {str(i)}, vec[i] = {str(vec[i])}, val = {str(val)}")  
        if vec[i] <= val:
            return i
    return INVALID_NUM

def AdditionalWriteCsvFile(dstFilePathName, oriFilePathName):
    if not os.path.exists(oriFilePathName):
        print(f"AdditionalWriteCsvFile oriFilePathName not exist, oriFilePathName = {oriFilePathName}")
        return
    with open(oriFilePathName, 'a', encoding = 'utf_8_sig', errors = 'ignore') as oriFile:
        if os.path.exists(dstFilePathName):
            MergeCsvFile(dstFilePathName, oriFilePathName)
        else:
            shutil.copy(oriFilePathName, dstFilePathName)
            print(f"AdditionalWriteCsvFile dstFilePathName not exist, oriFilePathName = {dstFilePathName}")

def DeleteAllFiles(filePath):
    files = os.listdir(filePath)
    for file in files:
        fileName = os.path.join(filePath, file)
        if os.path.isdir(fileName):
            os.rmdir(fileName)
        else:
            os.remove(fileName)

def DeleteFileOrPath(filePath):
    if os.path.exists(filePath):
        if os.path.isdir(filePath):
            os.rmdir(filePath)
        else:
            os.remove(filePath)

def IsFindStrInList(eleList, targetStr):
    for ele in eleList:
        if ele.find(targetStr) != -1:
            return True
    return False

def GetFirstFindPosInStr(oriStr, findEleList, startPos):
    findPos = len(oriStr)
    findEle = ""
    for ele in findEleList:
        pos = oriStr.find(ele, startPos)
        if (pos != -1) and (pos < findPos):
            findPos = pos
            findEle = ele
    if ele == "":
        findPos = -1
    return findPos, findEle

def SplitStrToList(oriStr, delimList):
    splitRstList = []
    lastPos = 0
    findPos, findEle = GetFirstFindPosInStr(oriStr, delimList, lastPos)
    while findPos != -1:
        splitRstList.append(oriStr[lastPos : findPos - lastPos])
        lastPos = findPos + len(findEle)
        findPos, findEle = GetFirstFindPosInStr(oriStr, delimList, lastPos)
    if len(oriStr) - lastPos > 0:
        splitRstList.append(oriStr[lastPos : len(oriStr) - lastPos])    
    return splitRstList

def AppendListToStr(outputStr, oriList, delim = '\n'):
    if len(oriList) != 0:
        for oriEle in oriList:
            outputStr = outputStr + str(oriEle) + delim
    return outputStr

def Range(start, end, step):
    if (end - start) * step < 0:
        return []
    result = []
    value = start
    while value < end:
        result.append(value)
        value = value + step
    return result

def PreciseRange(start, end, step):
    if (end - start) * step < 0:
        return []
    result = []
    value = decimal.Decimal(str(start))
    end = decimal.Decimal(str(end))
    while value < end:
        result.append(value)
        value = value + decimal.Decimal(str(step))
    return result

def FloatRange(start, end, step):
    result = list()
    while (start < end):
        result.append(start)
        start = start + step
    return result

def RedirectPrintToFileBegin(fileNamePath):
    if os.path.exists(fileNamePath):
        os.remove(fileNamePath)
    logFileHandle = open(fileNamePath, 'w+')
    sys.stdout = logFileHandle
    return logFileHandle

def RedirectPrintToFileEnd(logFileHandle):
    sys.stdout = sys.__stdout__
    if not logFileHandle.closed:
        logFileHandle.close()

def GetPathFolderName(filePath):
    pos = filePath.rfind("\\")
    if pos != -1:
        fileName = filePath[pos + 1 : ]
        return fileName
    pos = filePath.rfind('/')
    if pos != -1:
        fileName = filePath[pos + 1 : ]
        return fileName
    return ""

def GetFileName(filePath):
    pos = filePath.rfind('/')
    fileName = filePath[pos + 1 : ]
    return fileName

def WriteFile(fileName, line):
    with open(fileName, 'a') as srcFile:
        srcFile.write(line)

def DeletefileInPath(folderPath):
    for root, dirList, fileList in os.walk(folderPath):
        for file in fileList:
            filePathName = os.path.join(root, file)
            os.remove(filePathName)
        for dir in dirList:
            dirPath = os.path.join(root, dir)
            os.rmdir(dirPath)

def HandleAllFileInDir(filePath, funcHandle, paraTuple):
    files = os.listdir(filePath)
    for file in files:
        pathName = os.path.join(filePath, file)
        if os.path.isdir(pathName):
            HandleAllFileInDir(pathName)
        else:
            filePathName = pathName
            funcHandle(filePathName, paraTuple)

def GetFileMaxLineCharNum(filePathName):
    try:
        with open(filePathName, "r", errors = 'ignore') as file:
            lineList = file.readlines()  # 读取整个文件并返回每一行的字符串列表
            maxLen = max(len(line) for line in lineList)
            return maxLen
    except Exception as e:
        print(f"ReadFileLineToList Cannot decode file {filePathName}")
        raise
    return 0

def ReadFileLineToList(filePathName):
    try:
        with open(filePathName, "r", errors = 'ignore') as file:
            lineList = file.readlines()  # 读取整个文件并返回每一行的字符串列表
            return lineList
    except Exception as e:
        print(f"ReadFileLineToList Cannot decode file {filePathName}")
        raise
    return []

def ReadFileContentToStr(filePathName):
    try:
        with open(filePathName, "r", errors = 'ignore') as file:
            contentStr = file.read()
            return contentStr
    except Exception as e:
        print(f"ReadFileContentToStr Cannot decode file {filePathName}")
        raise
    return ''

def ReadFileContentToJson(filePathName):
    try:
        with open(filePathName, 'r', errors = 'ignore') as file:
            # 使用json.load方法将文件内容加载成Python的字典或列表
            jsonData = json.load(filePathName)    
            return jsonData
    except Exception as e:
        print(f"ReadFileContentToJson Cannot decode file {filePathName}")
        raise
    return {}

def ReadStringToJson(oriStr):
    try:
        jsonData = json.loads(oriStr)
        return jsonData
    except Exception as e:
        print(f"ReadStringToJson Cannot decode file {oriStr}")
        raise
    return {}

def SplitStrToChunkList(text, chunkSize):
    # 将文本按照指定的长度进行切分
    chunkList = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunkSize]
        chunkList.append(chunk)
        start += chunkSize
    return chunkList

def SplitStrToChunkSentences(fileLineList):
    # 将文本按照句子进行分割
    sentenceList = []
    splitCharList = ['。', '！', '？', '...', '……']
    for fileLine in fileLineList:
        lineSentenceList = SplitStrToList(fileLine, splitCharList)
        sentenceList = sentenceList + lineSentenceList
    return sentenceList

def GetSplitPartStr(oriStr, delim, partIndex):
    if delim in oriStr:
        result = oriStr.split(delim)[partIndex]
    else:
        result = oriStr
    return result

if __name__ == "__main__":
    print(GetDayOffsetDaySecondNum(0))
    print(GetDayOffsetDaySecondNum(-7))
    print(GetDayOffsetDaySecondNum(7))

