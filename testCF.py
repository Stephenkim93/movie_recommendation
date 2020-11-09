from datetime import datetime
import csv
import numpy as np
import math

commonCntLimit = 10

def loadUsersAndItems(dataFileName):
    userIDSet = set()
    itemIDSet = set()

    f = open(dataFileName, "r")
    f.readline()
    f_csvReader = csv.reader(f)

    readCnt = 0
    for line in f_csvReader:
        userIDSet.add(line[0])
        itemIDSet.add(line[1])

        readCnt = readCnt + 1

    f.close()
    print("[" + str(datetime.now()) + "] Total {0} lises are processed (users & items) from file '{1}'..".format(readCnt, dataFileName))

    userIDList = list(userIDSet)
    userIDList.sort()

    itemIDList = list(itemIDSet)
    itemIDList.sort()

    return np.array(userIDList), np.array(itemIDList)

def loadUtilityMatrix(userIDList, itemIDList, dataFileName):
    utilityMatrix = np.full((len(userIDList), len(itemIDList)), 0, dtype=np.float)

    f = open(dataFileName, "r")
    f.readline()
    f_csvReader = csv.reader(f)

    readCnt = 0
    for line in f_csvReader:
        userIndex = np.where(userIDList == line[0])[0][0];
        itemIndex = np.where(itemIDList == line[1])[0][0];
        utilityMatrix[userIndex][itemIndex] = float(line[2])

        readCnt = readCnt + 1
    f.close()
    print("[" + str(datetime.now()) + "] Total {0} lines are processed (utilty matrix) from file '{1}'..".format(readCnt, dataFileName))

    return utilityMatrix


def getPCC(list_x, list_y):
    commonCnt = 0
    xSum, ySum = 0, 0
    xxSum, yySum = 0, 0
    xySum = 0

    for i in range(0, len(list_x)):
        x = list_x[i]
        y = list_y[i]
        if x > 0 and y > 0:
            commonCnt = commonCnt + 1
            xSum = xSum + x
            ySum = ySum + y
            xxSum = xxSum + (x * x)
            yySum = yySum + (y * y)
            xySum = xySum + (x * y)

    pccVal = 0
    if commonCnt >= commonCntLimit:
        numerator = (commonCnt * xySum) - (xSum * ySum)
        d_x = (commonCnt * xxSum) - (xSum * xSum)
        d_y = (commonCnt * yySum) - (ySum * ySum)
        denominator = math.sqrt(d_x) * math.sqrt(d_y)

        if numerator != 0 and denominator != 0:
            pccVal = numerator / denominator

    return pccVal

def calulateUserPCC(userIDList, utilityMatrix):
    userPCC = []

    userLen = len(userIDList)

    calcCnt = 0
    for i in range(userLen-1):
        for j in range(i+1, userLen):
            pccVal = getPCC(utilityMatrix[i], utilityMatrix[j])

            if pccVal > 0:
                userPCC.append((userIDList[i], userIDList[j], pccVal))

            calcCnt = calcCnt + 1
            if (calcCnt%10000) == 0:
                print("[" + str(datetime.now()) + "] pcc calurated - {0}({1}%)".format(calcCnt, calcCnt/360000*200))

    return np.array(userPCC)

def savePCC(pccList, saveFileName):
    f = open(saveFileName, "w")

    for pcc in pccList:
        f.write("{0},{1},{2}\n".format(pcc[0], pcc[1], pcc[2]))

    f.close()


def loadPCCFromFile(fileNmae):
    pccList = []

    f = open(fileNmae, "r")
    f_csvReader = csv.reader(f)

    readCnt = 0
    for line in f_csvReader:
        pccList.append((line[0], line[1], line[2]))
        readCnt = readCnt + 1
    f.close()

    print("[" + str(datetime.now()) + "] Total {0} lines are loaded (pcc info) from file '{1}'..".format(readCnt, fileNmae))

    return np.array(pccList)


def getSimilarUserList(targetUserID, userPCCList, threshold):
    similarUserList = []

    for pcc in userPCCList:
        if pcc[0] == targetUserID and float(pcc[2]) > threshold:
            similarUserList.append(pcc[1])
            # similarUserList.append((pcc[1], pcc[2]))
        elif pcc[1] == targetUserID and float(pcc[2]) > threshold:
            similarUserList.append(pcc[0])
            # similarUserList.append((pcc[0], pcc[2]))

    return similarUserList


def getScoreListOfSimUser(similarUserList, userIDList, targetItemID, itemIDList, utilityMatrix):
    scoreListOfSimUser = []

    iIdx = np.where(itemIDList == targetItemID)[0][0];

    for uID in similarUserList:
        uIdx = np.where(userIDList == uID)[0][0];
        rating = utilityMatrix[uIdx][iIdx]

        if rating > 0:
            scoreListOfSimUser.append(rating)

    return scoreListOfSimUser


def execCF():
    # 1. 파일을 로드 - 다음과 같은 정보들을 생성
    #     - 사용자ID 목록, 아이템ID 목록
    userIDList, itemIDList = loadUsersAndItems("ratings_1m.csv")
    print("[" + str(datetime.now()) + "] List of Users({0}) & Items({1}) are loaded ..".format(len(userIDList), len(itemIDList)))

    # 2. 사용자ID 목록과, 아이템ID 목록을 참고, 원래 파일을 다시 읽어서, Utility Matrix
    utilityMatrix = loadUtilityMatrix(userIDList, itemIDList, "ratings_1m.csv")
    print("[" + str(datetime.now()) + "] Loaded Utility Matrix Size: {0} * {1} = {2}".format(len(userIDList),len(itemIDList),len(userIDList) * len(itemIDList)))

    # 3. 사용자들간의 유사도를 구함 (PCC)
    # userPCCList = calulateUserPCC(userIDList, utilityMatrix)
    # print("[" + str(datetime.now()) + "] Total {0} PCCs are calculated..".format(len(userPCCList)))
    # savePCC(userPCCList, "userPCC.csv")
    # print("[" + str(datetime.now()) + "] PCC file is generated..")
    userPCCList = loadPCCFromFile("userPCC.csv")

    ####################################################################################################################
    ####################################################################################################################
    # 477, 4643, 3.5
    # 313, 4180, 5.0
    # 191, 303, 1.0

    targetUserID = "477"
    targetItemID = "4643"

    print("[" + str(datetime.now()) + "] Recommendation Process >> Target User: {0}, Target Item: {1}".format(targetUserID, targetItemID))

    # 4. 타켓 사용자에 대한 유사 사용자를 찾는다. (유사도가 0.7이상인 사람들만)
    similarUserList = getSimilarUserList(targetUserID, userPCCList, 0.3)
    print("[" + str(datetime.now()) + "] Total {0} similar users are selected..".format(len(similarUserList)))
    # print(similarUserList)

    # 5. 유사 사용들이 타겟 아이템에 대해 평가한 점수를 골라낸다.
    scoreListOfSimUser = getScoreListOfSimUser(similarUserList, userIDList, targetItemID, itemIDList, utilityMatrix)
    print("[" + str(datetime.now()) + "] Total {0} scores of similar users are extracted..".format(len(scoreListOfSimUser)))
    print(scoreListOfSimUser)

    # 6. 평균 등을 써서 예측값을 계산해 낸다.








    ###########################################################
# main execution
if __name__ == "__main__":
    print("[" + str(datetime.now()) + "] testCF is started..")

    execCF()

    print("[" + str(datetime.now()) + "] testCF is finished..")