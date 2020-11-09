import csv
import math
import numpy as np
from datetime import datetime

###########################################################
def loadIDList_MovieLensData_csv(fileName):
    userSet = set()
    movieSet = set()

    f = open(fileName, "r")
    f.readline()
    f_reader = csv.reader(f)
    readCnt = 0
    for line in f_reader:
        readCnt = readCnt + 1
        uid, mid = line[0], line[1]

        userSet.add(uid)
        movieSet.add(mid)

    f.close()

    userList = np.array(list(userSet))
    userList.sort()

    movieList = np.array(list(movieSet))
    movieList.sort()

    return readCnt, userList, movieList

###########################################################
def loadRatings_MovieLensData_csv(fileName, totalLineCnt, userList, movieList):
    utilityMatrix = np.full((len(userList),len(movieList)), 0, dtype=np.int32)
    print("[" + str(datetime.now()) + "] Utility Matrix is initialized> {0} cells.".format(len(movieList)*len(userList)))

    f = open(fileName, "r")
    f.readline()
    f_reader = csv.reader(f)
    readCnt = 0
    for line in f_reader:
        readCnt = readCnt + 1

        u_idx = np.where(userList == line[0])[0][0]
        m_idx = np.where(movieList == line[1])[0][0]

        utilityMatrix[u_idx][m_idx] = float(line[2])

        if (readCnt % 100000) == 0:
            print("[" + str(datetime.now()) + "] {0}/{1}({2:.2f}%) lines are processed".format(readCnt, totalLineCnt, readCnt/totalLineCnt*100))

    f.close()

    return readCnt, utilityMatrix

###########################################################
def cal_totalPCC(userList, utilityMatrix, commonCntLimit):
    pccIDList = []
    pccValueList = []

    totalCnt = ((len(userList) * len(userList)) - len(userList))/2
    calcCnt = 0
    for i in range(0, len(userList)-1):
        list_x = utilityMatrix[i]
        for j in range(i+1, len(userList)):
            list_y = utilityMatrix[j]
            pccVal = cal_pcc(list_x, list_y, commonCntLimit)

            if pccVal > 0:
                pccIDList.append((userList[i], userList[j]))
                pccValueList.append(pccVal)

            calcCnt = calcCnt + 1
            if (calcCnt % 1000) == 0:
                print("[" + str(datetime.now()) + "] {0}/{1}({2:.2f}%) pairs are calculated..".format(calcCnt, totalCnt, calcCnt/totalCnt*100))


    return np.array(pccIDList), np.array(pccValueList)

###########################################################
def cal_pcc(list_x, list_y, commonCntLimit):
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

###########################################################
def writePCC(fileName, pccIDList, pccValueList):
    f = open(fileName, "w")
    for i in range(0, len(pccIDList)):
        f.write("{0},{1},{2}\n".format(pccIDList[i][0],pccIDList[i][1],pccValueList[i]))
    f.close()

###########################################################
def load_totalPCC(fileName):
    pccIDList = []
    pccValueList = []

    readCnt = 0
    pccCnt = 0

    f = open(fileName, "r")
    f.readline()
    f_reader = csv.reader(f)
    for line in f_reader:
        pccVal = float(line[2])
        if abs(pccVal) > 0:
            pccIDList.append((line[0], line[1]))
            pccValueList.append(pccVal)
            pccCnt = pccCnt + 1
        readCnt = readCnt + 1
    f.close()

    print("{0} lines read and {1} meaningful pcc loaded..".format(readCnt, pccCnt))

    return np.array(pccIDList), np.array(pccValueList)

###########################################################
def getTopSimilarUserList(userID, topCnt, isNegativePccContain, similarityThreshold, pccIDList, pccValueList):
    simUserAndValList = []
    for i in range(0, len(pccIDList)):
        if pccIDList[i][0] == userID:
            simUserAndValList.append(list([pccIDList[i][1], pccValueList[i]]))
        elif pccIDList[i][1] == userID:
            simUserAndValList.append(list([pccIDList[i][0], pccValueList[i]]))

    if isNegativePccContain:
        simUserAndValList.sort(key=lambda x: abs(x[1])*(-1))
    else:
        simUserAndValList.sort(key=lambda x: x[1]*(-1))

    filteredList = []
    for simUserVal in simUserAndValList[0:topCnt]:
        if simUserVal[1] >= similarityThreshold:
            filteredList.append(simUserVal)


    return filteredList

###########################################################
def getRecomResult_Avg(testMovieID, similarUserAndValueList, utilityMatrix):
    cnt, sum = 0, 0
    for simUserVal in similarUserAndValueList:
        val = utilityMatrix[np.where(userList == simUserVal[0])[0][0]][np.where(movieList == testMovieID)[0][0]]
        if val > 0:
            sum = sum + val
            cnt = cnt + 1

    recomVal = 0

    if cnt == 0:
        recomVal = None
    else:
        recomVal = sum / cnt

    return recomVal



###########################################################
# main execution
if __name__ == "__main__":
    print("[" + str(datetime.now()) + "] pcc is started")
    #fileName = "ratings_full.csv"
    fileName = "ratings_1m.csv"
    totalLineCnt, userList, movieList = loadIDList_MovieLensData_csv(fileName)
    print("[" + str(datetime.now()) + "] {0} Users & {1} Movies List are loaded ({2} lines)..".format(len(userList), len(movieList), totalLineCnt))

    # totalRatingCnt, utilityMatrix = loadRatings_MovieLensData_csv(fileName, totalLineCnt, userList, movieList)
    # print("[" + str(datetime.now()) + "] {0} Ratings are loaded..".format(totalRatingCnt))

    #pccIDList, pccValueList = cal_totalPCC(userList, utilityMatrix, 10)
    #writePCC(fileName + ".pcc.csv", pccIDList, pccValueList)
    #print("[" + str(datetime.now()) + "] {0} PCC results for {1} Users are calculated..".format(len(pccValueList), len(userList)))
    #print("[" + str(datetime.now()) + "] {0} - file is written..".format(fileName + ".pcc.csv"))

    # pccIDList, pccValueList = load_totalPCC(fileName + ".pcc.csv")
    # print("[" + str(datetime.now()) + "] {0} PCC results for {1} Users are loaded..".format(len(pccValueList), len(userList)))



    # test section
    # testUserID = "5"
    # testMovieID = "588"
    # topCnt = 50
    # isNegativePccContain = False
    # similarityThreshold = 0.6
    #
    # similarUserAndValueList = getTopSimilarUserList(testUserID, topCnt, isNegativePccContain, similarityThreshold, pccIDList, pccValueList)
    #print(np.array(similarUserAndValueList))

    # recomVal = getRecomResult_Avg(testMovieID, similarUserAndValueList, utilityMatrix)




    # print("\tUser=\'{0}\', Movie=\'{1}\'".format(testUserID, testMovieID))
    # print("\tOrginal Rating={0}".format(utilityMatrix[np.where(userList == testUserID)[0][0]][np.where(movieList == testMovieID)[0][0]]))
    # print("\tPredicted Rating={0}".format(recomVal))
    # print("[" + str(datetime.now()) + "] pcc is finished")

