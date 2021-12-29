from PyQt5.QtWidgets import (QWidget, QComboBox, QVBoxLayout, QLabel, QSpinBox, QPushButton, QMessageBox, QDoubleSpinBox, QHBoxLayout, QMessageBox,QPushButton, QApplication, QSizePolicy, QLineEdit)
import sys
import simpy
import random
import numpy as np
from queue import Queue
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter
import math as m
import gurobipy as gp
from gurobipy import *
import ast
import json

csv_columns = ["run","voter ID", "arrival time", "check-in time", "leave check-in", "enter voting", "leave voting", "enter scanner", "departure time","check-in counter" ,"voting machine number", "scanner number", "balking","no of working machines", "people checking in", "people voting", "Check-in queue", "Voting queue", "Scanner queue", "total people in queue", "cycle time","waiting time","check-in cycle","voting cycle","scanner cycle"]
data_dict = []
runsMachineFailureRecord = []

class pollLocation(object):
    def __init__(self, env, num_checkIn, num_machines, num_scanners, cleaningBool, checkInMean, checkInSD, votingMean, percentExtraAssistance):
        self.env = env
        self.checkIn = simpy.Resource(env, num_checkIn)
        self.votingMachine = simpy.PriorityResource(env, num_machines)
        self.scanner = simpy.Resource(env, num_scanners)
        self.cleaningBool = cleaningBool
        self.machinesInUse = num_machines
        self.checkInCounterQ = Queue()
        self.votingMachineQ = Queue()
        self.scannerQ = Queue()
        self.failTime = 0
        self.Q = [0, 0, 0]
        self.machineFailureRecord = {}
        self.checkInMean = checkInMean
        self.checkInSD = checkInSD
        self.votingMean = votingMean
        self.percentExtraAssistance = percentExtraAssistance
        for i in range(num_checkIn):
            self.checkInCounterQ.put(i)
        for i in range(num_machines):
            self.votingMachineQ.put(i)
        for i in range(num_scanners):
            self.scannerQ.put(i)

    def getQ(self):
        return self.Q

    def setQ(self, index, count):
        self.Q[index] += count

    def get_failTime(self):
        return self.failTime

    def set_failTime(self, now):
        self.failTime = np.random.exponential(16 * 60) + now

    def removeCheckIn(self):
        return self.checkInCounterQ.get()

    def putCheckIn(self, num):
        self.checkInCounterQ.put(num)

    def removeVotingMachine(self):
        return self.votingMachineQ.get()

    def putVotingMachine(self, num):
        self.votingMachineQ.put(num)

    def removeScanner(self):
        return self.scannerQ.get()

    def putScanner(self, num):
        self.scannerQ.put(num)

    def checkIn_voter(self, voter):
        voterType = random.random()
        if (voterType < 1-self.percentExtraAssistance):
            x = np.random.normal(self.checkInMean, self.checkInSD)
        else:
            x = 2 * np.random.normal(self.checkInMean, self.checkInSD)
        while x <= 0.33:
            x = np.random.normal(self.checkInMean, self.checkInSD)
            if x > 0.33:
                break
        if (self.cleaningBool == "True"):
            yield self.env.timeout(1.2 * x)
        else:
            yield self.env.timeout(x)

    def voting(self, voter):
        x = np.random.normal(self.votingMean, 0.25 * self.votingMean)
        while x <= 1:
            x = np.random.normal(self.votingMean, 0.25 * self.votingMean)
            if x > 1:
                break
        if (self.cleaningBool == "True"):
            yield self.env.timeout(1.2 * x)
        else:
            yield self.env.timeout(x)

    def scan_ballot(self, voter):
        yield self.env.timeout(random.randrange(30, 60) / 60)

    def break_machine(self):
        with self.votingMachine.request(priority=0) as request:
            yield request
            self.machinesInUse -= 1
            machine = self.removeVotingMachine()
            time = random.uniform(15, 45)
            try:
                self.machineFailureRecord[machine] += time
            except:
                self.machineFailureRecord[machine] = time
            yield self.env.timeout(time)
        self.putVotingMachine(machine)
        self.machinesInUse += 1

    def get_machinesInUse(self):
        return self.machinesInUse

    def get_machineFailureRecord(self):
        return self.machineFailureRecord

def checkFailure(env, pollLocation, failureBool):
    if (env.now >= pollLocation.get_failTime()):
        if failureBool == "True":
            env.process(pollLocation.break_machine())
            pollLocation.set_failTime(env.now)

def go_to_vote(env, voter, pollLocation, balkingLimit, balkingProb, failureBool, runNo):
    # voter arrives to the polling location
    checkFailure(env, pollLocation, failureBool)
    voter_data = {}
    voter_data["run"] = runNo
    voter_data["voter ID"] = voter
    voter_data["arrival time"] = env.now
    stayProb = 1
    if (pollLocation.getQ()[0] + pollLocation.getQ()[1]) >= balkingLimit:
        stayProb = random.random()
    if stayProb < balkingProb:
        checkFailure(env, pollLocation, failureBool)
        # Voter leaves
        voter_data["departure time"] = env.now
        voter_data["balking"] = "True"
        voter_data["people checking in"] = pollLocation.checkIn.count
        voter_data["people voting"] = pollLocation.votingMachine.count
        voter_data["Check-in queue"] = pollLocation.getQ()[0]
        voter_data["Voting queue"] = pollLocation.getQ()[1]
        voter_data["Scanner queue"] = pollLocation.getQ()[2]
        voter_data["total people in queue"] = pollLocation.getQ()[0] + pollLocation.getQ()[1] + pollLocation.getQ()[2]
        data_dict.append(voter_data)
        failureUpTime = {}
        for key in pollLocation.get_machineFailureRecord().keys():
            failureUpTime[key] = pollLocation.get_machineFailureRecord()[key] / env.now
        try:
            runsMachineFailureRecord[runNo] = failureUpTime
        except:
            runsMachineFailureRecord.append(failureUpTime)
    else:
        pollLocation.setQ(0, 1)
        # voter checks in
        checkFailure(env, pollLocation, failureBool)
        with pollLocation.checkIn.request() as request:
            yield request
            pollLocation.setQ(0, -1)
            voter_data['check-in counter'] = pollLocation.removeCheckIn()
            voter_data["check-in time"] = env.now
            yield env.process(pollLocation.checkIn_voter(voter))
        pollLocation.putCheckIn(voter_data['check-in counter'])
        voter_data["leave check-in"] = env.now
        pollLocation.setQ(1, 1)
        checkFailure(env, pollLocation, failureBool)
        with pollLocation.votingMachine.request(priority=1) as request:
            yield request
            pollLocation.setQ(1, -1)
            voter_data["enter voting"] = env.now
            voter_data['voting machine number'] = pollLocation.removeVotingMachine()
            yield env.process(pollLocation.voting(voter))
        pollLocation.putVotingMachine(voter_data['voting machine number'])
        voter_data["leave voting"] = env.now
        pollLocation.setQ(2, 1)
        checkFailure(env, pollLocation, failureBool)
        with pollLocation.scanner.request() as request:
            yield request
            pollLocation.setQ(2, -1)
            voter_data["enter scanner"] = env.now
            voter_data["scanner number"] = pollLocation.removeScanner()
            yield env.process(pollLocation.scan_ballot(voter))
        pollLocation.putScanner(voter_data["scanner number"])
        checkFailure(env, pollLocation, failureBool)
        # Voter leaves
        voter_data["departure time"] = env.now
        voter_data["balking"] = "False"
        voter_data["no of working machines"] = pollLocation.get_machinesInUse()
        voter_data["people checking in"] = pollLocation.checkIn.count
        voter_data["people voting"] = pollLocation.votingMachine.count
        voter_data["Check-in queue"] = pollLocation.getQ()[0]
        voter_data["Voting queue"] = pollLocation.getQ()[1]
        voter_data["Scanner queue"] = pollLocation.getQ()[2]
        voter_data["total people in queue"] = pollLocation.getQ()[0] + pollLocation.getQ()[1] + pollLocation.getQ()[2]
        voter_data["cycle time"] = voter_data["departure time"] - voter_data["arrival time"]
        voter_data["waiting time"] = (voter_data["check-in time"] - voter_data["arrival time"]) + (
                    voter_data["enter voting"] - voter_data["leave check-in"]) + (
                                                 voter_data["enter scanner"] - voter_data["leave voting"])
        voter_data["check-in cycle"] = voter_data["leave check-in"] - voter_data["check-in time"]
        voter_data["voting cycle"] = voter_data["leave voting"] - voter_data["enter voting"]
        voter_data["scanner cycle"] = voter_data["departure time"] - voter_data["enter scanner"]
        data_dict.append(voter_data)
        failureUpTime = {}
        for key in pollLocation.get_machineFailureRecord().keys():
            failureUpTime[key] = pollLocation.get_machineFailureRecord()[key] / env.now
        try:
            runsMachineFailureRecord[runNo] = failureUpTime
        except:
            runsMachineFailureRecord.append(failureUpTime)

def generate_interarrivalTimes(locationExpArrivals):
    rateArray = []
    interarrival_time = []
    for i in range(len(locationExpArrivals)):
        if (locationExpArrivals[i] != 0):
            rateArray.append(60 / locationExpArrivals[i])
    while sum(interarrival_time) < (60 * len(rateArray)):
        if sum(interarrival_time) < 60:
            time = np.random.exponential(rateArray[0])
        elif sum(interarrival_time) < 120:
            time = np.random.exponential(rateArray[1])
        elif sum(interarrival_time) < 180:
            time = np.random.exponential(rateArray[2])
        elif sum(interarrival_time) < 240:
            time = np.random.exponential(rateArray[3])
        elif sum(interarrival_time) < 300:
            time = np.random.exponential(rateArray[4])
        elif sum(interarrival_time) < 360:
            time = np.random.exponential(rateArray[5])
        elif sum(interarrival_time) < 420:
            time = np.random.exponential(rateArray[6])
        elif sum(interarrival_time) < 480:
            time = np.random.exponential(rateArray[7])
        elif sum(interarrival_time) < 540:
            time = np.random.exponential(rateArray[8])
        elif sum(interarrival_time) < 600:
            time = np.random.exponential(rateArray[9])
        elif sum(interarrival_time) < 660:
            time = np.random.exponential(rateArray[10])
        else:
            if (len(rateArray) == 12):
                time = np.random.exponential(rateArray[11])
            else:
                if sum(interarrival_time) < 720:
                    time = np.random.exponential(rateArray[11])
                elif sum(interarrival_time) < 780:
                    time = np.random.exponential(rateArray[12])
                else:
                    time = np.random.exponential(rateArray[13])

        if (sum(interarrival_time) + time) > (60 * len(rateArray)):
            break
        else:
            interarrival_time.append(time)
    return interarrival_time

def mainSimRun(params):
    # Setup
    random.seed(42)
    data_dict.clear()
    runsMachineFailureRecord.clear()
    for i in range(30):
        # print(i)
        env = simpy.Environment()
        env.process(run_pollLocation(env, params[0], params[1], params[2],0,0,False, False, i, params[3], params[4], params[5], params[6], params[7]))
        env.run()

def run_pollLocation(env, num_checkIn, num_machines, num_scanners, balkingLimit, balkingProb, cleaningBool, failureBool,
                     runNo, locationExpArrivals, checkInMean, checkInSD, votingMean, percentExtraAssistance):
    PollLocation = pollLocation(env,num_checkIn, num_machines, num_scanners, cleaningBool, checkInMean, checkInSD, votingMean, percentExtraAssistance)
    # print(PollLocation)
    # print(locationExpArrivals)
    voterNo = 0
    PollLocation.set_failTime(0)
    interarrival_times = generate_interarrivalTimes(locationExpArrivals)
    # print(interarrival_times[0])
    for i in range(len(interarrival_times)):
        yield env.timeout(interarrival_times[i])
        voterNo += 1
        env.process(go_to_vote(env, voterNo, PollLocation, balkingLimit, balkingProb, failureBool, runNo))

def calculateUtil(df, numCheckIn, numVoting, num_scanners):
    runsCompleteData = []
    for i in range(30):
        runData = {}
        runData["close time"] = (df[df["run"] == i]["departure time"].max())
        for j in range(num_scanners):
            runData[f"scanner_{j}"] = (df[(df["run"] == i) & (df["scanner number"] == j)]["scanner cycle"]).sum()
        for j in range(numCheckIn):
            runData[f"checkIn_{j}"] = (df[(df["run"] == i) & (df["check-in counter"] == j)]["check-in cycle"]).sum()
        for j in range(numVoting):
            runData[f"voting_{j}"] = (df[(df["run"] == i) & (df["voting machine number"] == j)]["voting cycle"]).sum()
        runsCompleteData.append(runData)

    utilCheckin = {"check-in": []}
    utilBMDs = {"BMDs": []}
    utilScanner = {"scanner": []}
    for i in range(30):
        runUtil = {}
        for j in range(num_scanners):
            utilScanner["scanner"].append(runsCompleteData[i][f"scanner_{j}"] / runsCompleteData[i]["close time"])
        for j in range(numCheckIn):
            utilCheckin["check-in"].append(runsCompleteData[i][f"checkIn_{j}"] / runsCompleteData[i]["close time"])
        for j in range(numVoting):
            utilBMDs["BMDs"].append(runsCompleteData[i][f"voting_{j}"] / runsCompleteData[i]["close time"])
    return [pd.DataFrame(utilCheckin)['check-in'].mean(), pd.DataFrame(utilBMDs)['BMDs'].mean(),
            pd.DataFrame(utilScanner)['scanner'].mean()]

def Performance_function(wait,eps):
    eps = eps/100
    if (wait<(30.0*(1+eps))):
        return 1
    elif (wait<(45.0*(1+eps))):
        return 0.9
    elif (wait < (60.0*(1+eps))):
        return 0.75
    elif (wait < (90.0*(1+eps))):
        return 0.5
    elif (wait < (150.0*(1+eps))):
        return 0.2
    else:
        return 0

def configure_data_files():
    registered_all = pd.read_excel("Registered_Voters.xlsx",sheet_name="Registered_Voters")
    registered_all = registered_all.filter(["Location", "Registered Voters"], axis=1)
    print(registered_all)
    registered_all = registered_all.sort_values(by=["Location"], axis=0)

    turnout_df = pd.read_csv("Nov3TurnoutByLocation.csv")
    print(turnout_df.columns)

    off_alloc = pd.read_excel("Official Allocation.xlsx")
    print(off_alloc.columns)

    tot = 0
    count = 0
    for i in range(len(registered_all)):
        if (turnout_df["Voting Location"][i] == registered_all["Location"][i]):
            count = count + 1
        if (off_alloc["Location"][i] == registered_all["Location"][i]):
            tot = tot + 1
    print(tot, count)

    registered_all["Turnout"] = turnout_df['3/11/2020']
    registered_all["Poll Pads"] = off_alloc["Poll Pads"]
    registered_all["BMDs"] = off_alloc["BMDs"]
    registered_all["Scanners"] = off_alloc["Scanners"]
    print(registered_all.columns)
    print(registered_all["Turnout"])

    election_hourly = pd.read_excel("Registered_Voters.xlsx", sheet_name="Hourly Distribution")
    print(election_hourly.columns)
    # election_hourly = election_hourly.sort_values(by = ["Location"], axis = 0).reset_index(drop = True)
    # print(election_hourly)
    election_hourly = election_hourly.loc[election_hourly["Location"].isin(list(registered_all["Location"]))].reset_index(drop=True)
    print(election_hourly)

    tot = 0
    for i in range(len(election_hourly)):
        if election_hourly["Location"][i] != registered_all["Location"][i]:
            tot = tot + 1
    print(tot)
    election_hourly["Total"] = registered_all["Turnout"]
    election_hourly["Check-in Mean"] = [3]*len(election_hourly)
    election_hourly["Check-in SD"] = [0.75]*len(election_hourly)
    election_hourly["Voting Mean"] = [8]*len(election_hourly)
    election_hourly["Voting SD"] = [2]*len(election_hourly)

    hourly_numbers = pd.DataFrame(columns=list(election_hourly.columns))
    print(hourly_numbers)

    list_of_columns = list(hourly_numbers.columns)

    for i in range(len(list_of_columns)):
        if (i == 0) or (i == len(list_of_columns) - 1):
            # if (i==0):
            hourly_numbers[list_of_columns[i]] = election_hourly[list_of_columns[i]]
        else:
            for j in range(len(election_hourly)):
                # hourly_numbers[list_of_columns[i]] =  hourly_numbers[list_of_columns[i]].append(election_hourly[list_of_columns[i]][j]*registered_all["Turnout"][j])
                # hourly_numbers[list_of_columns[i]].append(election_hourly[list_of_columns[i]][j]*registered_all["Turnout"][j])
                hourly_numbers.loc[j, list_of_columns[i]] = int(election_hourly[list_of_columns[i]][j] * registered_all["Turnout"][j])
    print(hourly_numbers)
    hourly_numbers["Check-in Mean"] = [3]*len(hourly_numbers)
    hourly_numbers["Check-in SD"] = [0.75]*len(hourly_numbers)
    hourly_numbers["Voting Mean"] = [8]*len(hourly_numbers)
    hourly_numbers["Voting SD"] = [2]*len(hourly_numbers)

    with pd.ExcelWriter("Voters_Info.xlsx") as writer:
        registered_all.to_excel(writer, sheet_name="Registered_Voters", index=False)
        election_hourly.to_excel(writer, sheet_name="Hourly_%", index=False)
        hourly_numbers.to_excel(writer, sheet_name="Hourly_numbers", index=False)


def index_columns(alist):
    a = 0
    a_dict = {}
    for i in alist:
        a_dict.setdefault(a,i)
        a = a + 1
    return a_dict

def updateAggregateInfo(Polls,aggregateInfo, util, df, machineCounts, queueLimit):
    wait = df['waiting time'].mean()+(3 * df['waiting time'].std())
    # print(wait)
    Perf = Performance_function(wait,0)
    # print(Perf)
    aggregateInfo = aggregateInfo.append({"Polling Location": Polls, "check-in": machineCounts["poll pads"], "BMDs": machineCounts["BMDs"], "Scanners": machineCounts["Scanners"],
         "check-in util": util[0],
         "BMD util": util[1],
         "scanner util": util[2],
         "avg CT": df['cycle time'].mean(), "avg WT": df['waiting time'].mean(),
         "Performance":Perf,
         "99.7% WT": df['waiting time'].mean() + (3 * df['waiting time'].std()),
         "Avg Check-in QL": df["Check-in queue"].mean(),
         "Avg Voting QL" : df["Voting queue"].mean(),
         "Avg Scanner QL": df["Scanner queue"].mean(),
         "avg QL": df['total people in queue'].mean(),
         "99.7% QL": df['total people in queue'].mean() + (3 * df['total people in queue'].std()),
        "Available Queuing Space (people)":queueLimit}, ignore_index=True)
    # aggregateInfo = aggregateInfo.append(
    #     {"Polling Location": Polls, "check-in": machineCounts["poll pads"], "BMDs": machineCounts["BMDs"], "Scanners": machineCounts["Scanners"],
    #      "Performance":Perf}, ignore_index=True)
    # print(type(df['waiting time'].mean()))
    return aggregateInfo,Perf,wait

def updatechart(Polls,aggregateInfo, util, df, machineCounts, queueLimit,t):
    wait = df['waiting time'].mean()+(3 * df['waiting time'].std())
    Perf = Performance_function(wait,0)
    aggregateInfo = aggregateInfo.append({"Polling Location": Polls,"Performance"+str(t):Perf}, ignore_index=False)
    # print(type(df['waiting time'].mean()))
    return aggregateInfo,Perf,wait

def Allocation_chart_combined(samp_dict,aggregate_df,registered,hourly_percent_dict,lb,ub):
    turnper = [10 * i for i in range(1, 11, 1)]
    print(turnper)
    check_in_limits = {}
    Polls_dict = {}
    Perf_per_dict = {}
    for i in range(lb,ub,1):
    # for i in list(samp_dict.keys):
        # if (i!= 82):
            for k in turnper:
                print(i,k)
                turnout = registered[i] * 0.01 * (k / 100)
                # print(turnout)
                # print(arrivals)
                mach = list(ast.literal_eval(samp_dict[i]))
                arrivals = [int(turnout * hourly_percent_dict[i][1]), int(turnout * hourly_percent_dict[i][2]),
                            int(turnout * hourly_percent_dict[i][3]), int(turnout * hourly_percent_dict[i][4]),
                            int(turnout * hourly_percent_dict[i][5]), int(turnout * hourly_percent_dict[i][6]),
                            int(turnout * hourly_percent_dict[i][7]), int(turnout * hourly_percent_dict[i][8]),
                            int(turnout * hourly_percent_dict[i][9]), int(turnout * hourly_percent_dict[i][10]),
                            int(turnout * hourly_percent_dict[i][11]), int(turnout * hourly_percent_dict[i][12]),
                            int(turnout * hourly_percent_dict[i][13]), int(turnout * hourly_percent_dict[i][14]),
                            int(turnout * hourly_percent_dict[i][15]), int(turnout * hourly_percent_dict[i][16]),
                            int(turnout * hourly_percent_dict[i][17]), int(turnout * hourly_percent_dict[i][18])]
                # print(arrivals)
                if(np.sum(arrivals)!= 0):
                    params = [mach[0], mach[1], mach[2], arrivals, 3.0, 0.75, 8.0, 0.0]
                    mainSimRun(params)
                    df = pd.DataFrame(data_dict)
                    util = calculateUtil(df, mach[0], mach[1], mach[2])
                    machineCounts = {"poll pads": mach[0], "BMDs": mach[1], "Scanners": mach[2]}
                    wait = df['waiting time'].mean() + (3 * df['waiting time'].std())
                    Perf = Performance_function(wait, 0)
                    Perf_per_dict.setdefault(i,[]).append(Perf)
                else:
                    Perf_per_dict.setdefault(i,[]).append(1)
            aggregate_df = update_rows(reverse_poll_index[i],aggregate_df,Perf_per_dict[i])
    return aggregate_df

def plot_histogram(new_list, bin, y_l,text):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    your_bins = bin
    arr = plt.hist(new_list, bins=your_bins, histtype='stepfilled', color='orange', edgecolor='black')
    print(arr)
    plt.xlabel('Performance', fontsize=15)
    plt.ylabel('#voting locations', fontsize=15)
    plt.xticks([0,0.2,0.5,0.75,0.9,1],fontsize=15)
    plt.yticks(fontsize=15)
    if(np.sum(new_list)>= 128):
        plt.xlim([-0.1,1.1])
    plt.ylim([0,y_l])
    plt.title('Frequency distribution of performance across polling locations - '+ text, fontsize=15)
    for i in range(your_bins):
        if arr[0][i] > 0:
            plt.text(arr[1][i], arr[0][i]+0.05, str(arr[0][i]))
    plt.show()

def Multiple_Days_KPI(dailyData,aggregateInfo):
    hourly = pd.read_excel("hourly primary checkin data.xlsx")
    print(hourly)
    for i in range(0,2,1):
        t = np.random.dirichlet(np.ones(14), size=1)[0]
        for c in range(3,7,1):
            for j in range(15,60,1):
                for k in range(2,4,1):
                    print(i,c,j,k)
                    arrivals = [int(t[0]*dailyData["Registered"][i]*0.1), int(t[1]*dailyData["Registered"][i]*0.1),int(t[2]*dailyData["Registered"][i]*0.1),int(t[3]*dailyData["Registered"][i]*0.1),
                        int(t[4]*dailyData["Registered"][i]*0.1), int(t[5]*dailyData["Registered"][i]*0.1), int(t[6]*dailyData["Registered"][i]*0.1),int(t[7]*dailyData["Registered"][i]*0.1),
                        int(t[8]*dailyData["Registered"][i]*0.1), int(t[9]*dailyData["Registered"][i]*0.1),int(t[10] * dailyData["Registered"][i]*0.1), int(t[11]*dailyData["Registered"][i]*0.1),
                        int(t[12]*dailyData["Registered"][i]*0.1),int(t[13]*dailyData["Registered"][i]*0.1)]
                    params = [int(dailyData["Check-ins"][i]),j,int(dailyData["Scanner"][i]), arrivals,float(dailyData["Check-in Mean"][i]),float(dailyData["Check-in SD"][i]), float(dailyData["Voting Mean"][i]),0.0]
                    mainSimRun(params)
                    df = pd.DataFrame(data_dict)
            # print(df.columns)
            #         util = calculateUtil(df, int(dailyData["Check-ins"][i]),j,k)
                    util = calculateUtil(df,c,j,k)
            # print(util)
                    machineCounts = {"poll pads": c, "BMDs":j, "Scanners":k}
            # print(machineCounts)
                    aggregateInfo = updateAggregateInfo(dailyData["Polling Location"][i],aggregateInfo, util, df, machineCounts,0)
    print(aggregateInfo["avg WT"].tolist()[0])
    aggregateInfo.to_csv("New_sim_results.csv")
    print(aggregateInfo)

def Simulation_runs(registered_all,hourly_numbers,aggregateInfo_Baseline):

    # base_registered = pd.read_excel("Voters_Info.xlsx", sheet_name="Registered_Voters",index_col= False)
    for i in range(len(registered_all)):
    # for i in range(5):
        print(i)
        arrivals = [int(hourly_numbers["Col1"][i]), int(hourly_numbers["Col2"][i]), int(hourly_numbers["Col3"][i]),
                    int(hourly_numbers["Col4"][i]), int(hourly_numbers["Col5"][i]), int(hourly_numbers["Col6"][i]), int(hourly_numbers["Col7"][i]),
                    int(hourly_numbers["Col8"][i]), int(hourly_numbers["Col9"][i]), int(hourly_numbers["Col10"][i]), int(hourly_numbers["Col11"][i]),
                    int(hourly_numbers["Col12"][i]),int(hourly_numbers["Col13"][i]), int(hourly_numbers["Col14"][i]), int(hourly_numbers["Col15"][i]),
                    int(hourly_numbers["Col16"][i]),int(hourly_numbers["Col17"][i]), int(hourly_numbers["Col18"][i])]
        params = [registered_all["Poll Pads"][i],registered_all["BMDs"][i],registered_all["Scanners"][i], arrivals, float(hourly_numbers["Check-in Mean"][i]),
                  float(hourly_numbers["Check-in SD"][i]), float(hourly_numbers["Voting Mean"][i]), 0.0]
        # print(arrivals, params)
        mainSimRun(params)
        df = pd.DataFrame(data_dict)
        util = calculateUtil(df, registered_all["Poll Pads"][i], registered_all["BMDs"][i], registered_all["Scanners"][i])
        machineCounts = {"poll pads": registered_all["Poll Pads"][i], "BMDs": registered_all["BMDs"][i], "Scanners": registered_all["Scanners"][i]}
        aggregateInfo_Baseline, Perfor, wait_sig = updateAggregateInfo(hourly_numbers["Location"][i], aggregateInfo_Baseline,util, df, machineCounts, 0)
    aggregateInfo_Baseline.to_excel("Baseline Performance Chart.xlsx", index = False)

def update_rows(ploc,aggregateInfo_Perf,Perf_per_dict):
    aggregateInfo_Perf = aggregateInfo_Perf.append({"Location":ploc,"Perf10":Perf_per_dict[0],"Perf20":Perf_per_dict[1],"Perf30":Perf_per_dict[2],"Perf40": Perf_per_dict[3],
                                                    "Perf50": Perf_per_dict[4],"Perf60": Perf_per_dict[5],"Perf70": Perf_per_dict[6],"Perf80": Perf_per_dict[7],"Perf90": Perf_per_dict[8],"Perf100": Perf_per_dict[9]},ignore_index = True)

    return aggregateInfo_Perf

def Allocation_IP(perf_dict,typ,lims,gap,perf_lb):
    "Polling locations considered"
    lcns = np.unique(list(perf_dict.keys()))
    # print(lcns)
    a = 0
    index_locs = {}
    for i in lcns:
        index_locs.setdefault(i, a)
        a = a + 1
    print(index_locs)
    locs_comb = []
    for i in range(len(index_locs)):
        for j in range(i + 1, len(index_locs)):
            locs_comb.append([i, j])
    print(locs_comb)

    cand_mod = gp.Model("Optimal Resource Allocation")

    "variable creation"
    var = {index_locs[loc]: {j: cand_mod.addVar(vtype=GRB.BINARY) for j in perf_dict[loc]} for loc in lcns}
    u = {str(i):cand_mod.addVar(vtype = GRB.CONTINUOUS,lb = 0.0) for i in locs_comb}
    v = {str(i):cand_mod.addVar(vtype = GRB.CONTINUOUS,lb = 0.0) for i in locs_comb}
    maxi = cand_mod.addVar(vtype=GRB.CONTINUOUS,lb=0.0)
    mini = cand_mod.addVar(vtype = GRB.CONTINUOUS,lb=0.0)
    # print(var)
    tot = 0
    for loc in lcns:
        tot = tot + 1
        print(tot)
        cand_mod.addConstr(gp.quicksum(var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.EQUAL, 1)
        # cand_mod.addConstr(gp.quicksum(perf_dict[loc][j]*var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.LESS_EQUAL,y)
        # cand_mod.addConstr(gp.quicksum(perf_dict[loc][j] * var[index_locs[loc]][j] for j in perf_dict[loc]),
        #                    GRB.GREATER_EQUAL, perf_lb)

    if(typ == "Range"):
        total_checks = LinExpr()
        total_bmds = LinExpr()
        total_scans = LinExpr()
        for loc in lcns:
            # cand_mod.addConstr(gp.quicksum(perf_dict[loc][j]*var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.GREATER_EQUAL,perf_lb)
            for j in perf_dict[loc]:
                # print(var[index_locs[loc]][j])
                total_checks = total_checks + check_c[loc][j] * var[index_locs[loc]][j]
                total_bmds = total_bmds + bmds_c[loc][j] * var[index_locs[loc]][j]
                total_scans = total_scans + scns_c[loc][j] * var[index_locs[loc]][j]
        # cand_mod.addConstr(total_checks, GRB.EQUAL, lims[0])
        # cand_mod.addConstr(total_bmds, GRB.EQUAL, lims[1])
        # cand_mod.addConstr(total_scans, GRB.EQUAL, lims[2])
        cand_mod.addConstr(total_checks, GRB.LESS_EQUAL, lims[0])
        cand_mod.addConstr(total_bmds, GRB.LESS_EQUAL, lims[1])
        cand_mod.addConstr(total_scans, GRB.LESS_EQUAL, lims[2])

        for loc in lcns:
            perf_l1 = gp.quicksum(perf_dict[loc][k] * var[index_locs[loc]][k] for k in perf_dict[loc])
            cand_mod.addConstr(perf_l1, GRB.LESS_EQUAL, maxi)
            cand_mod.addConstr(perf_l1, GRB.GREATER_EQUAL, mini)

        cand_mod.Params.MIPGap = gap
        cand_mod.setObjective(maxi - mini, GRB.MINIMIZE)
        cand_mod.optimize()
        opt_solution = {}
        for i in var:
            y = list(index_locs.keys())[i]
            for j in var[i]:
                if (var[i][j].x >= 1):
                    # opt_solution.setdefault(y, []).append(j)
                    opt_solution.setdefault(y, j)
        # print(opt_solution)
        # Optimization Model
        opt_dict = {}
        for i in opt_solution:
            # for j in opt_solution[i]:
                j = opt_solution[i]
                opt_dict.setdefault(i, perf_dict[i][j])

        return opt_solution, opt_dict

    if(typ == "Fair"):
        total_checks = LinExpr()
        total_bmds = LinExpr()
        total_scans = LinExpr()
        for loc in lcns:
            cand_mod.addConstr(gp.quicksum(perf_dict[loc][j]*var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.GREATER_EQUAL,perf_lb)
            for j in perf_dict[loc]:
                # print(var[index_locs[loc]][j])
                total_checks = total_checks + check_c[loc][j] * var[index_locs[loc]][j]
                total_bmds = total_bmds + bmds_c[loc][j] * var[index_locs[loc]][j]
                total_scans = total_scans + scns_c[loc][j] * var[index_locs[loc]][j]
        # cand_mod.addConstr(total_checks, GRB.EQUAL,lims[0])
        # cand_mod.addConstr(total_bmds, GRB.EQUAL,lims[1])
        # cand_mod.addConstr(total_scans, GRB.EQUAL,lims[2])
        cand_mod.addConstr(total_checks, GRB.LESS_EQUAL,lims[0])
        cand_mod.addConstr(total_bmds, GRB.LESS_EQUAL,lims[1])
        cand_mod.addConstr(total_scans, GRB.LESS_EQUAL,lims[2])
        obj = LinExpr()
        for pair in locs_comb:
            i,j = pair
            # print(pair)
            obj = obj + (u[str([i,j])] + v[str([i,j])])
            perf_l1 = gp.quicksum(perf_dict[lcns[i]][k] * var[i][k] for k in perf_dict[lcns[i]])
            perf_l2 = gp.quicksum(perf_dict[lcns[j]][l] * var[j][l] for l in perf_dict[lcns[j]])
            cand_mod.addConstr(perf_l2 - perf_l1, GRB.EQUAL,(u[str([i, j])] - v[str([i, j])]))

        cand_mod.Params.MIPGap = gap
        cand_mod.setObjective(obj, GRB.MINIMIZE)
        cand_mod.optimize()
        opt_solution = {}
        for i in var:
            y = list(index_locs.keys())[i]
            for j in var[i]:
                if (var[i][j].x >= 1):
                    # opt_solution.setdefault(y, []).append(j)
                    opt_solution.setdefault(y, j)
        # print(opt_solution)
        # Optimization Model
        opt_dict = {}
        for i in opt_solution:
            # for j in opt_solution[i]:
                j = opt_solution[i]
                opt_dict.setdefault(i, perf_dict[i][j])

        return opt_solution, opt_dict

    total_checks = LinExpr()
    total_bmds = LinExpr()
    total_scans = LinExpr()
    obj = LinExpr()
    for loc in lcns:
        for j in perf_dict[loc]:
               total_checks = total_checks+check_c[loc][j]*var[index_locs[loc]][j]
               total_bmds = total_bmds+bmds_c[loc][j]*var[index_locs[loc]][j]
               total_scans = total_scans+scns_c[loc][j]*var[index_locs[loc]][j]
               obj = obj + perf_dict[loc][j]*var[index_locs[loc]][j]
    cand_mod.addConstr(total_checks, GRB.LESS_EQUAL, lims[0])
    cand_mod.addConstr(total_bmds, GRB.LESS_EQUAL, lims[1])
    cand_mod.addConstr(total_scans, GRB.LESS_EQUAL, lims[2])
    # cand_mod.addConstr(total_checks, GRB.EQUAL, lims[0])
    # cand_mod.addConstr(total_bmds, GRB.EQUAL, lims[1])
    # cand_mod.addConstr(total_scans, GRB.EQUAL, lims[2])
    cand_mod.setObjective(obj,GRB.MAXIMIZE)
    # cand_mod.setObjective(gp.quicksum(perf_dict[loc][j] * var[index_locs[loc]][j] for j in perf_dict[loc] for loc in lcns), GRB.MAXIMIZE)
    cand_mod.optimize()
    cand_mod.write("Allocation.lp")


    opt_solution = {}
    for i in var:
        y = list(index_locs.keys())[i]
        for j in var[i]:
            if(var[i][j].x>=1):
                # opt_solution.setdefault(y,[]).append(j)
                opt_solution.setdefault(y, j)
    # print(opt_solution)
    #Optimization Model
    opt_dict ={}
    for i in opt_solution:
        # for j in opt_solution[i]:
            j = opt_solution[i]
            opt_dict.setdefault(i,perf_dict[i][j])

    return opt_solution,opt_dict

def box_plots_df(data_df,Ratio):
    new_data_df = pd.DataFrame(columns=["Polling Locations","Type",Ratio])
    col_bmd_reg = list(data_df.columns)
    print(col_bmd_reg)
    new_data_dict = {i:[] for i in new_data_df.columns}
    print(new_data_dict)
    for i in range(len(data_df)):
        for j in range(1,len(col_bmd_reg),1):
            new_data_dict["Polling Locations"].append(data_df["Polling Locations"][i])
            new_data_dict["Type"].append(col_bmd_reg[j])
            new_data_dict[Ratio].append(data_df[col_bmd_reg[j]][i])
    print(new_data_dict)
    for i in new_data_df.columns:
        new_data_df[i] = list(new_data_dict[i])
    print(new_data_df)
    new_data_df.to_excel("Election_Day_Box_"+Ratio+".xlsx",index=False)
    
def bi_objective(perf_dict,diff,lims,gap,perf_lb):
    "Polling locations considered"
    lcns = np.unique(list(perf_dict.keys()))
    # print(lcns)
    a = 0
    index_locs = {}
    for i in lcns:
        index_locs.setdefault(i, a)
        a = a + 1
    # print(index_locs)
    locs_comb = []
    for i in range(len(index_locs)):
        for j in range(i + 1, len(index_locs)):
            locs_comb.append([i, j])
    # print(locs_comb)

    cand_mod = gp.Model("Optimal Resource Allocation")

    "variable creation"
    var = {index_locs[loc]: {j: cand_mod.addVar(vtype=GRB.BINARY) for j in perf_dict[loc]} for loc in lcns}
    u = {str(i):cand_mod.addVar(vtype = GRB.CONTINUOUS,lb = 0.0) for i in locs_comb}
    v = {str(i):cand_mod.addVar(vtype = GRB.CONTINUOUS,lb = 0.0) for i in locs_comb}
    maxi = cand_mod.addVar(vtype=GRB.CONTINUOUS,lb=0.0)
    mini = cand_mod.addVar(vtype = GRB.CONTINUOUS,lb=0.0)
    # print(var)
    for loc in lcns:
        cand_mod.addConstr(gp.quicksum(var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.EQUAL, 1)
        # cand_mod.addConstr(gp.quicksum(perf_dict[loc][j]*var[index_locs[loc]][j] for j in perf_dict[loc]), GRB.LESS_EQUAL,y)

    total_checks = LinExpr()
    total_bmds = LinExpr()
    total_scans = LinExpr()
    obj = LinExpr()
    for loc in lcns:
        for j in perf_dict[loc]:
               total_checks = total_checks+check_c[loc][j]*var[index_locs[loc]][j]
               total_bmds = total_bmds+bmds_c[loc][j]*var[index_locs[loc]][j]
               total_scans = total_scans+scns_c[loc][j]*var[index_locs[loc]][j]
               obj = obj + perf_dict[loc][j]*var[index_locs[loc]][j]
    cand_mod.addConstr(total_checks, GRB.LESS_EQUAL, lims[0])
    cand_mod.addConstr(total_bmds, GRB.LESS_EQUAL, lims[1])
    cand_mod.addConstr(total_scans, GRB.LESS_EQUAL, lims[2])
    # cand_mod.addConstr(total_checks, GRB.EQUAL, lims[0])
    # cand_mod.addConstr(total_bmds, GRB.EQUAL, lims[1])
    # cand_mod.addConstr(total_scans, GRB.EQUAL, lims[2])

    obj_2 = LinExpr()

    for loc in lcns:
        perf_l1 = gp.quicksum(perf_dict[loc][k] * var[index_locs[loc]][k] for k in perf_dict[loc])
        cand_mod.addConstr(perf_l1,GRB.LESS_EQUAL,maxi)
        cand_mod.addConstr(perf_l1,GRB.GREATER_EQUAL,mini)
    cand_mod.addConstr(maxi - mini,GRB.LESS_EQUAL,diff)


    cand_mod.setObjective(obj, GRB.MAXIMIZE)
    cand_mod.optimize()
    cand_mod.write("Allocation.lp")
    opt_solution = {}
    for i in var:
        y = list(index_locs.keys())[i]
        for j in var[i]:
            if (var[i][j].x >= 1):
                # opt_solution.setdefault(y, []).append(j)
                opt_solution.setdefault(y, j)
    print(opt_solution)
    # Optimization Model
    opt_dict = {}
    for i in opt_solution:
        # for j in opt_solution[i]:
        j = opt_solution[i]
        opt_dict.setdefault(i, perf_dict[i][j])

    return opt_solution, opt_dict

def sensitivity_plot(new_df,k):
    plt.xlabel("ε", fontsize=15)
    plt.ylabel('Performance', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Performance vs '+"ε"+" for different polling locations", fontsize=15)
    for ind in range(0,10,1):
        x = []
        y = []
        for t in range(0,k,1):
            x.append(t)
            y.append(new_df["Performance"+str(t)][ind])
        plt.plot(x,y,label = new_df["Polling Location"][ind])
    plt.legend()

def finding_checkin_limits(aggregateInfo,lb,ub):
    d = [30]
    check_in_limits = {}
    for i in range(lb,ub,1):
        try:
            count = 0
            for ch in range(5,20,1):
                for bm in d:
                    # if(bm/ch<=4) and (bm/ch>=0.65):
                        arrivals = [int(hourly_numbers["Col1"][i]), int(hourly_numbers["Col2"][i]), int(hourly_numbers["Col3"][i]), int(hourly_numbers["Col4"][i]),
                                    int(hourly_numbers["Col5"][i]), int(hourly_numbers["Col6"][i]), int(hourly_numbers["Col7"][i]), int(hourly_numbers["Col8"][i]),
                                    int(hourly_numbers["Col9"][i]), int(hourly_numbers["Col10"][i]), int(hourly_numbers["Col11"][i]),int(hourly_numbers["Col12"][i]),
                                    int(hourly_numbers["Col13"][i]), int(hourly_numbers["Col14"][i]), int(hourly_numbers["Col15"][i]),
                                    int(hourly_numbers["Col16"][i]), int(hourly_numbers["Col17"][i]), int(hourly_numbers["Col18"][i])]
                        params = [ch, bm,3, arrivals, float(hourly_numbers["Check-in Mean"][i]), float(hourly_numbers["Check-in SD"][i]),
                                  float(hourly_numbers["Voting Mean"][i]), 0.0]
                        mainSimRun(params)
                        df = pd.DataFrame(data_dict)
                        util = calculateUtil(df, ch, bm, 3)
                        machineCounts = {"poll pads": ch, "BMDs": bm, "Scanners": 3}
                        aggregateInfo, Perfor, wait_sig= updateAggregateInfo(hourly_numbers["Location"][i], aggregateInfo, util, df,machineCounts, 0)
                        print(i,ch,bm,round(wait_sig,2))
                        if((wait_sig>=20) & (wait_sig<=120)):
                            check_in_limits.setdefault(i,[]).append(ch)
                        if(wait_sig<20):
                            count = count + 1
                            check_in_limits.setdefault(i, []).append(ch)
                            if(count == 2):
                                raise check_cot
        except Check_count:
            continue
    return aggregateInfo,check_in_limits

def finding_all_combs(aggregateInfo,lb,ub,limits_poll):
    count_check = 0
    print(limits_poll)
    for i in range(lb,ub,1):
        lim = limits_poll[hourly_numbers["Location"][i]]
        print(lim)
        arrivals = [int(hourly_numbers["Col1"][i]), int(hourly_numbers["Col2"][i]), int(hourly_numbers["Col3"][i]), int(hourly_numbers["Col4"][i]),
                    int(hourly_numbers["Col5"][i]), int(hourly_numbers["Col6"][i]), int(hourly_numbers["Col7"][i]), int(hourly_numbers["Col8"][i]),
                    int(hourly_numbers["Col9"][i]), int(hourly_numbers["Col10"][i]), int(hourly_numbers["Col11"][i]), int(hourly_numbers["Col12"][i]),
                    int(hourly_numbers["Col13"][i]), int(hourly_numbers["Col14"][i]), int(hourly_numbers["Col15"][i]), int(hourly_numbers["Col16"][i]),
                    int(hourly_numbers["Col17"][i]), int(hourly_numbers["Col18"][i])]
        for c in range(lim[0],lim[1]+1,1):
            for mc in range(2,40,1):
                    for k in range(1,4,1):
                        print(i, c, mc, k)
                        params = [c, mc, k, arrivals, float(hourly_numbers["Check-in Mean"][i]),
                                  float(hourly_numbers["Check-in SD"][i]), float(hourly_numbers["Voting Mean"][i]), 0.0]
                        print(arrivals,params)
                        mainSimRun(params)
                        df = pd.DataFrame(data_dict)
                        util = calculateUtil(df, c, mc, k)
                        machineCounts = {"poll pads": c, "BMDs": mc, "Scanners": k}
                        aggregateInfo,Perfor,wait_sig = updateAggregateInfo(hourly_numbers["Location"][i],aggregateInfo, util, df, machineCounts,0)
        # print(aggregateInfo["avg WT"].tolist()[0])
    aggregateInfo.to_excel("all_comb_"+str(ub)+".xlsx",index = False)
    return aggregate_Info

class ContinueI(Exception):
    pass

class zeroCont(Exception):
    pass

class Check_count(Exception):
    pass

if __name__ == '__main__':
    "compiling data"
    configure_data_files()

    "Reading data file"
    registered_all = pd.read_excel("Voters_Info.xlsx", sheet_name="Registered_Voters",index_col= False)
    hourly = pd.read_excel("Voters_Info.xlsx",sheet_name="Hourly_%",index_col=False)
    hourly_numbers = pd.read_excel("Voters_Info.xlsx",sheet_name="Hourly_numbers",index_col=False)
    # print(hourly_numbers)
    # print(registered_all.columns,hourly.columns,hourly_percent.columns)
    # print(hourly_numbers.columns)

    # new_hourly = pd.DataFrame(columns=["Hour,Location,Turnout"])
    # new_hourly_dict = {"Hour":[],"Location":[],"Turnout":[]}
    # new_hourly_numbers = hourly_numbers.drop(["Voting Mean","Voting SD","Check-in Mean","Check-in SD","Total"],axis = 1)
    # print(new_hourly_numbers)
    # cl = list(new_hourly_numbers.columns)
    # print(len(cl))
    # print(hourly_numbers)
    # for i in range(len(hourly_numbers)):
    #     for j in range(1,len(cl),1):
    #         new_hourly_dict["Location"].append(hourly_numbers["Location"][i])
    #         # print(new_hourly_dict)
    #         new_hourly_dict["Hour"].append(j+6)
    #         # print(new_hourly_dict)
    #         new_hourly_dict["Turnout"].append(hourly_numbers[cl[j]][i])
    #         # print(new_hourly_dict)
    # new_hourly_df = pd.DataFrame.from_dict(new_hourly_dict)
    # new_hourly_df.to_excel("TableauHourly.xlsx",index=False)

    column_names = ["check-in", "BMDs", "Scanners", "check-in util", "BMD util", "scanner util", "avg CT", "avg WT",
                    "99.7% WT", "avg QL", "99.7% QL"]    
    aggregateInfo = pd.DataFrame(columns=column_names)
    aggregateInfo_Baseline = pd.DataFrame(columns=column_names)
    continue_i = ContinueI()
    zero = zeroCont()
    check_cot = Check_count()


    # Simulation_runs(registered_all,hourly_numbers,aggregateInfo_Baseline)
    Baseline_perf_df = pd.read_excel("Baseline Performance Chart.xlsx", index_col = None)
    # print(Baseline_perf_df)

    Baseline_perf_dict = {Baseline_perf_df["Polling Location"][i]:Baseline_perf_df["Performance"][i] for i in range(len(Baseline_perf_df))}
    # print(Baseline_perf_dict)
    # plot_histogram(list(Baseline_perf_dict.values()),25,240,"Official Allocation")
    sum(list(Baseline_perf_dict.values()))
    Baseline_allocation = {registered_all["Location"][i]:(registered_all["Poll Pads"][i],registered_all["BMDs"][i],registered_all["Scanners"][i]) for i in range(len(registered_all))}
    # print(Baseline_allocation)
    Baseline_performance = {Baseline_perf_df["Polling Location"][i]: Baseline_perf_df["Performance"][i] for i in range(len(Baseline_perf_df))}
    # print(Baseline_performance)

    "Generating Simulation Table"
    # Multiple_Days_KPI(dailyData,aggregateInfo)

    # "Getting check-in limits"
    # d = [6,34]
    # check_in_limits = {}
    # for i in range(0,1,1):
    #     try:
    #         for ch in range(2,15,1):
    #             for bm in d:
    #                 print(i,ch,bm)
    #                 arrivals = [int(hourly_numbers["Col1"][i]), int(hourly_numbers["Col2"][i]), int(hourly_numbers["Col3"][i]), int(hourly_numbers["Col4"][i]),
    #                             int(hourly_numbers["Col5"][i]), int(hourly_numbers["Col6"][i]), int(hourly_numbers["Col7"][i]), int(hourly_numbers["Col8"][i]),
    #                             int(hourly_numbers["Col9"][i]), int(hourly_numbers["Col10"][i]), int(hourly_numbers["Col11"][i]),int(hourly_numbers["Col12"][i]),
    #                             int(hourly_numbers["Col13"][i]), int(hourly_numbers["Col14"][i]), int(hourly_numbers["Col15"][i]),
    #                             int(hourly_numbers["Col16"][i]), int(hourly_numbers["Col17"][i]), int(hourly_numbers["Col18"][i])]
    #                 params = [ch, bm, 1, arrivals, float(hourly_numbers['Check-in Mean'][i]), float(hourly_numbers['Check-in SD'][i]),
    #                           float(hourly_numbers["Voting Mean"][i]), 0.0]
    #                 mainSimRun(params)
    #                 df = pd.DataFrame(data_dict)
    #                 util = calculateUtil(df, ch, bm, 1)
    #                 machineCounts = {"poll pads": ch, "BMDs": bm, "Scanners": 1}
    #                 aggregateInfo, Perfor, wait_sig= updateAggregateInfo(hourly_numbers["Location"][i], aggregateInfo, util, df,machineCounts, 0)
    #                 if((wait_sig>=15) & (wait_sig<=120)):
    #                     check_in_limits.setdefault(i,[]).append(ch)
    #                 if(wait_sig<15):
    #                     raise check_cot
    #     except Check_count:
    #         continue
    # print(aggregateInfo)
    # print(check_in_limits)

    # checkin_30,check_in_limits_30 = finding_checkin_limits(aggregateInfo,0,30)
    # print(check_in_limits_30)
    # checkin_60,check_in_limits_60 = finding_checkin_limits(aggregateInfo,30,60)
    # checkin_90,check_in_limits_90 = finding_checkin_limits(aggregateInfo,60,90)
    # checkin_90,check_in_limits_90 = finding_checkin_limits(aggregateInfo,60,90)
    # checkin_120,check_in_limits_120 = finding_checkin_limits(aggregateInfo,90,120)
    # checkin_150,check_in_limits_150 = finding_checkin_limits(aggregateInfo,120,150)
    # checkin_180,check_in_limits_180 = finding_checkin_limits(aggregateInfo,150,180)
    # checkin_210,check_in_limits_210 = finding_checkin_limits(aggregateInfo,180,210)
    # checkin_238,check_in_limits_238 = finding_checkin_limits(aggregateInfo,210,len(hourly))
    # with open("Poll_Pad_limits_new.json", 'r') as f:
    #     limits_poll = json.load(f)
    # print(limits_poll)
    # checkin_195,check_in_limits_195 = finding_checkin_limits(aggregateInfo,195,196)

    # aggragateInfo_238.to_excel("238locs.xlsx",index = False)
    # checkin_139.to_excel("139locs.xlsx",index = False)
    # with open("Checkin139.json",'w') as f:
    #     json.dump(check_in_limits_139,f)
    # aggregate_all_intervals = pd.DataFrame(columns=column_names)
    # list_intervals = [30, 60, 90, 120,150,180,210,238,105,141,169,139,195,235]
    # check_in_limits = {str(i): 0 for i in range(0, 238, 1)}
    # for i in list_intervals:
    #     globals()["checkin_" + str(i)] = pd.read_excel(str(i) + "locs.xlsx", index_col=None)
    #     with open("Checkin" + str(i) + ".json", 'r') as f:
    #         new_check = json.load(f)
    #     check_in_limits = {**check_in_limits, **new_check}
    #     aggregate_all_intervals = pd.concat([aggregate_all_intervals, globals()["checkin_" + str(i)]],
    #                                         ignore_index=True, axis=0)
    # print(aggregate_all_intervals)
    # print(len(check_in_limits))
    # print(len(check_in_limits))
    #
    # print(registered_all["Location"][195])
    #
    # new_agg = {}
    # for i in range(len(aggregate_all_intervals)):
    #     new_agg.setdefault(aggregate_all_intervals["Polling Location"][i],[]).append(aggregate_all_intervals["Performance"][i])
    # print(new_agg)
    # outliers = []
    # new_agg = {i:max(new_agg[i]) for i in new_agg}
    # for i in new_agg:
    #     if(new_agg[i]<1):
    #         outliers.append(i)
    # print(outliers)



    # for i in range(len(registered_all)):
    #     if registered_all["Location"][i] in outliers:
    #         # globals()["checkin_"+str(i)],globals()["check_in_limits_"+str(i)] = finding_checkin_limits(aggregateInfo,i,i+1)
    #         globals()["checkin_"+str(i)].to_excel(str(i)+"locs.xlsx", index=False)
    #         with open("Checkin"+str(i)+".json", 'w') as f:
    #             json.dump(globals()["check_in_limits_"+str(i)], f)

    # print(check_in_limits)
    # print(len(set(aggregate_all_intervals["Polling Location"])))

    # aggregate_all_intervals.to_excel("All_intervals_aggregates_checkin.xlsx",index = False)
    # tot = 0
    # count = 0
    # print(check_in_limits)
    # for i in check_in_limits:
    #     j = ast.literal_eval(i)
    #     if(registered_all["Poll Pads"][j]<= min(check_in_limits[i])-1):
    #             tot = tot+1
    #             print(registered_all["Poll Pads"][j],check_in_limits[i])
    #             check_in_limits[i].append(registered_all["Poll Pads"][j])
    #     elif(registered_all["Poll Pads"][j]> max(check_in_limits[i])):
    #             count = count + 1
    #             print(registered_all["Poll Pads"][j], check_in_limits[i])
    #             check_in_limits[i].append(registered_all["Poll Pads"][j])
    # print(tot)
    # print(count)
    # print(check_in_limits)
    #
    #
    # limits_poll = {}
    # for i in check_in_limits:
    #         j = ast.literal_eval(i)
    #     # if j not in [139,195]:
    #         print(j)
    #         minimum = min(check_in_limits[i])
    #         maximum = max(check_in_limits[i])
    #         # if (minimum == maximum):
    #         #     limits_poll.update({hourly["Precinct Name (group)"][i]:[minimum]})
    #         # else:
    #         limits_poll.update({hourly["Location"][j]: [minimum, maximum]})
    #
    # print(len(limits_poll))
    # #
    # # with open("Poll_Pad_limits_238.json",'w') as g:
    # #     json.dump(limits_poll,g)
    #
    # limit_poll_dict = {"Location": [i for i in limits_poll],"Limits":list(limits_poll.values())}
    # print(limit_poll_dict)
    #
    # limit_poll_df = pd.DataFrame.from_dict(limit_poll_dict)
    # print(limit_poll_df)
    # limit_poll_df.to_excel("Poll_Pad_limits_238.xlsx",index=False)

    limits_poll = pd.read_excel("Poll_Pad_limits_238.xlsx",index_col=None)
    limits_poll = {limits_poll["Location"][i]:ast.literal_eval(limits_poll["Limits"][i]) for i in range(len(limits_poll))}
    # print(limits_poll)

    # for i in limits_poll:
    #         if (limits_poll[i]== limits_poll_new[i]):
    #             print(i)

    # with open("Poll_Pad_limits_new.json", 'r') as f:
    #     limits_poll = json.load(f)
    # print(limits_poll)

    # limits_poll = {}
    # for i in check_in_limits:
    #     minimum = min(check_in_limits[i])
    #     maximum = max(check_in_limits[i])
    #     # if (minimum == maximum):
    #     #     limits_poll.update({hourly["Precinct Name (group)"][i]:[minimum]})
    #     # else:
    #     limits_poll.update({hourly["Location"][i]: [minimum, maximum]})
    #
    # print(limits_poll)
    #
    # with open("Poll_Pad_limits.json",'w') as f:
    #     json.dump(limits_poll,f)

    # with open("Poll_Pad_limits.json",'r') as f:
    #     limits_poll = json.load(f)
    # print(limits_poll)

    # limits_poll.update({hourly["Precinct Name (group)"][5]:[1,2]})
    # print(limits_poll)
    # bmd_limits = {}
    # for i in range(0,20,1):
    #     ch = int(sum(limits_poll[hourly["Precinct Name (group)"][i]])/2)
    #     # ch = max(limits_poll[hourly["Precinct Name (group)"][i]])
    #     for bm in range(10,60,1):
    #         print(i,ch,bm)
    #         arrivals = [int(hourly["Col1"][i]), int(hourly["Col2"][i]), int(hourly["Col3"][i]), int(hourly["Col4"][i]),
    #                     int(hourly["Col5"][i]), int(hourly["Col6"][i]), int(hourly["Col7"][i]), int(hourly["Col8"][i]),
    #                     int(hourly["Col9"][i]), int(hourly["Col10"][i]), int(hourly["Col11"][i]),int(hourly["Col12"][i]),
    #                     int(hourly["Col13"][i]), int(hourly["Col14"][i]), int(hourly["Col15"][i]),
    #                     int(hourly["Col16"][i]), int(hourly["Col17"][i]), int(hourly["Col18"][i])]
    #         params = [ch, bm, 2, arrivals, float(hourly["Check-in Mean"][i]), float(hourly["Check-in SD"][i]),
    #                   float(hourly["Voting Mean"][i]), 0.0]
    #         mainSimRun(params)
    #         df = pd.DataFrame(data_dict)
    #         util = calculateUtil(df, ch, bm, 2)
    #         machineCounts = {"poll pads": ch, "BMDs": bm, "Scanners": 2}
    #         aggregateInfo, Perfor, wait_sig= updateAggregateInfo(hourly["Precinct Name (group)"][i], aggregateInfo, util, df,machineCounts, 0)
    #         if((wait_sig>=30) & (wait_sig<=200)):
    #             bmd_limits.setdefault(i,[]).append(ch)
    #         if(wait_sig<30):
    #             break
    # print(aggregateInfo)
    # print(bmd_limits)
    #
    #
    # limits_bmd = {}
    # for i in bmd_limits:
    #     # if len(bmd_limits[i])>1:
    #         limits_bmd.update({hourly["Precinct Name (group)"][i]:[min(bmd_limits[i]),max(bmd_limits[i])]})
    #     # else:
    #     #     limits_poll.update({hourly["Precinct Name (group)"][i]: [min(check_in_limits[i])-1, max(check_in_limits[i])+1]})
    # print(limits_bmd)
    #
    # with open("BMD_limits.json",'w') as f:
    #     json.dump(limits_bmd,f)

    # aggregateInfo = pd.DataFrame(columns=column_names)
    # config_df = pd.DataFrame(columns= column_names)
    # count_check = 0
    # print(limits_poll)
    # for i in range(0,20,1):
    #     lim = limits_poll[hourly["Precinct Name (group)"][i]]
    #     print(lim)
    #     arrivals = [int(hourly["Col1"][i]), int(hourly["Col2"][i]), int(hourly["Col3"][i]), int(hourly["Col4"][i]),
    #                 int(hourly["Col5"][i]), int(hourly["Col6"][i]), int(hourly["Col7"][i]), int(hourly["Col8"][i]),
    #                 int(hourly["Col9"][i]), int(hourly["Col10"][i]), int(hourly["Col11"][i]), int(hourly["Col12"][i]),
    #                 int(hourly["Col13"][i]), int(hourly["Col14"][i]), int(hourly["Col15"][i]), int(hourly["Col16"][i]),
    #                 int(hourly["Col17"][i]), int(hourly["Col18"][i])]
    #     try:  # if performance == 1
    #         for c in range(lim[0],lim[1]+1,1):
    #             count_check = count_check + 1
    #             try:
    #                 count_bmd = 0
    #                 mc = 10
    #                 best_list = []
    #                 while(mc<=50):
    #                     print(best_list)
    #                     count_bmd = count_bmd + 1
    #                     try:
    #                         count_scan = 0
    #                         for k in range(2, 6, 1):
    #                             count_scan = count_scan + 1
    #                             print(i, c, mc, k)
    #                             params = [c, mc, k, arrivals, float(hourly["Check-in Mean"][i]),
    #                                       float(hourly["Check-in SD"][i]), float(hourly["Voting Mean"][i]), 0.0]
    #                             mainSimRun(params)
    #                             df = pd.DataFrame(data_dict)
    #                             util = calculateUtil(df, c, mc, k)
    #                             machineCounts = {"poll pads": c, "BMDs": mc, "Scanners": k}
    #                             aggregateInfo,Perfor,wait_sig = updateAggregateInfo(hourly["Precinct Name (group)"][i],aggregateInfo, util, df, machineCounts,0)
    #                             if (Perfor == 1.0):
    #                                 if mc not in best_list:
    #                                     best_list.append(mc)
    #                                 if(len(best_list)>=4):
    #                                     raise check_cot
    #                             if(wait_sig>=60):
    #                                 if(count_scan>=2):
    #                                     raise zero
    #                             if(wait_sig>=40):
    #                                 if(count_scan>=2):
    #                                     # mc = mc+2
    #                                     break
    #                         # config_df, Perfor, wait_sig = updateAggregateInfo(hourly["Precinct Name (group)"][i],
    #                         #                                                   config_df, util, df, machineCounts, 0)
    #
    #                         mc = mc+2
    #                             # if (wait_sig <= 60):
    #                             #     if (count_scan >= 2):
    #                             #         if (count_bmd < 4):
    #                             #             raise zero
    #                             #         else:
    #                             #             raise check_cot
    #                     except zeroCont:
    #                         mc = mc + 10
    #             except Check_count:
    #                 continue
    #     except ContinueI:
    #         continue
    # # print(aggregateInfo["avg WT"].tolist()[0])
    # # aggregateInfo.to_csv("Simulation_results_20locs.csv")
    # print(aggregateInfo)

    # baseline_performance_df =


    # aggregateInfo = pd.DataFrame(columns=column_names)
    # aggregateInfo_30_30 = finding_all_combs(aggregateInfo,0,1,limits_poll)

    # aggregateInfo_60_60 = finding_all_combs(aggregateInfo, 30, 60)
    # aggregateInfo_90_90 = finding_all_combs(aggregateInfo,60,90)
    # aggregateInfo_120_120 = finding_all_combs(aggregateInfo,90,120)
    # aggregateInfo_150_150 = finding_all_combs(aggregateInfo,120,150)
    # aggregateInfo_180_180 = finding_all_combs(aggregateInfo,150,180)
    # aggregateInfo_210_210 = finding_all_combs(aggregateInfo,180,210)
    # aggragateInfo_238_238 = finding_all_combs(aggregateInfo,210,len(hourly))
    # print(aggregateInfo_30_30)
    # print(aggregateInfo_60_60)
    # print(aggregateInfo_30)

    # for i in range(0,20,1):
    #     y = limits_check(hourly["Precinct Name (group)"][i])
    #     try: # if performance == 1
    #         for c in range(y[0],y[1],1):
    #             count_check = count_check + 1
    #             try:
    #                 count_bmd = 0
    #                 for j in range(15,80,1):
    #                     count_bmd = count_bmd + 1
    #                     try:
    #                         count_scan = 0
    #                         for k in range(2,6,1):
    #                             count_scan = count_scan + 1
    #                             print(i,c,j,k)
    #                             arrivals = [int(hourly["Col1"][i]), int(hourly["Col2"][i]),int(hourly["Col3"][i]),int(hourly["Col4"][i]),
    #                                 int(hourly["Col5"][i]), int(hourly["Col6"][i]), int(hourly["Col7"][i]),int(hourly["Col8"][i]),
    #                                 int(hourly["Col9"][i]), int(hourly["Col10"][i]),int(hourly["Col11"][i]), int(hourly["Col12"][i]),
    #                                 int(hourly["Col13"][i]),int(hourly["Col14"][i]),int(hourly["Col15"][i]),int(hourly["Col16"][i]),int(hourly["Col17"][i]),int(hourly["Col18"][i])]
    #                             params = [c,j,k, arrivals,float(hourly["Check-in Mean"][i]),float(hourly["Check-in SD"][i]), float(hourly["Voting Mean"][i]),0.0]
    #                             mainSimRun(params)
    #                             df = pd.DataFrame(data_dict)
    #                             util = calculateUtil(df,c,j,k)
    #                             machineCounts = {"poll pads": c, "BMDs":j, "Scanners":k}
    #                             aggregateInfo,Perfor = updateAggregateInfo(hourly["Precinct Name (group)"][i],aggregateInfo, util, df, machineCounts,0)
    #                             if(Perfor == 1.0):
    #                                 raise continue_i
    #                             if(Perfor <= 0.85):
    #                                 if(count_scan>=2):
    #                                     if(count_bmd<4):
    #                                         raise zero
    #                                     else:
    #                                         raise check_cot
    #                     except zeroCont:
    #                             continue
    #             except Check_count:
    #                      continue
    #     except ContinueI:
    #             continue
    # # print(aggregateInfo["avg WT"].tolist()[0])
    # aggregateInfo.to_csv("Simulation_results_50locs.csv")
    # print(aggregateInfo)

    # Mcall_data = pd.read_csv("Multiple Day Machine KPI_NoCIN.csv")
    # Mcall_data2 = pd.read_csv("Simulation_results_20locs.csv")
    # Mcall_data = pd.read_csv("Simulation_results_20locs.csv")
    # Mcall_data = pd.read_csv("All_locations_combs.csv")
    # Mcall_data = pd.read_csv("All_congfigurations.csv")
    Mcall_data = pd.read_excel("ElectionDay_Combinations.xlsx",index_col=None)
    print(Mcall_data)
    Mcall_data = Mcall_data.drop(Mcall_data.index[Mcall_data['Performance'] <=0.6]).reset_index(drop=True)
    print(len(Mcall_data))
    # Mcall_data.drop(Mcall_data.index[Mcall_data['Performance'] == 0], inplace = True)
    # Mcall_data = Mcall_data.reset_index()
    print(Mcall_data)

    #
    #
    # # top = {}
    # # for i in range(len(Mcall_data)):
    # #     top.setdefault(Mcall_data["Polling Location"][i],[]).append(Mcall_data["Performance"][i])
    # # top = {i:min(top[i]) for i in top}
    # # # # print(top)
    # # new_list = []
    # # for i in top:
    # #     # if ((top[i] < 0.85) and (top[i] == 1)):
    # #     if (top[i] == 1):
    # #         new_list.append(i)
    # # # print(len(new_list))
    # # #
    # # Mcall_data = Mcall_data.drop(Mcall_data.index[Mcall_data['Polling Location'].isin(new_list)]).reset_index(drop=True)
    # # print(len(set(Mcall_data["Polling Location"])))
    # # print(Mcall_data.columns)
    # Mcall_data = pd.read_csv("Multiple Day Machine KPI.csv")
    Locations = list(set(Mcall_data["Polling Location"]))
    print(len(Locations))

    # print(limits_poll)
    # sum = 0
    # for i in range(len(list(limits_poll.keys()))):
    # # for i in range(0,16,1):
    #     if i not in new_list:
    #         sum = sum + limits_poll[list(limits_poll.keys())[i]][0]
    # print(sum)
    # cmbs = {}
    # bmds_c = {}
    # scns_c = {}
    # check_c = {}
    # # for i in range(0,3,1):
    # #     for l in range(3,10,1):
    # #         for j in range(10,60,1):
    # #             for k in range(2,6,1):
    # #                 cmbs.setdefault(i, []).append([l,j,k])
    # #                 # bmds_c.setdefault(i, {}).update({(j, k): j})
    # #                 # scns_c.setdefault(i, {}).update({(j, k): k})
    # #                 check_c.setdefault(i, {}).update({(l, j, k): l})
    # #                 bmds_c.setdefault(i,{}).update({(l,j,k):j})
    # #                 scns_c.setdefault(i,{}).update({(l,j,k):k})
    #
    # for idx in range(len(Mcall_data)):
    #     y = Mcall_data["Polling Location"][idx]
    #     cmbs.setdefault(y,[]).append((Mcall_data["check-in"][idx],Mcall_data["BMDs"][idx],Mcall_data["Scanners"][idx]))
    #     check_c.setdefault(y, {}).update({(Mcall_data["check-in"][idx],Mcall_data["BMDs"][idx],Mcall_data["Scanners"][idx]):(Mcall_data["check-in"][idx])})
    #     bmds_c.setdefault(y,{}).update({(Mcall_data["check-in"][idx], Mcall_data["BMDs"][idx],Mcall_data["Scanners"][idx]): (Mcall_data["BMDs"][idx])})
    #     scns_c.setdefault(y, {}).update({(Mcall_data["check-in"][idx],Mcall_data["BMDs"][idx],Mcall_data["Scanners"][idx]): (Mcall_data["Scanners"][idx])})
    #
    # # print(cmbs)
    # # # print(check_c)
    # # # print(bmds_c)
    # # # print(scns_c)
    # # #
    # new_df = pd.DataFrame()
    # new_df_bi = pd.DataFrame()
    # wait_dict = {}
    # for t in range(0,1,1):
    #     allocation_df = pd.DataFrame()
    #     perf_dict = {}
    #     count = 0
    #     for loc in Locations:
    #         count = count + 1
    #         print(count)
    #         for ind in range(len(Mcall_data)):
    #             if (Mcall_data["Polling Location"][ind] == loc) :
    #                 # perf = Performance_function(Mcall_data["avg WT"][ind],t)
    #                 perf = Performance_function(Mcall_data["99.7% WT"][ind],t)
    #                 # config_dict.setdefault(loc,[]).append([Mcall_data["BMDs"][ind],Mcall_data["99.7% WT"][ind],perf])
    #                 # perf_dict.setdefault(loc,{}).update({str([Mcall_data["BMDs"][ind],Mcall_data["Scanners"][ind]]):perf})
    #                 # wait_dict.setdefault(loc,{}).update({(Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind],t) :Mcall_data["99.7% WT"][ind]})
    #                 # wait_dict.setdefault(loc,{}).update({(Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind],t) :Mcall_data["avg WT"][ind]})
    #                 # wait_dict.setdefault(loc, {}).update({(Mcall_data["check-in"][ind], Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind],t): Mcall_data["avg WT"][ind]})
    #                 wait_dict.setdefault(loc, {}).update({(Mcall_data["check-in"][ind], Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind],t): Mcall_data["99.7% WT"][ind]})
    #                 # perf_dict.setdefault(loc,{}).update({(Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind]):perf})
    #                 perf_dict.setdefault(loc,{}).update({(Mcall_data["check-in"][ind], Mcall_data["BMDs"][ind], Mcall_data["Scanners"][ind]):perf})
    #
    #
    #     # print(wait_dict)
    #     print(perf_dict)
    #
    #     no_1 = {}
    #     for i in perf_dict:
    #         for val in perf_dict[i]:
    #             if perf_dict[i][val] == 1:
    #                 no_1.update({i: {val:perf_dict[i][val]}})
    #     # print(len(no_1))
    #     # ################################################# OPTIMIZATION MODEL ##################################################################
    #     # constraint_limits = [498,1290,450]
    #     constraint_limits = [sum(list(registered_all["Poll Pads"])),sum(list(registered_all["BMDs"])),sum(list(registered_all["Scanners"]))]
    #     # constraint_limits = [510,1350,450]
    #     gap = 0.20
    #     perf_lb = 0.75
    #     "Performance"
    #     allo_sol_p, allo_dict_p = Allocation_IP(perf_dict,"Perf",constraint_limits,gap,perf_lb)
    #     print(allo_sol_p)
    #     allocation_df["Polling Locations"] = list(allo_sol_p.keys())
    #     allocation_df["Max Performance"] = list(allo_sol_p.values())
    #     # plot_histogram(list(allo_dict_p.values()),25,250,"Performance Maximization")
    #     # plot_histogram(list(allo_dict_p.values()),20,135,"All Models")
    #
    #     "Bi-objective"
    #     allo_sol_bi, allo_dict_bi = bi_objective(perf_dict,0.15,constraint_limits,gap,perf_lb)
    #     allocation_df["Bi Objective"] = list(allo_sol_bi.values())
    #     # plot_histogram(list(allo_dict_bi.values()),25,250,"Bi-objective Model")
    #
    #     "Fairness"
    #     # allo_sol_f, allo_dict_f = Allocation_IP(perf_dict, "Fair", constraint_limits, gap,perf_lb)
    #     # allocation_df["Min Variability"] = list(allo_sol_f.values())
    #     # plot_histogram(list(allo_dict_f.values()),20,135, " Minimizing sum of variabilities in performance")
    #
    #     "Range"
    #     allo_sol_r, allo_dict_r = Allocation_IP(perf_dict, "Range", constraint_limits, gap,perf_lb)
    #     allocation_df["Min Range"] = list(allo_sol_r.values())
    #     # plot_histogram(list(allo_dict_r.values()),25,250, "Minimizing Range of Performance")
    #
    #     print(allo_sol_p)
    #     print(allo_sol_r)
    #     print(allo_sol_bi)
    #
    # #     # tot = 0
    # #     # for i in allo_sol_p:
    # #     #     if allo_sol_p[i]!= allo_sol_r[i]:
    # #     #         tot = tot + 1
    # #     # print(tot)
    # #
    # #     # wait_time = {}
    # #     # for i in allo_sol_p:
    # #     #     for j in allo_sol_p[i]:
    # #     #         # m,n= j[0],j[1]
    # #     #         # wait_time.setdefault(i, []).append([wait_dict[i][(m,n,t)], t])
    # #     #         m,n,p = j[0],j[1],j[2]
    # #     #         wait_time.setdefault(i,[]).append([wait_dict[i][(m,n,p,t)],t])
    # #     # # print(wait_time)
    # #     # wait_time_bi = {}
    # #     # for i in allo_sol_bi:
    # #     #     for j in allo_sol_bi[i]:
    # #     #         # m,n= j[0],j[1]
    # #     #         # wait_time.setdefault(i, []).append([wait_dict[i][(m,n,t)], t])
    # #     #         m,n,p = j[0],j[1],j[2]
    # #     #         wait_time_bi.setdefault(i,[]).append([wait_dict[i][(m,n,p,t)],t])
    # #     # # print(wait_time)
    # #     #
    # #     #
    # #     # if t==0:
    # #     #     new_df["Polling Location"] = list(allo_dict_p.keys())
    # #     #     # new_df["Registered Voters"] = list(subset_voters.values())
    # #     # new_df["Performance"+str(t)] = list(allo_dict_p.values())
    # #     # new_df["Allocation"+str(t)] = list(allo_sol_p.values())
    # #     # new_df["Wait_time"+str(t)] = list(wait_time.values())
    # #     #
    # #     # if t == 0:
    # #     #     new_df_bi["Polling Location"] = list(allo_dict_bi.keys())
    # #     #     # new_df["Registered Voters"] = list(subset_voters.values())
    # #     # new_df_bi["Performance" + str(t)] = list(allo_dict_bi.values())
    # #     # new_df_bi["Allocation" + str(t)] = list(allo_sol_bi.values())
    # #     # new_df_bi["Wait_time" + str(t)] = list(wait_time.values())
    # # #
    # # # # # turnover_dict = {}
    # # # # # for i in range(len(hourly)):
    # # # # #     if(hourly["Precinct Name (group)"][i] in list(perf_dict.keys())):
    # # # # #         turnover_dict.setdefault(hourly["Precinct Name (group)"][i],hourly["Total"][i])
    # # # # # print(turnover_dict)
    # # # # # new_list = list(turnover_dict.values())
    # # # # # fig = plt.figure(figsize=(20, 10))
    # # # # # ax = plt.subplot(111)
    # # # # # plt.xlim([0, 1800])
    # # # # # plt.ylim([0,8])
    # # # # # your_bins = 50
    # # # # # arr = plt.hist(new_list, bins=your_bins, histtype='stepfilled', color='teal', edgecolor='black')
    # # # # # plt.xlabel('Voter Turnout on Election Day', fontsize=15)
    # # # # # plt.ylabel('#voting locations', fontsize=15)
    # # # # # plt.xticks(fontsize=15)
    # # # # # plt.yticks(fontsize=15)
    # # # # # # if(np.sum(new_list)>= 128):
    # # # # # #     plt.xlim([0,1.1])
    # # # # # plt.title('Frequency distribution of voter turnout across polling locations ', fontsize=15)
    # # # # # plt.show()
    # # # # #     # print(new_df["Allocation0"])
    # # # # # print(new_df)
    # # # # # print(new_df_bi)
    # # # # # new_df.to_csv("Sensitivity_Performance.csv",index=None)
    # # # # # new_df.to_csv("Sensitivity_Bi_obj.csv",index=None)
    # # # #
    # # # # # sensitivity_plot(new_df,t)
    # # # # # sensitivity_plot(new_df_bi,t)
    # # # #
    # # # # # plt.xlabel("99.7% Waiting time", fontsize=15)
    # # # # # plt.ylabel('Performance', fontsize=15)
    # # # # # plt.xticks(fontsize=15)
    # # # # # plt.yticks(fontsize=15)
    # # # # # plt.title('Performance curve', fontsize=15)
    # # # # # x = np.array([0,30,55,75,90,100])
    # # # # # y = np.array([1,0.9,0.75,0.5,0.2,0])
    # # # # # plt.ylim(0,1.1)
    # # # # # plt.xlim(0,105)
    # # # # # plt.plot(x, y,'r')
    # # # # # plt.step(x, y, 'cs', where='mid')
    # # # # # # plt.bar(x,y)
    # # # # # plt.show()
    # allocation_performance = pd.DataFrame(columns= list(allocation_df.columns))
    # allocation_performance["Polling Locations"] = list(allo_dict_p.keys())
    # allocation_performance["Max Performance"] = list(allo_dict_p.values())
    # allocation_performance["Bi Objective"] = list(allo_dict_bi.values())
    # allocation_performance["Min Range"] = list(allo_dict_r.values())
    #
    # allocation_df["Official Allocation"] = [Baseline_allocation[allocation_df["Polling Locations"][i]] for i in range(len(allocation_df))]
    # allocation_performance["Official Allocation"] = [Baseline_performance[allocation_df["Polling Locations"][i]] for i in range(len(allocation_df))]
    # print(allocation_performance)
    # print(allocation_df["Official Allocation"], allocation_df["Max Performance"])


    # # # allocation_performance.to_excel("Allocation_Performance_ElectionDay_NotAllMachines.xlsx",index = False)
    # allocation_df.to_excel("ElectionDay_Allocation_Chart.xlsx",index = False)
    # # allocation_df.to_excel("ElectionDay_Allocation_Chart_NotAllMachines.xlsx",index = False)
    # print(allocation_df)
    # print(aa)
    #
    # # allocation_df = pd.read_excel("ElectionDay_Allocation_Chart.xlsx", index = False)
    # allocation_df = pd.read_excel("ElectionDay_Allocation_Chart_NotAllMachines.xlsx", index = False)
    # print(allocation_df)
    #
    # # print(allocation_performance)
    # #
    # # print(constraint_limits)
    #
    # tot = 0
    # count = 0
    # addition = 0
    # for i in range(len(allocation_df)):
    #     if(allocation_df["Max Performance"][i] != allocation_df["Bi Objective"][i]):
    #         tot = tot + 1
    #     if(allocation_df["Bi Objective"][i] != allocation_df["Min Range"][i]):
    #         count = count + 1
    #     if(allocation_df["Max Performance"][i] != allocation_df["Min Range"][i]):
    #         addition = addition + 1
    # print(tot)
    # print(count)
    # print(addition)
    # allocation_df_new.to_excel("ElectionDay_Allocation_Chart.xlsx", index = False)
    # allocation_df_new = pd.read_excel("Election_day_Allocation_Chart.xlsx",index_col = None)
    allocation_df_new = pd.read_excel("ElectionDay_Allocation_Chart.xlsx",index_col = None)
    print(allocation_df_new)
    # allocation_df_new = pd.read_excel("ElectionDay_Allocation_Chart_NotAllMachines.xlsx",index_col = None)
    # print(allocation_df_new)
    # print(allocation_df_new.columns)

    # for i in range(len(allocation_performance.columns)):

    tot = 0
    count = 0
    addition = 0
    for i in range(len(allocation_df_new)):
        if(allocation_df_new["Max Performance"][i] != allocation_df_new["Bi Objective"][i]):
            tot = tot + 1
        if(allocation_df_new["Bi Objective"][i] != allocation_df_new["Min Range"][i]):
            count = count + 1
        if(allocation_df_new["Max Performance"][i] != allocation_df_new["Min Range"][i]):
            addition = addition + 1
    # print(tot)
    # print(count)
    # print(addition)

    poll_index = {}
    reverse_poll_index = {}
    a = 0
    for i in list(allocation_df_new["Polling Locations"]):
        poll_index.setdefault(i,a)
        reverse_poll_index.setdefault(a,i)
        a = a + 1
    # print(hourly_percent["Louise Watley Library at Southeast Atlanta"])
    # print(reverse_poll_index)
    list_locs = list(poll_index.keys())
    # len(list_locs)

    chart_perf = {poll_index[allocation_df_new["Polling Locations"][i]]:allocation_df_new["Max Performance"][i] for i in range(len(allocation_df_new))}
    print(chart_perf)
    chart_biobj = {poll_index[allocation_df_new["Polling Locations"][i]]:allocation_df_new["Bi Objective"][i] for i in range(len(allocation_df_new))}
    chart_range= {poll_index[allocation_df_new["Polling Locations"][i]]:allocation_df_new["Min Range"][i] for i in range(len(allocation_df_new))}
    chart_off= {poll_index[allocation_df_new["Polling Locations"][i]]:allocation_df_new["Official Allocation"][i] for i in range(len(allocation_df_new))}
    registered = {poll_index[registered_all["Location"][i]] : registered_all["Registered Voters"][i] for i in range(len(registered_all))}
    hourly_percent_dict = {poll_index[hourly["Location"][i]]:{j:hourly[list(hourly.columns)[j]][i]*100 for j in range(1,19,1)} for i in range(len(hourly))}
    # print(hourly_percent_dict)
    # print(registered)
    # print(chart_perf)
    # print(chart_biobj)
    # print(chart_range)
    # print(hourly.columns)


    # print(hourly.columns)

    # hourly_percent_dict = {i:{} for i in chart_perf}
    # registered = {i:0 for i in chart_perf}
    # print(hourly_percent_dict)
    # tot = 0
    # print(hourly_percent)
    # for i in range(len(hourly_percent)):
    #     k = hourly_percent["Precinct Name (group)"][i]
    #     if k in list(poll_index.keys()):
    #         # if k in list(registered_all_dict.keys()):
    #             registered[i] = int(registered_all_dict[k])
    #         else:
    #             tot = tot + 1
    #             registered[i] = int(1.5*hourly_percent["Total"][i])
    #         for t in range(1,19,1):
    #             hourly_percent_dict[poll_index[k]].update({t:round(hourly_percent["Col"+str(t)][i],1)})
    # print(tot)
    # print(hourly_percent_dict)
    # print(registered)
    # print(poll_index)
    # print(reverse_poll_index)
    # hourly_percent_dict = {i:{} for i in chart_perf}
    # registered = {i:0 for i in chart_perf}
    # print(len(hourly_percent_dict))
    # tot = 0
    # print(hourly_percent)
    # for i in range(len(hourly_percent)):
    #     k = hourly_percent["Precinct Name (group)"][i]
    #     if k in list(poll_index.keys()):
    #         # if k in list(registered_all_dict.keys()):
    #             registered[i] = int(registered_all_dict[k])
    #         # else:
    #         #     tot = tot + 1
    #         #     registered[i] = int(1.5*hourly_percent["Total"][i])
    #         for t in range(1,19,1):
    #             hourly_percent_dict[poll_index[k]].update({t:round(hourly_percent["Col"+str(t)][i],1)})
    # print(tot)
    # print(hourly_percent_dict)
    # print(registered)
    #
    # for i in list(hourly_percent["Precinct Name (group)"]):
    #     if i in list(poll_index.keys()):
    #         registered[poll_index[i]] = int(registered_all_dict[i])
    #         for t in range(1,19,1):
    #             hourly_percent_dict[poll_index[i]].update({t:round(hourly_percent["Col"+str(t)][i],1)})

    # for i in range(len(hourly_percent)):
    #     k = hourly_percent["Precinct Name (group)"][i]
    #     if k in list(poll_index.keys()):
    #         registered[poll_index[k]] = int(registered_all_dict[k])
    #         for t in range(1,19,1):
    #             hourly_percent_dict[poll_index[k]].update({t:round(hourly_percent["Col"+str(t)][i],1)})
    # #
    # # print(reverse_poll_index[0])
    # # print(len(hourly_percent_dict))
    # # print(hourly_percent_dict)
    # # print(hourly_percent_dict[poll_index["Birmingham Falls Elementary"]])
    # print(hourly_percent_dict[82])
    # print(registered[82])
    #
    samp_dict = chart_perf
    column_chart = ["Location","Perf10","Perf20","Perf30","Perf40","Perf50","Perf60","Perf70","Perf80","Perf90","Perf100"]
    aggregateInfo_Perf = pd.DataFrame(columns = column_chart)
    aggregateInfo_BiObj = pd.DataFrame(columns= column_chart)
    aggregateInfo_Range = pd.DataFrame(columns = column_chart)
    aggregateInfo_off = pd.DataFrame(columns = column_chart)

    # aggregateInfo_Perf_30 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,0,30)
    # aggregateInfo_Perf_60 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,30,60)
    # aggregateInfo_Perf_90 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,60,90)
    # aggregateInfo_Perf_120 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,90,120)
    # aggregateInfo_Perf_150 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,120,150)
    # aggregateInfo_Perf_180 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,150,180)
    # aggregateInfo_Perf_210 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,180,210)
    # aggregateInfo_Perf_238 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,210,238)
    # aggregateInfo_Perf_210.to_excel("AggregatePerf_210.xlsx", index=False)

    # aggregateInfo_MaxPerf = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,0,238)
    # aggregateInfo_range = Allocation_chart_combined(chart_range,aggregateInfo_Range,registered,hourly_percent_dict,0,238)
    # print(aggregateInfo_range)
    # # aggregateInfo_biobj = Allocation_chart_combined(chart_biobj,aggregateInfo_BiObj,registered,hourly_percent_dict,0,238)
    # print(sum(aggregateInfo_biobj['Perf10']))
    # aggregateInfo_BiObj = aggregateInfo_biobj
    # aggregateInfo_BiObj.to_csv("Bi_Objective_Turnover.csv", index = False)
    # aggregateInfo_range.to_csv("Fairness_Turnover.csv", index = False)
    # aggregateInfo_Range = pd.read_excel("Fairness_Turnover.xlsx")
    # aggregateInfo_Perf_210.to_excel("AggregatePerf_.xlsx", index=False)
    # print(aggregateInfo_MaxPerf)
    # aggregateInfo_MaxPerf.to_csv("MaxPerf Turnover.csv", index= False)

    # aggregateInfo_range = pd.DataFrame(columns=column_chart)
    # aggregateInfo_off = pd.DataFrame(columns=column_chart)
    # list_intervals = [30, 60, 90, 120,150,180,210,238]
    # # check_in_limits = {str(i): 0 for i in range(0, 238, 1)}
    # # for i in list_intervals:
    # #     globals()["checkin_" + str(i)] = pd.read_excel("Aggregate_range_"+ str(i) +".xlsx", index_col=None)
    # #     aggregateInfo_range = pd.concat([aggregateInfo_range, globals()["checkin_" + str(i)]],
    # #                                         ignore_index=True, axis=0)
    # # print(aggregateInfo_range)
    # for i in list_intervals:
    #     globals()["checkin_" + str(i)] = pd.read_excel("Aggregate_off_" + str(i) + ".xlsx", index_col=None)
    #     aggregateInfo_off = pd.concat([aggregateInfo_off, globals()["checkin_" + str(i)]],
    #                                    ignore_index=True, axis=0)
    # print(aggregateInfo_off)

    # aggregateInfo_Perf_128 = Allocation_chart_combined(chart_perf,aggregateInfo_Perf,registered,hourly_percent_dict,100,len(chart_perf))
    # print(aggregateInfo_Perf)
    # aggregateInfo_Perf.to_csv("Maximize_Perf_Turnover.csv",index=False)
    #
    # aggregateInfo_BiObj = Allocation_chart_combined(chart_biobj,aggregateInfo_BiObj,registered,hourly_percent_dict,1,len(chart_biobj))
    # print(aggregateInfo_BiObj)
    # aggregateInfo_BiObj.to_csv("Bi_Obj_Turnover.csv",index=False)
    # #
    # aggregateInfo_Range = Allocation_chart_combined(chart_range,aggregateInfo_Range,registered,hourly_percent_dict,1,len(chart_range))
    # aggregateInfo_range.to_csv("Min_Range_Turnover.csv",index=False)

    aggregateInfo_off = Allocation_chart_combined(chart_off, aggregateInfo_off, registered, hourly_percent_dict,1, len(chart_range))
    aggregateInfo_off.to_csv("Off_Turnover.csv",index=False)
    print(aggregateInfo_off)

    aggregateInfo_Perf = pd.read_csv("MaxPerf Turnover.csv",index_col=None)
    print(aggregateInfo_Perf)
    aggregateInfo_BiObj = pd.read_csv("Bi_Obj_Turnover.csv",index_col=None)
    aggregateInfo_Range = pd.read_csv("Fairness_Turnover.csv",index_col=None)
    print(aggregateInfo_Range)
    aggregateInfo_off = pd.read_csv("Off_Turnover.csv",index_col=None)
    print(aggregateInfo_off)
    # # print(aggregateInfo_Perf)
    # # columns_turnover = ["Type","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"]
    column_turnover= ["Type","Perf10","Perf20","Perf30","Perf40","Perf50","Perf60","Perf70","Perf80","Perf90","Perf100"]
    new_aggregateInfo_Perf = pd.DataFrame(columns = column_turnover)
    new_aggregateInfo_Perf["Type"] = ["Max_Perf","Bi_Obj","Min_Range","Official"]
    col_list = list(new_aggregateInfo_Perf.columns)
    print(col_list)
    for i in range(1,len(col_list),1):
        new_aggregateInfo_Perf[col_list[i]] = [sum(list(aggregateInfo_Perf[col_list[i]])),sum(list(aggregateInfo_BiObj[col_list[i]])),sum(list(aggregateInfo_Range[col_list[i]])),sum(list(aggregateInfo_off[col_list[i]]))]
    print(new_aggregateInfo_Perf)

    Perf_agg = pd.DataFrame(columns=["%","Type","Perf_Value"])
    new_perf_dict = {i: [] for i in Perf_agg.columns}
    print(new_perf_dict)
    new_perf_dict["%"] = [10*i for i in range(1,11,1)]*4
    print(new_perf_dict)
    for i in range(len(new_aggregateInfo_Perf)):
        if i==0:
            for j in range(1,11,1):
                new_perf_dict["Type"].append("Max Performance")
                new_perf_dict["Perf_Value"].append(new_aggregateInfo_Perf["Perf"+str(10*j)][i])
        elif i==1:
            for j in range(1,11,1):
                new_perf_dict["Type"].append("Bi Objective")
                new_perf_dict["Perf_Value"].append(new_aggregateInfo_Perf["Perf"+str(10*j)][i])
        elif i==2:
            for j in range(1,11, 1):
                new_perf_dict["Type"].append("Min Range")
                new_perf_dict["Perf_Value"].append(new_aggregateInfo_Perf["Perf" + str(10 *j)][i])
        else:
            for j in range(1,11, 1):
                new_perf_dict["Type"].append("Official Allocation")
                new_perf_dict["Perf_Value"].append(new_aggregateInfo_Perf["Perf" + str(10 *j)][i])

    print(new_perf_dict)

    for i in Perf_agg.columns:
        Perf_agg[i] = list(new_perf_dict[i])
    print(Perf_agg)
    #
    Perf_agg.to_excel("TableauPerformancevsTurnout.xlsx",index=False)
    #
    # new_aggregateInfo_Perf.to_excel("Aggregate_Allocations.xlsx",index=False)


    # col_bmd_reg = list(new_aggregateInfo_Perf.columns)
    # print(col_bmd_reg)
    # new_perf_dict = {i:[] for i in Perf_agg.columns}
    # print(new_perf_dict)
    # for i in range(len(new_aggregateInfo_Perf)):
    #     for j in range(1,len(col_bmd_reg),1):
    #         new_bmd_reg_dict["Polling Locations"].append(bmd_reg_df["Polling Locations"][i])
    #         new_bmd_reg_dict["Type"].append(col_bmd_reg[j])
    #         new_bmd_reg_dict["BMDRegratio"].append(bmd_reg_df[col_bmd_reg[j]][i])
    # print(new_bmd_reg_dict)
    #
    # type_plot = ["Max_Perf","Bi_Obj","Min_Range"]
    # fig = plt.figure(figsize=(20, 10))
    # x = [10*i for i in range(1,11,1)]
    # m_type = ['x','v','o']
    # for i in range(len(type_plot)):
    #     y = []
    #     for j in x:
    #         y.append(new_aggregateInfo_Perf["Perf"+str(j)][i])
    #     plt.plot(x,y,marker=m_type[i],markersize =7,label = type_plot[i])
    # plt.title("Overall Performance of Polling Locations vs %Turnout of Registered Voters",fontsize = 25)
    # plt.xlabel("% Turnout of Registered Voters",fontsize = 20)
    # plt.ylabel("Overall Performance", fontsize=20)
    # plt.xticks(x,fontsize=20)
    # plt.yticks(fontsize=20)
    # # plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.legend(fontsize = 20)
    # plt.show()
    # plt.savefig("PerfvsTurnout.png")
    # #
    # print(registered_all)
    # registered = {poll_index[registered_all["Location"][i]]:registered_all["Registered Voters"][i] for i in range(len(registered_all))}
    # box_df = pd.DataFrame(columns= list(allocation_df_new.columns))
    # print(box_df.columns)
    # box_df["Polling Locations"] = list(allocation_df_new["Polling Locations"])
    # box_df["Max Performance"] = [round((ast.literal_eval(allocation_df_new["Max Performance"][i])[1])/(ast.literal_eval(allocation_df_new["Max Performance"][i])[0]),2) for i in range(len(allocation_df_new))]
    # box_df["Bi Objective"] = [round((ast.literal_eval(allocation_df_new["Bi Objective"][i])[1])/(ast.literal_eval(allocation_df_new["Bi Objective"][i])[0]),2)for i in range(len(allocation_df_new))]
    # box_df["Min Range"] = [round((ast.literal_eval(allocation_df_new["Min Range"][i])[1])/(ast.literal_eval(allocation_df_new["Min Range"][i])[0]),2) for i in range(len(allocation_df_new))]
    # box_df["Official Allocation"] = [round((ast.literal_eval(allocation_df_new["Official Allocation"][i])[1])/(ast.literal_eval(allocation_df_new["Official Allocation"][i])[0]),2) for i in range(len(allocation_df_new))]
    # print(box_df)
    # # print(hourly_percent)
    # box_df.to_excel("ED_Box_BMD_PP_NotAll.xls",index=False)
    #
    # bmd_reg_df = pd.DataFrame(columns= list(allocation_df_new.columns))
    # # print(box_df.columns)
    # bmd_reg_df["Polling Locations"] = list(allocation_df_new["Polling Locations"])
    # bmd_reg_df["Max Performance"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Max Performance"][i])[1]),2) for i in range(len(allocation_df_new))]
    # bmd_reg_df["Bi Objective"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Bi Objective"][i])[1]),2) for i in range(len(allocation_df_new))]
    # bmd_reg_df["Min Range"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Min Range"][i])[1]),2) for i in range(len(allocation_df_new))]
    # bmd_reg_df["Official Allocation"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Official Allocation"][i])[1]),2) for i in range(len(allocation_df_new))]
    # print(bmd_reg_df)
    #
    #
    # poll_reg_df = pd.DataFrame(columns= list(allocation_df_new.columns))
    # # print(box_df.columns)
    # poll_reg_df["Polling Locations"] = list(allocation_df_new["Polling Locations"])
    # poll_reg_df["Max Performance"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Max Performance"][i])[0]),2) for i in range(len(allocation_df_new))]
    # poll_reg_df["Bi Objective"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Bi Objective"][i])[0]),2) for i in range(len(allocation_df_new))]
    # poll_reg_df["Min Range"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Min Range"][i])[0]),2) for i in range(len(allocation_df_new))]
    # poll_reg_df["Official Allocation"] = [round(registered[poll_index[allocation_df_new["Polling Locations"][i]]]/(ast.literal_eval(allocation_df_new["Official Allocation"][i])[0]),2) for i in range(len(allocation_df_new))]
    # print(poll_reg_df)
    #
    # # print(hourly_percent)
    # # total_turnout = {hourly_percent["Precinct Name (group)"][i]:hourly_percent["Total"][i] for i in range(len(hourly_percent))}
    #
    # # turn_poll_reg_df = pd.DataFrame(columns= list(allocation_df_new.columns))
    # # # print(box_df.columns)
    # # turn_poll_reg_df["Polling Locations"] = list(allocation_df_new["Polling Locations"])
    # # turn_poll_reg_df["Max Performance"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Max Performance"][i])[0]),2) for i in range(len(allocation_df_new))]
    # # turn_poll_reg_df["Bi Objective"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Bi Objective"][i])[0]),2) for i in range(len(allocation_df_new))]
    # # turn_poll_reg_df["Min Range"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Min Range"][i])[0]),2) for i in range(len(allocation_df_new))]
    # # turn_poll_reg_df["Official Allocation"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Official Allocation"][i])[0]),2) for i in range(len(allocation_df_new))]
    # # print(turn_poll_reg_df)
    # #
    # # turn_bmd_reg_df = pd.DataFrame(columns= list(allocation_df_new.columns))
    # # # print(box_df.columns)
    # # turn_bmd_reg_df["Polling Locations"] = list(allocation_df_new["Polling Locations"])
    # # turn_bmd_reg_df["Max Performance"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Max Performance"][i])[1]),2) for i in range(len(allocation_df_new))]
    # # turn_bmd_reg_df["Bi Objective"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Bi Objective"][i])[1]),2) for i in range(len(allocation_df_new))]
    # # turn_bmd_reg_df["Min Range"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Min Range"][i])[1]),2) for i in range(len(allocation_df_new))]
    # # turn_bmd_reg_df["Official Allocation"] = [round(total_turnout[allocation_df_new["Polling Locations"][i]]/(ast.literal_eval(allocation_df_new["Official Allocation"][i])[1]),2) for i in range(len(allocation_df_new))]
    # # print(turn_bmd_reg_df)
    #
    #
    #
    # new_bmd_reg_df = pd.DataFrame(columns=["Polling Locations","Type","BMDRegratio"])
    # col_bmd_reg = list(bmd_reg_df.columns)
    # print(col_bmd_reg)
    # new_bmd_reg_dict = {i:[] for i in new_bmd_reg_df.columns}
    # print(new_bmd_reg_dict)
    # for i in range(len(bmd_reg_df)):
    #     for j in range(1,len(col_bmd_reg),1):
    #         new_bmd_reg_dict["Polling Locations"].append(bmd_reg_df["Polling Locations"][i])
    #         new_bmd_reg_dict["Type"].append(col_bmd_reg[j])
    #         new_bmd_reg_dict["BMDRegratio"].append(bmd_reg_df[col_bmd_reg[j]][i])
    # print(new_bmd_reg_dict)
    #
    # for i in new_bmd_reg_df.columns:
    #     new_bmd_reg_df[i] = list(new_bmd_reg_dict[i])
    # print(new_bmd_reg_df)
    #
    # new_bmd_reg_df.to_excel("ED_Box_BMD_Registered_NotAll.xlsx",index=False)
    #
    # box_plots_df(box_df,"PollPads_BMDs_NotAll")
    # box_plots_df(bmd_reg_df,"Registered_BMDs_NotAll")
    # box_plots_df(poll_reg_df,"Registered_PollPads_NotAll")
    # box_plots_df(turn_poll_reg_df,"TurnReg_PollPads")
    # box_plots_df(turn_bmd_reg_df,"TurnReg_BMDs")
