from wautorunner.scenario.scenario import Scenario
from pathlib import Path
from fnmatch import fnmatch
import pandas as pd
import json, os, glob, datetime, logging
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import numpy as np
from networkx.algorithms.cycles import simple_cycles


class ExperimentAnalyzer:
    """Analyzes the emuation results generated from a specific scenario."""

    MEASURES = {
        "bus": ["voltage"],
        "line": ["loading"]
    }

    CONFIGS = {
        "switch": ["closed"]
    }

    bussesFeeder2 = [12, 13, 14]
    linesFeeder2 = [10, 11, 14]

    def __init__(self, working_dir: Path, scenario: Scenario):
        self.logger = logging.getLogger("ExperimentAnalyzer")
        self.working_dir = working_dir
        self.scenario = scenario
        self.measures: pd.DataFrame = pd.DataFrame(columns=["timestamp", "element-type", "element-id", "measure-name", "measure"])
        self.configurations: pd.DataFrame = pd.DataFrame(columns=["timestamp", "element-type", "element-id", "config-name", "state"])
        self._loadResults()

    def genTraces(self, dt: int, baseDelay: float) -> dict:
        if self.measures.empty:
            raise ValueError("No measures data available.")
        
        self.measures.sort_index(ascending=True, inplace=True)
        self.configurations.sort_index(ascending=True, inplace=True)

        # Exclude all measures before the base delay
        self.measures = self.measures[self.measures["timestamp"] >= (self.measures["timestamp"].min() + datetime.timedelta(seconds=baseDelay))]

        startTime = self.measures["timestamp"].min()
        endTime = self.measures["timestamp"].max()
        timeBins = pd.date_range(start=startTime, end=endTime, freq=f"{dt}S")

        percAbnormalBussesF1 = []
        percAbnormalBussesF2 = []
        percOverloadedLinesF1 = []
        percOverloadedLinesF2 = []
        percNodesInCycles = []
        switchStates = []

        fullTraceDf: pd.DataFrame = pd.DataFrame()
        for i in range(len(timeBins) - 1):
            fullTraceDict = {}
            binStart = timeBins[i]
            binEnd = timeBins[i + 1]
            self.logger.debug(f"Time bin {i} - {binStart} to {binEnd}")
            measuresInTimeBin = self.measures[(self.measures["timestamp"] >= binStart) & (self.measures["timestamp"] < binEnd)]
            configurationsInTimeBin = self.configurations[(self.configurations["timestamp"] >= binStart) & (self.configurations["timestamp"] < binEnd)]
            ppNetworksInTimeBin = [network for network in self.power_networks if binStart <= network["timestamp"] < binEnd]

            if not measuresInTimeBin.empty:
                busVoltagesInTimeBin = measuresInTimeBin[(measuresInTimeBin["element-type"] == "bus") & (measuresInTimeBin["measure-name"] == "voltage")].sort_values(by="element-id")
                lineLoadingsInTimeBin = measuresInTimeBin[(measuresInTimeBin["element-type"] == "line") & (measuresInTimeBin["measure-name"] == "loading")]
                # Just take the last measure for each bus and line in the time bin
                busVoltagesInTimeBin = busVoltagesInTimeBin.groupby(["element-id", "measure-name"], as_index=False).agg({"measure": "last"})
                lineLoadingsInTimeBin = lineLoadingsInTimeBin.groupby(["element-id", "measure-name"], as_index=False).agg({"measure": "last"})
                numAbnormalBussesF1 = len(busVoltagesInTimeBin[((busVoltagesInTimeBin["measure"] < 0.95) | (busVoltagesInTimeBin["measure"] > 1.05) | (busVoltagesInTimeBin["measure"].isin([np.nan, None]))) & (~busVoltagesInTimeBin["element-id"].isin(ExperimentAnalyzer.bussesFeeder2))])
                numAbnormalBussesF2 = len(busVoltagesInTimeBin[((busVoltagesInTimeBin["measure"] < 0.95) | (busVoltagesInTimeBin["measure"] > 1.05) | (busVoltagesInTimeBin["measure"].isin([np.nan, None]))) & (busVoltagesInTimeBin["element-id"].isin(ExperimentAnalyzer.bussesFeeder2))])
                numOverloadedLinesF1 = len(lineLoadingsInTimeBin[((lineLoadingsInTimeBin["measure"] > 100) | (lineLoadingsInTimeBin["measure"].isin([np.nan, None])) & (~lineLoadingsInTimeBin["element-id"].isin(ExperimentAnalyzer.linesFeeder2)))])
                numOverloadedLinesF2 = len(lineLoadingsInTimeBin[((lineLoadingsInTimeBin["measure"] > 100) | (lineLoadingsInTimeBin["measure"].isin([np.nan, None])) & (lineLoadingsInTimeBin["element-id"].isin(ExperimentAnalyzer.linesFeeder2)))])
                numBusses = len(busVoltagesInTimeBin)
                numLines = len(lineLoadingsInTimeBin)
                percAbnormalBussesF1.append(numAbnormalBussesF1 / (numBusses-len(ExperimentAnalyzer.bussesFeeder2)) if (numBusses-len(ExperimentAnalyzer.bussesFeeder2)) > 0 else 0)
                percAbnormalBussesF2.append(numAbnormalBussesF2 / len(ExperimentAnalyzer.bussesFeeder2) if len(ExperimentAnalyzer.bussesFeeder2) > 0 else 0)
                percOverloadedLinesF1.append(numOverloadedLinesF1 / (numLines-len(ExperimentAnalyzer.linesFeeder2)) if (numLines-len(ExperimentAnalyzer.linesFeeder2)) > 0 else 0)
                percOverloadedLinesF2.append(numOverloadedLinesF2 / len(ExperimentAnalyzer.linesFeeder2) if len(ExperimentAnalyzer.linesFeeder2) > 0 else 0)

                busVoltages: dict = {}
                lineLoadings: dict = {}
                for index, row in busVoltagesInTimeBin.iterrows():
                    busVoltages[row["element-id"]] = row["measure"]
                    # Insert a new element with key BusVoltage_i and value voltage_of_bus_i in the dictionary
                    fullTraceDict[f"BusVoltage_{row['element-id']}"] = [row["measure"]]
                for index, row in lineLoadingsInTimeBin.iterrows():
                    lineLoadings[row["element-id"]] = row["measure"]
                    # Insert a new element with key LineLoading_i and value loading_of_line_i in the dictionary
                    fullTraceDict[f"LineLoading_{row['element-id']}"] = [row["measure"]]

                self.logger.debug(f"Bus voltages: {busVoltages}")
                self.logger.debug(f"Line loadings: {lineLoadings}")
                self.logger.debug(f"Number of abnormal busses Feeder 1: {numAbnormalBussesF1} / {(numBusses-len(ExperimentAnalyzer.bussesFeeder2))}")
                self.logger.debug(f"Number of abnormal busses Feeder 2: {numAbnormalBussesF2} / {len(ExperimentAnalyzer.bussesFeeder2)}")
                self.logger.debug(f"Number of overloaded lines Feeder 1: {numOverloadedLinesF1} / {(numLines-len(ExperimentAnalyzer.linesFeeder2))}")
                self.logger.debug(f"Number of overloaded lines Feeder 2: {numOverloadedLinesF2} / {(len(ExperimentAnalyzer.linesFeeder2))}")
            else:
                self.logger.debug("No measures in this time bin.")
                
            if not configurationsInTimeBin.empty:
                switchConfigsInTimeBin = configurationsInTimeBin[configurationsInTimeBin["element-type"] == "switch"]
                switchConfigsInTimeBin = switchConfigsInTimeBin.groupby(["element-id", "config-name"]).agg({"state": "last"})
                switchConfigsInTimeBin["state"] = switchConfigsInTimeBin["state"].apply(lambda x: True if x > 0.5 else False)
                switchStatesDict: dict = {}
                for index, row in switchConfigsInTimeBin.iterrows():
                    switchStatesDict[index[0]] = row["state"]
                    # Insert a new element with key SwitchClosed_i and value state_of_switch_i in the dictionary
                    fullTraceDict[f"SwitchClosed_{index[0]}"] = [row["state"]]
                self.logger.debug(f"Switch configurations: {switchStatesDict}")
                switchStates.append(switchStatesDict)
            else:
                self.logger.debug(f"No configurations in time bin {i}.")

            if len(ppNetworksInTimeBin):
                # Find unsupplied busses
                unsuppliedBusses = set()
                for network in ppNetworksInTimeBin:
                    unsuppliedBusses.update(top.unsupplied_buses(network["pp_network"]))
                    nxGraph: nx.Graph = top.create_nxgraph(network["pp_network"])
                    self.logger.debug(f"Network impedances: {nxGraph.edges.data()}:")
                self.logger.debug(f"Unsupplied busses: {unsuppliedBusses}")

                # Find cycles
                cycles = list(simple_cycles(nxGraph))
                self.logger.debug(f"Basic cycles: {cycles}")
                uniqueNodesInCycles = set([n for c in cycles for n in c])
                percInCycles = len(uniqueNodesInCycles) / len(nxGraph.nodes) if len(nxGraph.nodes) > 0 else 0
                self.logger.debug(f"Percentage of nodes in cycles: {percInCycles}")
                percNodesInCycles.append(percInCycles)

                # Count to how many cycles each bus belongs to
                busCycleCount = {n: 0 for n in nxGraph.nodes}
                for c in cycles:
                    for n in c:
                        busCycleCount[n] += 1

                # Add a BusCycles_i feature to the dictionary for each bus
                for bus_id, count in busCycleCount.items():
                    fullTraceDict[f"BusCycles_{bus_id}"] = [count]

                # A series of derived features for cycles information
                cycleLengths = [len(c) for c in cycles]
                fullTraceDict["NumCycles"] = [len(cycles)]
                fullTraceDict["PercNodesInCycles"] = [percInCycles]
                fullTraceDict["MeanCycleLength"] = [np.mean(cycleLengths)] if cycleLengths else [0]
                fullTraceDict["MaxCycleLength"] = [max(cycleLengths)] if cycleLengths else [0]
                fullTraceDict["MinCycleLength"] = [min(cycleLengths)] if cycleLengths else [0]

                # Add features for generation and load profiles
                fullTraceDict["Load"] = [self.scenario.loadFactor]
                fullTraceDict["Generation"] = [self.scenario.generationFactor]

                # Add a sequence number feature
                fullTraceDict["SeqNumber"] = [i]

                # Memorize the attack strategy type
                fullTraceDict["AttackStrategy"] = [self.scenario.getAttackStrategyType()]
            if i == 0:
                fullTraceDf = pd.DataFrame.from_dict(fullTraceDict, orient="columns")
            else:
                fullTraceDf = pd.concat([fullTraceDf, pd.DataFrame.from_dict(fullTraceDict, orient="columns")], ignore_index=True)
            self.logger.debug(" ----------------------------------------")


        return {
            "percAbnormalBussesF1": percAbnormalBussesF1,
            "percAbnormalBussesF2": percAbnormalBussesF2,
            "percOverloadedLinesF1": percOverloadedLinesF1,
            "percOverloadedLinesF2": percOverloadedLinesF2,
            "percNodesInCycles": percNodesInCycles,
            "switchStates": switchStates,
            "fullTraceDf": fullTraceDf
        }
    
    def discretizeTraces(self, dt: int, baseDelay: float, totT: float) -> dict:
        contTraces = self.genTraces(dt, baseDelay=baseDelay)
        self.logger.debug(f"Len abnormal busses Feeder 1: {len(contTraces['percAbnormalBussesF1'])}")
        self.logger.debug(f"Len abnormal busses Feeder 2: {len(contTraces['percAbnormalBussesF2'])}")
        self.logger.debug(f"Len overloaded lines Feeder 1: {len(contTraces['percOverloadedLinesF1'])}")
        self.logger.debug(f"Len overloaded lines Feeder 2: {len(contTraces['percOverloadedLinesF2'])}")
        self.logger.debug(f"Len nodes in cycles: {len(contTraces['percNodesInCycles'])}")
        self.logger.debug(f"Len switch states: {len(contTraces['switchStates'])}")

        numSteps: int = round(totT / dt)

        if (len(contTraces["percAbnormalBussesF1"]) < numSteps or len(contTraces["percAbnormalBussesF2"]) < numSteps or len(contTraces["percOverloadedLinesF1"]) < numSteps or 
                len(contTraces["percOverloadedLinesF2"]) < numSteps or len(contTraces["percNodesInCycles"]) < numSteps or len(contTraces["switchStates"]) < numSteps):
            raise ValueError("Not enough data points to discretize traces.")

        discreteDict = {}
        for j in range(1,3):
            for i in range(numSteps):
                value = ""
                if contTraces[f"percAbnormalBussesF{j}"][i] < 0.25:
                    value = "BV_0"
                elif contTraces[f"percAbnormalBussesF{j}"][i] < 0.5:
                    value = "BV_1"
                elif contTraces[f"percAbnormalBussesF{j}"][i] < 0.75:
                    value = "BV_2"
                else:
                    value = "BV_3"


                if i == 0:
                    discreteDict[f"Bus_Voltages_Feeder_{j}"] = [value]
                else:
                    discreteDict[f"Bus_Voltages_Feeder_{j}_{i}"] = [value]

        for j in range(1,3):
            for i in range(numSteps):
                value = ""
                if contTraces[f"percOverloadedLinesF{j}"][i] < 0.25:
                    value = "LL_0"
                elif contTraces[f"percOverloadedLinesF{j}"][i] < 0.5:
                    value = "LL_1"
                elif contTraces[f"percOverloadedLinesF{j}"][i] < 0.75:
                    value = "LL_2"
                else:
                    value = "LL_3"

                if i == 0:
                    discreteDict[f"Line_Loads_Feeder_{j}"] = [value]
                else:
                    discreteDict[f"Line_Loads_Feeder_{j}_{i}"] = [value]

        
        for i in range(numSteps):
            value = ""
            if contTraces["percNodesInCycles"][i] < 0.25:
                value = "NC_0"
            elif contTraces["percNodesInCycles"][i] < 0.5:
                value = "NC_1"
            elif contTraces["percNodesInCycles"][i] < 0.75:
                value = "NC_2"
            else:
                value = "NC_3"

            if i == 0:
                discreteDict["Node_Cycles"] = [value]
            else:
                discreteDict[f"Node_Cycles_{i}"] = [value]
        
        for i in range(numSteps):
            for index, state in contTraces["switchStates"][i].items():
                if i == 0:
                    discreteDict[f"Switch_{index}"] = ["C"] if state else ["O"]
                else:
                    discreteDict[f"Switch_{index}_{i}"] = ["C"] if state else ["O"]

        # Establish the first seq. number when the switch state has been modified by the attacker
        atkStartSeqNum = 0
        for i in range(numSteps):
            if i == 0:
                originalSwitchState = contTraces["switchStates"][i]
            else:
                currentSwitchState = contTraces["switchStates"][i]
                if currentSwitchState != originalSwitchState:
                    atkStartSeqNum = i
                    break
        
        for i in range(numSteps):
            from wautorunner.scenario.modifiers.modifier_concrete import StrategyType
            if i == 0:
                discreteDict["ManipulationOfControlExplicit"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.EXPLICIT.value else "N"]
            else:
                discreteDict[f"ManipulationOfControlExplicit_{i}"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.EXPLICIT.value else "N"]

        for i in range(numSteps):
            if i == 0:
                discreteDict["ManipulationOfControlAlternate"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT.value else "N"]
            else:
                discreteDict[f"ManipulationOfControlAlternate_{i}"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT.value else "N"]

        for i in range(numSteps):
            if i == 0:
                discreteDict["ManipulationOfControlOFFOnly"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT_CLOSED.value else "N"]
            else:
                discreteDict[f"ManipulationOfControlOFFOnly_{i}"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT_CLOSED.value else "N"]

        for i in range(numSteps):
            if i == 0:
                discreteDict["ManipulationOfControlONOnly"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT_OPEN.value else "N"]
            else:
                discreteDict[f"ManipulationOfControlONOnly_{i}"] = ["S" if i >= atkStartSeqNum and self.scenario.getAttackStrategyType() == StrategyType.INTERMITTENT_OPEN.value else "N"]


        # Add load and generation profiles traces
        if self.scenario.loadFactor < 1:
            discreteDict["Load"] = "L_0"
        elif self.scenario.loadFactor < 1.25:
            discreteDict["Load"] = "L_1"
        elif self.scenario.loadFactor < 1.5:
            discreteDict["Load"] = "L_2"
        else:
            discreteDict["Load"] = "L_3"

        if self.scenario.generationFactor < 1:
            discreteDict["Generation"] = ["G_0"]
        elif self.scenario.generationFactor < 1.25:
            discreteDict["Generation"] = ["G_1"]
        elif self.scenario.generationFactor < 1.5:
            discreteDict["Generation"] = ["G_2"]
        else:
            discreteDict["Generation"] = ["G_3"]
        
        return contTraces["fullTraceDf"].head(numSteps), pd.DataFrame.from_dict(discreteDict)


    def _loadResults(self):
        matchingDirs = [dir for dir in os.listdir(self.working_dir) if fnmatch(dir, f"{self.scenario.getName()}*")]
        matchingDirs.sort(reverse=False)
        scenarioWorkingDir = self.working_dir.joinpath(matchingDirs[-1])
        gridExportsPath: Path = scenarioWorkingDir.joinpath("power_grid_exports") 
        exportFiles = [gridExportsPath.joinpath(fileStr) for fileStr in glob.glob("power_grid_*.json", root_dir=gridExportsPath)]

        measures_data = []
        configs_data = []
        power_networks = []

        for filePath in exportFiles:
            with open(filePath, "r") as file:
                data: dict = json.load(file)
                timestamp = datetime.datetime.fromtimestamp(data.get("timestamp"))
                values: dict = data.get("values", {})
                ppNet: pp.pandapowerNet = pp.from_json_string(data.get("pp_net", ""))

                # Record pp network
                power_networks.append({
                    "timestamp": timestamp,
                    "pp_network": ppNet
                })

                # Process measures
                for element_type, measure_names in ExperimentAnalyzer.MEASURES.items():
                    for measure_name in measure_names:
                        for key, value in values.items():
                            if fnmatch(key, f"{element_type}.*.MEASUREMENT.{measure_name}"):
                                element_id = key.split(".")[1]
                                measures_data.append({
                                    "timestamp": timestamp,
                                    "element-type": element_type,
                                    "element-id": element_id,
                                    "measure-name": measure_name,
                                    "measure": value
                                })

                # Process configurations
                for element_type, config_names in ExperimentAnalyzer.CONFIGS.items():
                    for config_name in config_names:
                        for key, value in values.items():
                            if fnmatch(key, f"{element_type}.*.CONFIGURATION.{config_name}"):
                                element_id = key.split(".")[1]
                                configs_data.append({
                                    "timestamp": timestamp,
                                    "element-type": element_type,
                                    "element-id": element_id,
                                    "config-name": config_name,
                                    "state": value
                                })

        # Initialize DataFrames
        self.measures = pd.DataFrame(measures_data)
        self.configurations = pd.DataFrame(configs_data)
        self.power_networks = power_networks

        
        
        
