from pathlib import Path
import shutil
import yaml
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

class ScenarioBuilder:
    
    @staticmethod
    def build(originPath: Path, targetPath: Path, resultsPath: Path, clearResults: bool):
        """ Logic to build the scenario """
        # Copy the scenario from originPath to targetPath
        if originPath.exists():
            shutil.copytree(originPath, targetPath, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"Origin path {originPath} does not exist.")
        pcapPath = resultsPath.joinpath("pcaps")
        if clearResults and resultsPath.exists():
            # Clear the results directory if it exists
            shutil.rmtree(resultsPath)
            resultsPath.mkdir(parents=True, exist_ok=True)
            pcapPath.mkdir(parents=True, exist_ok=True)
        elif not resultsPath.exists():
            # Create the results directory if it does not exist
            resultsPath.mkdir(parents=True, exist_ok=True)
            pcapPath.mkdir(parents=True, exist_ok=True)

        # Create a Scenario object
        return Scenario(targetPath, resultsPath, pcapPath)


class Scenario:
    def __init__(self, scenarioPath: Path, resultsPath: Path, pcapPath: Path):
        self.scenarioPath: Path = scenarioPath
        self.resultsPath: Path = resultsPath
        self.pacpPath: Path = pcapPath
        self.powerGridFilePath: Path = self.scenarioPath.joinpath("power-grid.yml")
        self.powerGridModel: dict = self.getPowerGridModel()
        self.generationFactor: float = 1.0
        self.loadFactor: float = 1.0
        self._name = self.scenarioPath.name
        self._execTime = None

    def getName(self) -> str:
        return self._name 

    def getScriptControllerConfig(self) -> dict:
        # Logic to get the script controller config file
        with open(self.scenarioPath.joinpath("extensions").joinpath("script-controller.yml"), 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
        
    def saveScriptControllerConfig(self, config: dict):
        # Logic to set the script controller config file
        with open(self.scenarioPath.joinpath("extensions").joinpath("script-controller.yml"), 'w') as file:
            file.truncate(0)
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    def getPowerGridModel(self) -> dict:
        # Logic to get the power grid file
        with open(self.powerGridFilePath, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
        
    def savePowerGridModel(self, powerGridModel: dict):
        # Logic to save the power grid file
        with open(self.powerGridFilePath, 'w') as file:
            file.truncate(0)
            yaml.dump(powerGridModel, file)

    def getNumBusses(self):
        return len(self.powerGridModel.get("elements", {}).get("bus", {}))

    def getNumLines(self):
        return len(self.powerGridModel.get("elements", {}).get("line", {}))

    def getNumSwitches(self):
        return len(self.powerGridModel.get("elements", {}).get("switch", {}))

    def getScenarioPath(self) -> Path:
        return self.scenarioPath
    
    def getResultsPath(self) -> Path:
        return self.resultsPath
    
    def getPcapPath(self) -> Path:
        return self.pacpPath
    
    def _buildSwitchesGraph(self) -> nx.DiGraph:
        # Build a networkx graph based on circuir breakers and switches
        # in the power network are connected by lines and busses

        # Create a map switch: bus
        switchBusMap = {}
        for switch in self.powerGridModel.get("elements", {}).get("switch", {}):
            switchBusMap[switch] = self.getBusIndex(self.powerGridModel["elements"]["switch"][switch]["attributes"]["PROPERTY"]["bus"])
        
        # Create a map bus: [(line, bus)] 
        # That's bidirectional!
        busLineMap = defaultdict(list)
        for line in self.powerGridModel.get("elements", {}).get("line", {}):
            fromBus = self.getBusIndex(self.powerGridModel["elements"]["line"][line]["attributes"]["PROPERTY"]["from_bus"])
            toBus = self.getBusIndex(self.powerGridModel["elements"]["line"][line]["attributes"]["PROPERTY"]["to_bus"])
            busLineMap[fromBus].append((line, toBus))
            busLineMap[toBus].append((line, fromBus))

        busTrafoMap = defaultdict(list)
        for trafo in self.powerGridModel.get("elements", {}).get("trafo", {}):
            fromBus = self.getBusIndex(self.powerGridModel["elements"]["trafo"][trafo]["attributes"]["PROPERTY"]["hv_bus"])
            toBus = self.getBusIndex(self.powerGridModel["elements"]["trafo"][trafo]["attributes"]["PROPERTY"]["lv_bus"])
            busTrafoMap[fromBus].append((trafo, toBus))
        print(busTrafoMap)

        swGraph: nx.Graph = nx.Graph()
        forbiddenEdges = [ (6, 5), (7, 2), (7, 4) ]
        for switch, bus in switchBusMap.items():
            busQueue: list = []
            busVisited = {}
            for busV in range(self.getNumBusses()):
                busVisited[busV] = False

            busQueue.append(bus)
            busVisited[bus] = True
            while len(busQueue) > 0:
                busIndex = busQueue.pop(0)

                elemTuples = busLineMap[busIndex]
                if len(elemTuples) == 0:
                    elemTuples = busTrafoMap[busIndex]
                print(f"Elem index: {busIndex}: {elemTuples}")
                for elem, toBus in elemTuples:
                    # if toBus has a switch associated with it, we can add an edge to the graph 
                    if toBus in switchBusMap.values():
                        for sw, bus in switchBusMap.items():
                            if bus == toBus and sw != switch:
                                swGraph.add_edge(switch, sw)
                    else:
                        # if toBus has no switch associated with it and it has not been visited yet, 
                        # we can add the bus to the graph
                        if not busVisited[toBus]:
                            busQueue.append(toBus)
                            busVisited[toBus] = True

            # Add an edge for all switches associated with the same bus
            for sw1, bus1 in switchBusMap.items():
                for sw2, bus2 in switchBusMap.items():
                    if sw1 != sw2 and bus1 == bus2:
                        swGraph.add_edge(sw1, sw2)
        
        # TODO Temporary solution to remove unwanted edges (must be reworked)
        swGraph.remove_edges_from(forbiddenEdges)

        return swGraph
    
    def getBusIndex(self, busName: str) -> int:
        # busName is bus.i so we want to obtain i from busName
        return int(busName.split(".")[1])
    
    def getLineIndex(self, lineName: str) -> int:
        # lineName is line.i so we want to obtain i from lineName
        return int(lineName.split(".")[1])
    
    def getExecTime(self) -> float: 
        return self._execTime
    
    def setExecTime(self, execTime: float):
        self._execTime = execTime
