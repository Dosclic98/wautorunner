from wautorunner.scenario.scenario import ScenarioBuilder, Scenario
from wautorunner.analyzer.experiment_analyzer import ExperimentAnalyzer
from wautorunner.scenario.modifiers.modifier_interface import ModifierInterface
from wautorunner.scenario.modifiers.modifier_concrete import MultiplyLoadsModifier, MultiplyGenerationModifier, SetAllSwitchesModifier
from wautorunner.scenario.modifiers.modifier_concrete import SetSwitchesModifier, AttackerStrategyModifier, StrategyType, StrategyBuilder
from wautorunner.scenario.modifiers.modifier_concrete import MultiplyMaxCurrentModifier, SetMinMaxVoltageModifier
from wautorunner.scenario.modifiers.modifier_concrete import ExecTimeModifier
from wattson.cosimulation.control.co_simulation_controller import CoSimulationController
from wattson.cosimulation.simulators.network.emulators.wattson_network_emulator import WattsonNetworkEmulator
from wattson.iec104.common import MTU_READY_EVENT
import pandapower.topology as tp
from pandapower.plotting import simple_plotly, simple_plot
from logging import getLogger
from pathlib import Path
import time, sys, traceback, signal
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import traceback, random, shutil
import pandas as pd

class AutorunnerManager():
    DEBUG_ANALYZER: bool = False
    BASE_DELAY: float = 20.0

    def __init__(self, **kwargs):
        self.logger = getLogger("AutorunnerManager")
        self.logger.info("Building scenario")
        self.scenario: Scenario = kwargs.get("scenario", self.rebuildBaseScenario(True))

        self.modifiers: list[ModifierInterface] = kwargs.get("modifiers", 
                                                        [ExecTimeModifier(self.scenario, 40.0),
                                                         MultiplyLoadsModifier(self.scenario, 2.0), 
                                                         MultiplyGenerationModifier(self.scenario, 0.5),
                                                         MultiplyMaxCurrentModifier(self.scenario, 1),
                                                         SetMinMaxVoltageModifier(self.scenario, minVoltage=0.95, maxVoltage=1.05),
                                                         SetSwitchesModifier(self.scenario, 
                                                                             status={
                                                                                 2: False,
                                                                                 1: False,
                                                                                 4: False
                                                                             }
                                                         ),
                                                         AttackerStrategyModifier(self.scenario, strategyType=StrategyType.EXPLICIT,
                                                                                 strategy=[
                                                                                     StrategyBuilder.build(switchId=1, time=10.0, isClosed=True),
                                                                                     StrategyBuilder.build(switchId=2, time=15.0, isClosed=True),
                                                                                     StrategyBuilder.build(switchId=4, time=20.5, isClosed=True)
                                                                                 ])
                                                        ])

    def rebuildBaseScenario(self, isFirstRun: bool) -> Scenario:
        """
        Rebuilds the base scenario from the template.
        """
        scenario: Scenario = ScenarioBuilder.build(
            originPath=Path("wautorunner/scenarios/powerowl_example_template"),
            targetPath=Path("wautorunner/scenarios/powerowl_example_final"),
            resultsPath=Path("wautorunner/scenarios/powerowl_example_results"),
            clearResults=isFirstRun
        )
        self.logger.info("Scenario rebuilt")
        return scenario

    def autoBatchExecute(self, runTime: float, numRuns: int):
        """
        Generates multiple scenarios with different modifiers and executes them.
        """
        fullTraces: pd.DataFrame
        discrTraces: pd.DataFrame 
        for i in range(0, numRuns):
            runStartTime = time.time()
            self.scenario = self.rebuildBaseScenario(i == 0)
            modList: list[ModifierInterface] = self._generateNewModifiers(time=runTime)
            newFullTraces, newDiscrTraces = self._execute(modList, i)
            if i == 0: discrTraces = newDiscrTraces
            else:
                # Append new traces
                discrTraces = pd.concat([discrTraces, newDiscrTraces], ignore_index=True)
            discrTraces.to_csv(self.scenario.resultsPath.joinpath("discrTraces.csv"), index=False)
            # Add a featur to newFullTraces to track the run number for each row
            newFullTraces["run"] = i
            if i == 0: fullTraces = newFullTraces
            else:
                # Append new traces
                fullTraces = pd.concat([fullTraces, newFullTraces], ignore_index=True)
            fullTraces.to_csv(self.scenario.resultsPath.joinpath("fullTraces.csv"), index=False)
            runEndTime = time.time()
            self.logger.info(f"Finished execution {i+1}/{numRuns}")
            self.logger.info(f"Run took: {runEndTime-runStartTime} s")
            if i < numRuns-1:
                self.logger.info(f"Starting next run...")
                time.sleep(1)  # Sleep for 1 second between runs
    
    def _generateNewModifiers(self, time: float) -> list[ModifierInterface]:
        """
        Generates multiple lists of modifiers
        """
        newModifiers: list[ModifierInterface] = []
        # Generates an ExecTimeModifier based on the input parameter
        newModifiers.append(ExecTimeModifier(self.scenario, time))
        newModifiers.append(SetMinMaxVoltageModifier(self.scenario, minVoltage=0.95, maxVoltage=1.05))
        # Randomly initialize load and generation modifiers
        loadMultiplier = round(random.uniform(0.75, 1.75), 2)
        generationMultiplier = round(random.uniform(0.75, 1.75), 2)
        newModifiers.append(MultiplyLoadsModifier(self.scenario, loadMultiplier))
        newModifiers.append(MultiplyGenerationModifier(self.scenario, generationMultiplier)) 

        # Set a lower limit for the max current
        newModifiers.append(MultiplyMaxCurrentModifier(self.scenario, 0.5))

        # Generate a switch configuration where 1, 2, 4 are open and the others are closed
        switchConfig = {}
        for i in range(0, 8):
            if i == 1 or i == 2 or i == 4:
                switchConfig[i] = False
            else:
                switchConfig[i] = True
        newModifiers.append(SetSwitchesModifier(self.scenario, status=switchConfig))

        # Randomly generate an attack strategy type 
        strategyType = random.choice([StrategyType.NOACTION, StrategyType.EXPLICIT, StrategyType.INTERMITTENT, StrategyType.INTERMITTENT_OPEN, StrategyType.INTERMITTENT_CLOSED])
        startDelay = 0
        actionDelay = 0
        # Randomly generate a number of switches to be attacked
        maxSwitchIndex = 7 # Change with 5 to exclude switch 6 and 7
        numAtkSwitches = random.randint(3, maxSwitchIndex+1)
        if strategyType == StrategyType.NOACTION:
            # No action strategy, no switches to be attacked
            strategy: list[dict] = []
            startDelay = 0
            actionDelay = 0
        elif strategyType == StrategyType.EXPLICIT:
            # Randomly generate a list of numAtkSwitches switches to be attacked with their respective times
            # and new status negating the previous status
            strategy: list[dict] = []
            swIds = []
            for i in range(0, numAtkSwitches):
                switchId = random.randint(0, maxSwitchIndex)
                while switchId in swIds:
                    switchId = random.randint(0, maxSwitchIndex)
                swIds.append(switchId) 
                atkTime = round(random.uniform(5.0, time-5.0), 2)
                isClosed = not switchConfig[switchId]
                strategy.append(StrategyBuilder.build(switchId=switchId, time=atkTime, isClosed=isClosed)) 
            startDelay = 0
        elif strategyType in [StrategyType.INTERMITTENT, StrategyType.INTERMITTENT_CLOSED, StrategyType.INTERMITTENT_OPEN]:
            # Randomly generate a list of numAtkSwitches switches to be attacked
            strategy: list[int] = []
            swIds = []
            for i in range(0, numAtkSwitches):
                switchId = random.randint(0, maxSwitchIndex)
                while switchId in swIds:
                    switchId = random.randint(0, maxSwitchIndex)
                swIds.append(switchId) 
                strategy.append(switchId)
            startDelay = round(random.uniform(5.0, time/2), 2)
            actionDelay = round(random.uniform(2.0, 5.0), 2)
        
        newModifiers.append(AttackerStrategyModifier(self.scenario, strategyType=strategyType, strategy=strategy, actionDelay=actionDelay, startDelay=startDelay))
        return newModifiers
    
    def _execute(self, modifiers: list[ModifierInterface], runNumber: int) -> pd.DataFrame:
        """
        Execute the scenario with the given modifiers.
        """
        self.logger.info("Applying modifiers")
        for modifier in modifiers:
            modifier.modify()
        
        if not AutorunnerManager.DEBUG_ANALYZER:
            config = {}
            controller: CoSimulationController = CoSimulationController(self.scenario.getScenarioPath(),
                                                network_emulator=WattsonNetworkEmulator(),
                                                **config)
            controller.network_emulator.enable_management_network()
            controller.load_scenario()
            controller.start()
            self.logger.info("Wattson started!")

            def teardown(_sig, _frame):
                AutorunnerManager.stopController(controller)

            signal.signal(signalnum=signal.SIGTERM, handler=teardown)
            signal.signal(signalnum=signal.SIGINT, handler=teardown)

            try:
                # Let it run for maximum period_s seconds
                ppNet = controller._physical_simulator._grid_model._pp_net
                nxGraph: nx.Graph = tp.create_nxgraph(controller._physical_simulator._grid_model._pp_net, respect_switches=True)
                nx.draw(nxGraph, with_labels=True)
                plt.savefig(self.scenario.scenarioPath.joinpath("fullPPGraph.png"))
                simple_plot(ppNet, plot_gens=True, plot_line_switches=True, plot_loads=True, plot_sgens=True)
                plt.savefig(self.scenario.scenarioPath.joinpath("full-plot.png"))
                
                controller.join(self.scenario.getExecTime() + self.BASE_DELAY)    
            except Exception as e:
                print(traceback.format_exc())
            finally:
                AutorunnerManager.stopController(controller)

        analyzer: ExperimentAnalyzer = ExperimentAnalyzer(Path("wattson-artifacts"), self.scenario)
        fullTraces, discreteTraces = analyzer.discretizeTraces(dt=2, baseDelay=self.BASE_DELAY, totT=self.scenario.getExecTime())
        if not AutorunnerManager.DEBUG_ANALYZER:
            self.retrievePcapFile(controller, self.scenario.getPcapPath(), runNumber)
        self.logger.info("Finished execution")
        self.logger.info("Cleaning artifacts")
        shutil.rmtree(controller.working_directory)
        return fullTraces, discreteTraces

    def retrievePcapFile(self, controller: CoSimulationController, pcapFolderPath: Path, runNumber: int):
        """
        Retrieves the pcap file from the controller's working directory.
        """
        try:
            if controller.working_directory is not None:
                # Find the pcap file matching the file name "n375-service-[0-9]+-eth1.pcap"
                pcapFilePath = self._findPcapFile(controller)
                if pcapFilePath.exists():
                    # Rename the file based on the run number
                    newPcapFilePath = pcapFolderPath.joinpath(f"pcap-run-{runNumber}.pcap")
                    shutil.copy(pcapFilePath, newPcapFilePath)
                    self.logger.info(f"PCAP file copied to {pcapFolderPath}")
                else:
                    self.logger.warning("PCAP file does not exist in the controller's working directory.")
            else:
                self.logger.warning("Controller's working directory is None.")
        except Exception as e:
            self.logger.error(f"Error retrieving PCAP file: {e}")

    def _findPcapFile(self, controller: CoSimulationController) -> Path:
        """
        Finds the pcap file in the controller's working directory (the service number can change).
        """
        if controller.working_directory is not None:
            # Find the pcap file matching the file name "n375-service-[0-9]+-eth1.pcap"
            for file in controller.working_directory.joinpath("n375").glob("n375-service-*-eth1.pcap"):
                if file.is_file():
                    return file
            

    @staticmethod
    def stopController(controller: CoSimulationController):
        try:
            controller.logger.info("Stopping Wattson")
            controller.stop()
        except Exception as e:
            controller.logger.warning(f"Error during teardown occurred - trying cleanup")
            controller.logger.error(f"{e=}")
            controller.logger.error(traceback.print_exception(e))
            
            from wattson.util.clean.__main__ import main as wattson_clean
            wattson_clean()      

    def stop(self):
        """
        Method to stop the AutorunnerManager.
        """
        pass

    