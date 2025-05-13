from wautorunner.scenario.scenario import ScenarioBuilder, Scenario
from wautorunner.analyzer.experiment_analyzer import ExperimentAnalyzer
from wautorunner.scenario.modifiers.modifier_interface import ModifierInterface
from wautorunner.scenario.modifiers.modifier_concrete import MultiplyLoadsModifier, MultiplyGenerationModifier, SetAllSwitchesModifier
from wautorunner.scenario.modifiers.modifier_concrete import SetSwitchesModifier, AttackerStrategyModifier, StrategyType, StrategyBuilder
from wautorunner.scenario.modifiers.modifier_concrete import MultiplyMaxCurrentModifier, SetMinMaxVoltageModifier
from wautorunner.scenario.modifiers.modifier_concrete import ExecTimeModifier
from wattson.cosimulation.control.co_simulation_controller import CoSimulationController
from wattson.cosimulation.simulators.network.emulators.wattson_network_emulator import WattsonNetworkEmulator
import pandapower.topology as tp
from pandapower.plotting import simple_plotly, simple_plot
from logging import getLogger
from pathlib import Path
import time, sys, traceback, signal
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import traceback

class AutorunnerManager():
    DEBUG_ANALYZER: bool = True

    def __init__(self, **kwargs):
        self.logger = getLogger("AutorunnerManager")
        self.logger.info("Building scenario")
        self.scenario: Scenario = kwargs.get("scenario", ScenarioBuilder.build(
            originPath=Path("wautorunner/scenarios/powerowl_example_template"),
            targetPath=Path("wautorunner/scenarios/powerowl_example_final")
        ))
        self.modifiers: list[ModifierInterface] = kwargs.get("modifiers", 
                                                        [ExecTimeModifier(self.scenario, 40.0),
                                                         MultiplyLoadsModifier(self.scenario, 2.0), 
                                                         MultiplyGenerationModifier(self.scenario, 0.5),
                                                         MultiplyMaxCurrentModifier(self.scenario, 0.5),
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

    def execute(self):
        """
        Execute the scenario with the given modifiers.
        """
        self.logger.info("Applying modifiers")
        for modifier in self.modifiers:
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
                simple_plotly(ppNet, filename=self.scenario.scenarioPath.joinpath("full-plot.html").absolute().__str__(), auto_open=False)
                controller.join(self.scenario.getExecTime())    
            except Exception as e:
                print(traceback.format_exc())
            finally:
                AutorunnerManager.stopController(controller)

        # TODO Perform log analysis
        analyzer: ExperimentAnalyzer = ExperimentAnalyzer(Path("wattson-artifacts"), self.scenario)
        traces: dict = analyzer.genTraces(2)
        print(traces)
        self.logger.info("Finished execution")

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

    