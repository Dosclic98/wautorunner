
import logging.handlers


def main():
    """
    Main function to run the AutorunnerManager.
    """
    from wautorunner.manager.autorunner_manager import AutorunnerManager
    import logging, random
    from logging import StreamHandler, basicConfig, INFO, DEBUG
    from wautorunner.utils.log_formatter import ColoredFormatter

    # Configure logging
    loggerHandler = StreamHandler()
    loggerHandler.setFormatter(ColoredFormatter())
    basicConfig(level=INFO, handlers=[loggerHandler])
    # Fixing random seed for reproducibility
    random.seed(98)
    # Create an instance of AutorunnerManager
    manager = AutorunnerManager()
    # Execute the scenario
    manager.autoBatchExecute(runTime=40.0, numRuns=200)


if __name__ == '__main__':
    main()

