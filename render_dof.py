from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)