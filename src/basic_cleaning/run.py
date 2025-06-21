#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info(f"Basic cleaning step is in progress {args.input_artifact}")

    with wandb.init(project="nyc_airbnb", group="basic_cleaning", job_type='basic_cleaning', save_code=True) as run:
        local_path = wandb.use_artifact(args.input_artifact).file()
        df = pd.read_csv(local_path)
        logger.info(f"Converting last_review to date")
        # Convert last_review to date
        df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')
        run.log({"last_review": type(df['last_review'][0])})

        # Drop outliers
        min_price = args.min_price
        max_price = args.max_price

        logger.info(f"Dropping values that are out of range {min_price}, {max_price}")
        idx = df['price'].between(min_price, max_price)
        df = df[idx].copy()

        logger.info(f"Easing skewness for minimum_nights")
        df['minimum_nights'] = np.log1p(df['minimum_nights']).skew()

        # Create output
        outfile = args.output_artifact
        df.to_csv(outfile, index=False)

        output_type = args.output_type
        output_description = args.output_description
        artifact = wandb.Artifact(name=outfile, type=output_type, description=output_description)
        artifact.add_file(outfile)

        run.log_artifact(artifact)

    logger.info(f"Finished Basic cleaning step {args.input_artifact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='The input artifact to be cleaned',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='The output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of the cleaned artifact',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Description of the output artifact',
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='The minimum price to consider',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='The maximum price to consider',
        required=True
    )

    args = parser.parse_args()

    go(args)
