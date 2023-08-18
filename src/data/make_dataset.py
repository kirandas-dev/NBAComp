# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Load the raw data from the input CSV file
    raw_data = pd.read_csv(input_filepath)
    
    # Data processing: Remove rows with missing values
    threshold = 50000
    filtered_data = raw_data.dropna(axis=1, thresh=threshold)
    
    # Save the filtered data to a new CSV file
    interim_output_filepath = output_filepath.replace('.csv', '_interim.csv')
    filtered_data.to_csv(interim_output_filepath, index=False)
    logger.info('filtered data saved to %s', interim_output_filepath)
    
    # Data processing: Remove rows with missing values in the filtered data
    cleaned_data = filtered_data
    
    # Save the cleaned data to the output CSV file
    cleaned_data.to_csv(output_filepath, index=False)
    logger.info('processed data saved to %s', output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
