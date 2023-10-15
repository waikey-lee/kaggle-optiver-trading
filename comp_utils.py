import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================================
# 1. Preprocessing Function for this competition
# ========================================================================================
def clean_df(df):
    df = df.drop(columns=['row_id', 'time_id'], errors="ignore")
    df = df.rename(
        columns={
            "seconds_in_bucket": "seconds",
            "imbalance_size": "imb_size",
            "imbalance_buy_sell_flag": "imb_flag",
            "reference_price": "ref_price",
            "wap": "wa_price", 
        }
    )
    return df


# ========================================================================================
# Postprocessing Function for this competition
# ========================================================================================
def goto_conversion(listOfOdds, total=1, eps=1e-6, isAmericanOdds=False):

    # Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            else: #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100)
            listOfOdds[i] = currDecimalOdds

    # Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds')

    # Computation
    listOfProbabilities = [1/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)]
    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    # Compute standard errors assuming standard deviation is same for all stocks
    listOfSe = [x**0.5 for x in listOfVolumes]
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices

