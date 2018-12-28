# kaggle Two Sigma: Using News to Predict Stock Movements

网址：https://www.kaggle.com/c/two-sigma-financial-news/data







## dataset的解释
1. Market data

    The data includes a subset of US-listed instruments. The set of included instruments changes daily and is determined based on the amount traded and the availability of information. This means that there may be instruments that enter and leave this subset of data. There may therefore be gaps in the data provided, and this does not necessarily imply that that data does not exist (those rows are likely not included due to the selection criteria).
    
    The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:

   - Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
   - Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
   - Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
   - Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.


