# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:23:21 2023

@author: gm17abn
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import scipy.optimize as opt



def read_file(file_name):

    df = pd.read_excel(file_name, sheet_name="Data",header=3)
    df.dropna(how="all", axis=1, inplace=True)
    df = df.drop(columns = ["Country Code","Indicator Name","Indicator Code"])
    countries_df = df.set_index("Country Name").T
    
    return countries_df, df


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   

def logistics(t, scale, growth, t0):
    """ 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

if __name__ == "__main__":
    co2_df, co2_val_df = read_file(
        "WB CO2 Emissions KT Per Capita.xlsx")
    gdp_df, gdp_val_df = read_file("WB GDP Per Capita.xlsx")
    
    print(co2_df.head(5))
    print(co2_val_df.head(5))
    print("co2 shape:",co2_val_df.shape,"GDP shape:",gdp_val_df.shape)
    single_co2_df = co2_val_df[["Country Name","1995"]]
    single_co2_df.rename(columns={"1995" : "CO2"}, inplace=True)
    single_gdp_df = gdp_val_df[["Country Name","1995"]]
    single_gdp_df.rename(columns={"1995" : "GDP"}, inplace=True)
    print(single_co2_df.head())
    print(single_gdp_df.head())
    
    xy_df = pd.merge(single_co2_df,single_gdp_df, on="Country Name")
    print(xy_df.head())
    print(xy_df.shape)
    xy_df = xy_df.dropna(how="any")
    print(xy_df.head())
    print(xy_df.shape)
    kmeans = cluster.KMeans(4)
    kmeans.fit(xy_df.drop(["Country Name"], axis=1))
    k_labels = kmeans.labels_
    xy_df["labels"] = k_labels
    xy_df.to_csv("labelled_data.csv")
    df_grouped = xy_df.groupby("labels")
    print(df_grouped.head()) 
    plt.figure()
    for label, group in df_grouped:
        plt.scatter(group["CO2"], group["GDP"], label=label)
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    plt.ylim(bottom,180000)
    plt.xlim(left,35)
    plt.xlabel("CO2 KT per Capita")
    plt.ylabel("GDP per Capita")
    plt.legend()
    plt.show()
    
    
    #curve_fit
    curve_df = gdp_df["United Kingdom"].to_frame()
    curve_df.index = curve_df.index.astype(int)
    print(curve_df.head())
    print(curve_df.values)
    print(curve_df.index.values)
    popt, covar = opt.curve_fit(logistics, curve_df.index.values,
                                curve_df["United Kingdom"], 
                                p0=(2.5e5, 0.05, 1995))

    curve_df["gdp_exp"] = logistics(curve_df.index,
                                     *popt)
    print(curve_df.head())
    #getting sigma from the covariance matrix
    sigma = np.sqrt(np.diag(covar))
    #getting the error ranges
    low, up = err_ranges(curve_df.index.values,logistics,popt,sigma)
    #creating prediction
    prediction_years = []
    for i in range(2020,2051):
        prediction_years.append(i)
    prediction_gdp = logistics(prediction_years, *popt)
    predict_low, predict_up = err_ranges(prediction_years,logistics,popt,sigma)
    plt.figure()
    plt.plot(curve_df["United Kingdom"],"+",label="Data")
    plt.plot(curve_df.index,curve_df["gdp_exp"],label="Fit")
    plt.plot(prediction_years,prediction_gdp,label="Prediction")
    plt.fill_between(curve_df.index.values,low, up, alpha=0.7)
    plt.fill_between(prediction_years, predict_low, predict_up, alpha=0.7)
    plt.title("United Kingdom")
    plt.ylabel("GDP per capita")
    plt.xlabel("Year")
    plt.legend()
    plt.show()
    
        
        
        
        
        
        