# Import necessary libraries
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import streamlit as st
from datetime import datetime
import sys
import time
import torch
import itertools
from datetime import datetime as dt, timedelta
import random
import os
from scipy.special import expit
from scipy.optimize import curve_fit#
from itertools import cycle
from scipy.optimize import minimize   # Importing the minimize function from scipy.optimize
from scipy.optimize import approx_fprime  # Importing the approx_fprime function from scipy.optimize

st.set_page_config(layout="wide")


csss="""
        <style>
            [data-testid=stSidebarContent] {
                background-color: #AEC0DA;

            }
            [data-testid=stNotification] {
                # background-color: #b8881e;
                color:black;
                 display: flex;
                # justify-content: center; /* Horizontally center */
                # align-items: center;display: grid;
                # place-items: center;

            }
            
            div.st-emotion-cache-z2tz16.e1f1d6gn0
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-1inydi6.e1f1d6gn0
            {
            background-color: #ffffff;
            }
             div.st-be.st-b4.st-c1.st-bx.st-dl.st-dm.st-dn.st-do.st-dp.st-dq.st-dr.st-ds.st-dt
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-14teyp2.e1f1d6gn0
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-1cvtqh0.e1f1d6gn0
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-1d8vwwt.e1lln2w84
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-6mta8.e1lln2w84
            {
            background-color: #ffffff;
            }
            [data-testid=stAppViewContainer] {
                background-color: #f0f0f0;
                 
                

            }
            # button.st-emotion-cache-wasuqa.ef3psqc13
            # {
            # background-color: #A6C9EC;
            
            
            # color: black;
            # }
            [data-testid=baseButton-secondary] {
                background-color: #A6C9EC;
            
            
            color: black;

            }
            button.st-emotion-cache-wasuqa.ef3psqc13
            {
            background-color: #C00000;
            color: white;
            }
            div.st-emotion-cache-17mhcy8.e1f1d6gn0
            {
            background-color: #ffffff;
            }
            # .stButton [data-testid="baseButton-secondary"] div {
            #      # font-size: 1.5rem;
            #     background-color: rgb(247, 150, 26);
            #     color:rgb(255, 255, 255);
        
            # }
            [data-testid=stMarkdownContainer]
            {
            font-size: 12px;
            }
            
            button.st-emotion-cache-b4mwjk.ef3psqc12
            {
            background-color: #A6C9EC;
            
            color: black;
            }
             button.st-emotion-cache-1jq1og.ef3psqc13
            {
            background-color: #A6C9EC;
            
            color: black;
            }
            button.st-emotion-cache-dgkg79.ef3psqc12
            {
            background-color: #A6C9EC;
            font-weight: bold; 
            
            color: black;
            font-size: 18px;
            }
            div.st-emotion-cache-14teyp2.e1f1d6gn0
            {
            background-color: #ffffff;
            
            
            color: black;
            }
            # button.st-emotion-cache-1cp8me5.ef3psqc12
            # {
            # background-color: #ffffff;
            # padding: 5px;

      
            # color: black;
            # }
            [data-testid=stSidebarNavItems] {
                font-size: 18px;
                background-color: #ffffff;
                padding: 20px;
                margin-top: 40px; /* Adjust this value as needed */
            }
   # div.st-emotion-cache-1629p8f.e1nzilvr2
   #          {
                           
                
   #               display: flex;
   #              justify-content: center; /* Horizontally center */
   #              align-items: center;display: grid;
   #              place-items: center;
   #              padding-bottom: 5px;
               

   #          }
     </style>
        """
st.markdown(csss, unsafe_allow_html=True)


def calculate_revenue(x, c1, c2, c3, c4, curve_type, CPM = 1000, adjustment_factor = 1, ref_adj_fctr = 1, 
                      tactic_adj_fctr = 1, seasonal_adj_fctr = 1, ADSTOCK_X = 1, ECOMM_ROI = 1):
    if CPM == 0:
        return 0
    if round(x) <= 1:
        return 0
    if curve_type == "Hill":
        return hill(x, c1, c2, c3, c4, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    elif curve_type == "Logistic":
        return logistic(x, c1, c2, c3, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    elif curve_type == "Power":
        return power(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    else:
        return NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    
def hill(x, c1, c2, c3, c4, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    x = x / CPM
    #return ((adjustment_factor * ref_adj_fctr * (c1 + ((c2 - c1) * ((x / CPM) + ADSTOCK_X) ** c3) / (c4**c3 + ((x / CPM) + ADSTOCK_X) ** c3))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    # return ((adjustment_factor * ref_adj_fctr * c1 / (1 + expit(c2 * (x - c3)))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return c1 + ((c2 - c1) * (np.power(x, c3))) / (np.power(c4, c3) + np.power(x, c3))


# Logistic function
def logistic(x, c1, c2, c3, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    
    x = x / CPM
    #return ((adjustment_factor * ref_adj_fctr * (c1 / (1 + expit(c2 * (((x / CPM) + ADSTOCK_X) - c3))))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    #return ((adjustment_factor * ref_adj_fctr * (c1 / (1 + expit(c2 * (x - c3))))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return c1 / (1 + np.exp(c2 * (x - c3)))

# Power function
def power(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    x = x/CPM
    #return ((adjustment_factor * ref_adj_fctr * (np.exp(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return np.exp(c1) * (x ** c2) 


# NullCurve function
def NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    return ((adjustment_factor * ref_adj_fctr * (expit(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr

def objective_to_minimize(x, df):
    idx = df.index.values
    C1_values = df['C1'].values
    C2_values = df['C2'].values
    C3_values = df['C3'].values
    C4_values = df['C4'].values
    CURVE_TYPE_values = df['CURVE_TYPE'].values
    CPM_values = df['CPM'].values
    adjustment_factor_values = df['ADJUSTMENT_FACTOR'].values
    ref_adj_fctr_values = df['REF_ADJ_FCTR'].values
    tactic_adj_fctr_values = df['TACTIC_ADJ_FCTR'].values
    seasonal_adj_fctr_values = df['SEASONAL_ADJ_FCTR'].values
    ADSTOCK_X_values = df['ADSTOCK_X'].values
    ECOMM_ROI_values = df['ECOMM_ROI'].values
    
    revenue = np.zeros(len(df))
    for i in range(len(df)):
        revenue[i] = calculate_revenue(x[idx[i]], C1_values[i], C2_values[i], C3_values[i], C4_values[i], CURVE_TYPE_values[i], CPM_values[i], adjustment_factor_values[i], ref_adj_fctr_values[i], tactic_adj_fctr_values[i], seasonal_adj_fctr_values[i], ADSTOCK_X_values[i], ECOMM_ROI_values[i])
    # print("Profit :",np.sum(revenue),"Spend :",np.sum(x))
    return -np.sum(revenue)

# rollup_name_qry=f"""select ROLLUP_NAME,sonic_batch_id from ANALYTICS.UMM_SONIC.UMM_SONIC_BATCH_ROLLUP_LKP where sonic_batch_id is not null order by sonic_batch_id desc limit 1 """
# rollup_name=session.sql(rollup_name_qry).to_pandas()
# st.session_state['rollup_name']=rollup_name
# BUDGET_BATCH_ID=rollup_name["SONIC_BATCH_ID"][0]

def get_snowflake_data():

    #data = pd.read_csv(r'C:\Users\Prakhar.saxena\Downloads\Amout_Calc_22-7-25.csv')
    data = pd.read_excel(r'Input_data.xlsx')

    
    # column_lengths = {col: len(col) for col in data.columns}
    # st.write(data.head())

    # data['MONTH_YEAR'] = pd.to_datetime(data['MONTH_YEAR'])
    # data['Yearly'] = data['Yearly'].astype(str)
    # data['Quarterly'] = data['Quarterly'].astype(str)
    # data['Monthly'] = data['Monthly'].dt.strftime('%b-%y')

    data['MONTH_YEAR'] = pd.to_datetime(data['BUDGET_WEEK_START_DT'])
    data['Yearly'] = data['MONTH_YEAR'].dt.year.astype(str)  # → "2025"
    data['Monthly'] = data['MONTH_YEAR'].dt.strftime('%b-%y')  # → "Jan-25"
    data['Quarterly'] = 'Q' + data['MONTH_YEAR'].dt.quarter.astype(str) + '-' + data['MONTH_YEAR'].dt.year.astype(str).str[-2:]
    
    data.rename(columns={'Spend': 'Actual Spend (Selected Months)',
                         'TACTIC' : 'Tactic'}, inplace=True)
    
    data['Subbrand'] = 'Brand1'
    data['FEC'] = data['Actual Profit']
    data['Publishername'] = 'Custom_Publish'
    data['Channel'] = "Channel1"
    data['ADSTOCK_WEEK'] = 1
    data['ADSTOCK'] = 0
    data['ADSTOCK_X'] = 1
    data["ADJUSTMENT_FACTOR"] = 1
    data["REF_ADJ_FCTR"] = 1
    data["ECOMM_ROI"] = 1
    data["TACTIC_ADJ_FCTR"] = 1
    data["SEASONAL_ADJ_FCTR"] = 1
    data['FEC_FCTR'] = 1
    data['LTE_FCTR'] = 1
    data['CPM_Original'] = data['CPM']
    data['Group Tactic'] = 'Group'
    data['MIN_BOUND']=-30
    data['MAX_BOUND']=30
    data['Actual Profit'] = data['Calculated']

    data.loc[:, "ADSTOCK_X_Ana"] = 0.0
    for (brand, tactic), group in data.groupby(["Brand", "Actual Tactic"]):
        max_week = int(group["ADSTOCK_WEEK"].max())
    
        for i in range(1, max_week + 1):
            group["ADSTOCK_X_Ana"] += (
              # ((group['Actual Spend (Selected Months)'] / group["CPM"]).shift(-1 * i).fillna(0))
                 np.where(group["CPM"]==0,0,((group['Actual Spend (Selected Months)'] / group["CPM"]).shift(-1 * i).fillna(0)))
                * ((group["ADSTOCK"]) ** (i - 1))
                * (group["ADSTOCK"])
            )
    
        # Update the 'ADSTOCK_X' column in the original dataframe for this group
        data.loc[group.index, "ADSTOCK_X_Ana"] = group["ADSTOCK_X_Ana"] 

    data["Actual FEC"] = data['Actual Profit']*data['FEC_FCTR']
    
    data["Actual LTE Profit"]=data["Actual Profit"]*data["LTE_FCTR"]
    data["Actual LTE FEC"]=data["Actual LTE Profit"]*data["FEC_FCTR"]
    data = data[data['Yearly'] == '2025']
    data['Brand'] = "C5i"

    # st.write(data.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                'Actual Profit' : 'sum'}))
    
    # st.write("SNOWFLAKE")
    # st.dataframe(data)

    data = data[['Brand','Level 1','Level 2','Level 3', 'Subbrand', 'Tactic', 'Actual Tactic', 'Group Tactic','Publishername', 'Channel', 'BUDGET_WEEK_START_DT', 'Actual Spend (Selected Months)','CPM', 'CURVE_TYPE',
                 'C1', 'C2', 'C3', 'C4', 'Actual Profit', 'Impression', 'MONTH_YEAR', "Yearly", 'Monthly', 'Quarterly', 'ADSTOCK', 'ADSTOCK_X' ,'ADSTOCK_WEEK', 'ADSTOCK_X_Ana',
                 'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'ECOMM_ROI', 'TACTIC_ADJ_FCTR', 'SEASONAL_ADJ_FCTR', "FEC_FCTR", "LTE_FCTR", 'CPM_Original', 'Actual FEC', 'MIN_BOUND', 
                 'MAX_BOUND', 'Actual LTE Profit', 'Actual LTE FEC']]

    st.session_state.snapshot_data = data
    st.session_state.sim_snowflake_data = data

    return data, data


# def old_get_snowflake_data():
    
#     if "error_comment_message" not in st.session_state:
#         st.session_state["error_comment_message"] = ""
#     if "comment_input" not in st.session_state:
#         st.session_state["comment_input"] = ""

#     mmm_df = pd.read_csv(r'02_MODELING_HBR_LEVEL1_SK_M_MS_output.csv')
#     curve_df = pd.read_csv(r'Best_Curve_Coefficients.csv')
   
#     data=curve_df.copy()
#     # data = pd.merge(mmm_df, curve_df, left_on= 'key' ,right_on='TACTIC', how='left')
#     data = data[data['key'] != 'Paid Traditional']
#     data.rename(columns={'spends': 'Actual Spend (Selected Months)',
#                              'nsv_value' : 'FEC',
#                              'brand' : 'Brand',
#                              'sub_brand_x' : 'Subbrand',
#                              'TACTIC' : 'Actual Tactic'}, inplace=True)
    
#     data['date'] = pd.to_datetime(data['date'])
#     data['date'] = data['date'] + DateOffset(years=1)
#     data['Yearly'] = data['date'].dt.year.astype(str)  # → "2025"
#     data['Monthly'] = data['date'].dt.strftime('%b-%y')  # → "Jan-25"
#     data['Quarterly'] = 'Q' + data['date'].dt.quarter.astype(str) + '-' + data['date'].dt.year.astype(str).str[-2:]
#     data['Channel'] = data['Actual Tactic']
#     data['Publishername'] = 'Custom_Publish'
#     data['BUDGET_WEEK_START_DT'] = data['date']
#     data['ADSTOCK_WEEK'] = 1
#     data['ADSTOCK_X'] = 1
#     data['CPM'] = 50
#     data['ADSTOCK'] = 0
#     data["ADJUSTMENT_FACTOR"] = 1
#     data["REF_ADJ_FCTR"] = 1
#     data["ECOMM_ROI"] = 1
#     data["TACTIC_ADJ_FCTR"] = 1
#     data["SEASONAL_ADJ_FCTR"] = 1
#     data['FEC_FCTR'] = 1
#     data['LTE_FCTR'] = 1
#     data['MONTH_YEAR'] = data['date']
#     data['Tactic'] = data['Actual Tactic']
#     data['CPM_Original'] = data['CPM'] * 0.6
#     data['Group Tactic'] = 'Group'
#     data = data[data['date'].dt.year == 2025]

#     data.reset_index(drop=True, inplace=True)
#     data=data.sort_values(by=['Brand', 'Actual Tactic', 'BUDGET_WEEK_START_DT'], ascending=[True, True, False])
    
#     data.loc[:, "ADSTOCK_X_Ana"] = 0.0
#     for (brand, tactic), group in data.groupby(["Brand", "Actual Tactic"]):
#         max_week = int(group["ADSTOCK_WEEK"].max())
    
#         for i in range(1, max_week + 1):
#             group["ADSTOCK_X_Ana"] += (
#               # ((group['Actual Spend (Selected Months)'] / group["CPM"]).shift(-1 * i).fillna(0))
#                  np.where(group["CPM"]==0,0,((group['Actual Spend (Selected Months)'] / group["CPM"]).shift(-1 * i).fillna(0)))
#                 * ((group["ADSTOCK"]) ** (i - 1))
#                 * (group["ADSTOCK"])
#             )
    
#         # Update the 'ADSTOCK_X' column in the original dataframe for this group
#         data.loc[group.index, "ADSTOCK_X_Ana"] = group["ADSTOCK_X_Ana"] 
    
#     data['Actual Profit']=[calculate_revenue(row['Actual Spend (Selected Months)'], 
#                                              row["C1"], row["C2"], row["C3"], row["C4"], 
#                                              row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], 
#                                              row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], row["SEASONAL_ADJ_FCTR"], 
#                                              row["ADSTOCK_X_Ana"], row["ECOMM_ROI"]) for i, row in data.iterrows()]

#     data['Actual Profit'] = data['Actual Profit'] / 100

     
#     data["Actual FEC"] = data['Actual Profit']*data['FEC_FCTR']
    
#     data["Actual LTE Profit"]=data["Actual Profit"]*data["LTE_FCTR"]
#     data["Actual LTE FEC"]=data["Actual LTE Profit"]*data["FEC_FCTR"]
#     data['MIN_BOUND']=-30
#     data['MAX_BOUND']=30
    
#     st.session_state.snapshot_data = data
#     snapshot_data = data
#     sim_snowflake_data = data

#     # st.write("GET SNOWFLAKE")
#     # st.dataframe(data)
#     return snapshot_data, sim_snowflake_data


output_path = "Output_data.csv"

if os.path.exists(output_path):
    try:
        check_df = pd.read_csv(output_path)
        #st.write(check_df.shape)
        comments_list = check_df['COMMENTS'].unique().tolist()
    except pd.errors.EmptyDataError:
        check_df = pd.DataFrame()
        comments_list = []
else:
    #st.warning(f"File not found: {output_path}")
    check_df = pd.DataFrame()
    comments_list = []

df1, df2 = get_snowflake_data()

BUDGET_BATCH_ID = 217
rollup_name = "2025_05 Rollup"

#st.dataframe(df1, use_container_width=True, hide_index=True)
#st.dataframe(df2, use_container_width=True, hide_index=True)
#st.dataframe(df3, use_container_width=True, hidbe_index=True)
#st.dataframe(df4, use_container_width=True, hide_index=True)

#data = df1.copy()
data = st.session_state.snapshot_data
data[["COEF_ADJU","CPM_ADJU"]]=1
brands = data['Brand'].to_numpy()
actual_tactics = data['Actual Tactic'].to_numpy()
group_tactics = data['Group Tactic'].to_numpy()

unique_brands = np.unique(brands)
unique_actual_tactics = np.unique(actual_tactics)
unique_group_tactics = np.unique(group_tactics)

with st.sidebar.container():
    #st.write(f"Batch Id: **{str(BUDGET_BATCH_ID)}** -> **{st.session_state['rollup_name']['ROLLUP_NAME'].iloc[0]}**")
    st.write(f"Batch Id: **{str(BUDGET_BATCH_ID)}** -> **{rollup_name}**")

with st.sidebar.container(border=True):

    #new_or_exi =st.selectbox(" New or Existing Scenario:", ['New Scenario','Existing Scenario']) 
    new_or_exi =st.selectbox(" New or Existing Scenario:", ['New Scenario']) 


# if new_or_exi=="Existing Scenario":
#     query_exi_scn=f"""    select distinct "Brand" as BRAND, "comments" as COMMENTS from  ANALYTICS.UMM_OPTIM.PROD_SONIC_OPTIMIZATION_INPUT_TABLE"""
#     exi_sce = session.sql(query_exi_scn).collect()
#     for_filter=pd.DataFrame(exi_sce)
#     with st.sidebar.container(border=True):
#         brand_filter = st.selectbox("Brand Name:", sorted(for_filter['BRAND'].unique().tolist())
                                    
#     )
#     with st.sidebar.container(border=True):
#         comments_filter = st.selectbox("Scenario:", sorted(for_filter[for_filter['BRAND']==brand_filter]['COMMENTS'].unique().tolist())
                                    
#     )
        
if new_or_exi == "New Scenario":
    
    # Streamlit sidebar for brand selection
    with st.sidebar.container(border=True):
        brand_filter = st.selectbox("Brand Name:", sorted(unique_brands.tolist()),index=0
    )
    
    # Filter data based on brand selection
    brand_filter_indices = np.isin(brands, [brand_filter])
    
    
    # with st.sidebar.container(border=True):
        
    #     brand_filter =st.multiselect(   " Brand Name:",data["Brand"].unique(),"Aleve"
        
    #     )
    # st.write()
    # ,"LT FEC","MULTI KPIs"
    with st.sidebar.container(border=True):
        target_mode =st.selectbox("Choose the Target Optimization type",["Revenue"])
        # ,"MULTI KPIs"
    
    with st.sidebar.container(border=True):
        # if target_mode=="MULTI KPIs":
        #     Opt_type_list=["Target FEC"]
        # else:
            # Opt_type_list=["Budget Adjustment"]
            # ,"Target FEC"
        Opt_type_list=["Budget Adjustment" ,"Target Revenue"]

        opt_type=st.selectbox("Choose the Optimization type",Opt_type_list)
        
        if opt_type=="Target FEC":
            Tactic_Type_filter_list=['UMM Tactic Only'] #,'EmPlanner Tactic'

        else:
            Tactic_Type_filter_list=['UMM Tactic Only'] #,'Combo' , 'EmPlanner Tactic'
    
    with st.sidebar.container(border=True):
        Tactic_Type_filter =st.selectbox(" Input Type :", Tactic_Type_filter_list)
    
    with st.sidebar.container(border=True):
        if Tactic_Type_filter =='UMM Tactic Only':
            tactic_list = np.unique(actual_tactics[brand_filter_indices])
            Tactic_filter = st.multiselect("Tactic Name:", ["Select All"] + tactic_list.tolist(), default="Select All")
            if "Select All" in Tactic_filter:
                Tactic_filter = tactic_list.tolist()
    
        elif Tactic_Type_filter =='EmPlanner Tactic':
            tactic_list = np.unique(group_tactics[brand_filter_indices])
            Tactic_filter = st.multiselect("Tactic Name:", ["Select All"] + tactic_list.tolist(), default="Select All")
            if "Select All" in Tactic_filter:
                Tactic_filter = tactic_list.tolist()
    
        elif Tactic_Type_filter == 'Combo':
            tactic_list = np.unique(np.concatenate([group_tactics[brand_filter_indices], actual_tactics[brand_filter_indices]]))
            Tactic_filter = st.multiselect("Tactic Name:", ["Select All"] + tactic_list.tolist(), default="Select All")
            if "Select All" in Tactic_filter:
                Tactic_filter = tactic_list.tolist()

    with st.sidebar.container(border=True):
        
        Period_filter =st.selectbox(   " Period Type:", ['Yearly','Quarterly','Monthly']
    
    )
    
    with st.sidebar.container(border=True):
        
        Yr_filter =st.selectbox(   " Year list:", data['Yearly'].unique().tolist()
    
    )
    
    with st.sidebar.container(border=True):
    
        lock_act_month=st.toggle("Lock Actualized Months",value=True) # original
        #lock_act_month=st.toggle("Lock Actualized Months",value=False)
        
        # month_list=["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
        if Yr_filter=='2022':
            month_list=["Jan-22", "Feb-22", "Mar-22", "Apr-22", "May-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22", "Oct-22", "Nov-22", "Dec-22"]
        elif Yr_filter=='2023':
            month_list=["Jan-23", "Feb-23", "Mar-23", "Apr-23", "May-23", "Jun-23", "Jul-23", "Aug-23", "Sep-23", "Oct-23", "Nov-23", "Dec-23"]
        elif Yr_filter=='2024':
            month_list=["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]

        else:
            if lock_act_month==False:
                month_list = ["Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"]
            else:
                month_list = [  "Jan-25","Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"]
    
        Month_filter =st.multiselect(   " Month list:",["Select All"]+ month_list,default="Select All")
        # Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
        if "Select All" in Month_filter:
            Month_filter=month_list
        st.session_state["Month_filter"]=Month_filter
        st.session_state["Yr_filter"]=Yr_filter
        st.session_state["comments_filter"]=''


with st.sidebar.container():
    Apply_button=st.button('Apply')

def brand_logo():
    # Check if Afrin is in the list
    if 'Afrin' ==brand_filter:
        result = 'https://www.afrin.com/sites/g/files/vrxlpx50106/files/afrin-new-header-logo.png'
    # Check if Aleve is in the list
    elif 'Aleve' ==brand_filter:
        result = 'https://www.aleve.com/sites/g/files/vrxlpx48721/files/aleve_logo_.png'
    # Check if Alka Seltzer Original is in the list
    elif 'Alka Seltzer Original' ==brand_filter:
        result = 'https://www.alkaseltzer.com/sites/g/files/vrxlpx50686/files/alka-seltzer-logo.png'
    # Check if Alka Seltzer Plus is in the list
    elif 'Alka Seltzer Plus' ==brand_filter:
        result = 'https://www.alkaseltzer.com/sites/g/files/vrxlpx50686/files/ASP-logo.png'
    # Check if Bayer Aspirin is in the list
    elif 'Bayer Aspirin' ==brand_filter:
        result = 'https://www.bayeraspirin.com/sites/g/files/vrxlpx46941/files/Bayer_Aspirin_Logo_New%20v2.png'
    # Check if Claritin is in the list
    elif 'Claritin' ==brand_filter:
        result = 'https://www.claritin.com/sites/g/files/vrxlpx41731/files/2022-11/claritin-logo-blue.png'
    # Check if Coricidin is in the list
    elif 'Coricidin' ==brand_filter:
        result = 'https://www.coricidinhbp.com/sites/g/files/vrxlpx50086/files/coricidin-HBP-logo_0.png'
    # Check if Flintstones is in the list
    elif 'Flintstones' ==brand_filter:
        result = 'https://www.flintstonesvitamins.com/sites/g/files/vrxlpx47286/files/flintstones-logo_0.png'
    # Check if Lotrimin is in the list
    elif 'Lotrimin' ==brand_filter:
        result = 'https://www.lotrimin.com/sites/g/files/vrxlpx50606/files/Lotrimin%20Original%20Logo-152x42.png'
    # Check if Miralax is in the list
    elif 'Miralax' ==brand_filter:
        result = 'https://www.miralax.com/sites/g/files/vrxlpx36946/files/2021-01/miralax-color-logo.png'
        # Check if One A Day is in the list
    elif 'Astepro' ==brand_filter:
        result = 'https://www.asteproallergy.com/sites/g/files/vrxlpx50111/files/astepro-300x100-logo-bro-TM.png'
    # Check if Phillips is in the list
        # Check if One A Day is in the list
    elif 'Midol' ==brand_filter:
        result = 'https://www.midol.com/sites/g/files/vrxlpx50716/files/midol-logo-header-v2.png'
    # Check if One A Day is in the list
    elif 'One A Day' ==brand_filter:
        result = 'https://www.oneaday.com/sites/g/files/vrxlpx50456/files/2020-07/OAD-color-logo_0.png'
    # Check if Phillips is in the list
    elif 'Phillips' ==brand_filter:
        result = 'https://www.phillipsdigestive.com/sites/g/files/vrxlpx50116/files/2023-04/Phillips%20Logo%20EBU%20-%20Blue%20Wave%20-%202022%20-%20RGB_0.png'
    else:
        result='https://www.c5i.ai/wp-content/themes/course5iTheme/new-assets/images/c5i-primary-logo.svg'

    return result
    
#st.dataframe(data.head(3))
if Apply_button:
   

    if new_or_exi=='Existing Scenario':
        
        # query_exi_scn=f"""    select * from  ANALYTICS.UMM_OPTIM.PROD_SONIC_OPTIMIZATION_INPUT_TABLE where "Brand"='{brand_filter}' and "comments" ='{comments_filter}' """
        # exi_sce = session.sql(query_exi_scn).collect()
        # s_df=pd.DataFrame(exi_sce)
        
        # s_df=s_df.rename(columns={'PERIOD':s_df['PERIOD_TYPE'][0]})
        
        
        
        
        # st.session_state["target_mode"]=s_df['TARGET_OPTIM_TYPE'][0]
        # target_mode=s_df['TARGET_OPTIM_TYPE'][0]
        # st.session_state["opt_type"]=s_df['OPTIM_TYPE'][0]
        # opt_type=s_df['OPTIM_TYPE'][0]
        # st.session_state["Tactic_Type_filter"]=s_df['INPUT_TYPE'][0]
        # Tactic_Type_filter=s_df['INPUT_TYPE'][0]
        # Tactic_filter=s_df['Tactic'].unique().tolist()
        # st.session_state["Period_filter"]=s_df['PERIOD_TYPE'][0]
        # Period_filter=s_df['PERIOD_TYPE'][0]
        # st.session_state["Yr_filter"]=s_df['YEAR_LIST'][0]
        # st.session_state["Month_filter"]=s_df['MONTH_LIST'][0].split(', ')
        # st.session_state["comments_filter"]=comments_filter
        # st.session_state['Exi_Total_value']=s_df['Total_value'][0]
        # st.session_state['Exi_Extra_value']=s_df['Extra_value'][0]
        # s_df_cols=["Brand","Tactic",s_df['PERIOD_TYPE'][0],"Actual Spend (Selected Months)", "MIN_BOUND","MAX_BOUND","COEF_ADJU","CPM_ADJU","MIN_BOUND_Spend","MAX_BOUND_Spend"]
        # st.session_state['input_data']=s_df[s_df_cols
        pass
    else:
        st.session_state.img=brand_logo()
        st.session_state["optim_brand_filter"]=brand_filter
        st.session_state["Period_filter"]=Period_filter
        st.session_state["Tactic_Type_filter"]=Tactic_Type_filter
        st.session_state["Tactic_filter"]=Tactic_filter
        st.session_state["opt_type"]=opt_type
        st.session_state["target_mode"]=target_mode
        st.session_state["Month_filter"]=Month_filter
        st.session_state["Yr_filter"]=Yr_filter
        comments_filter=''
        
def all_filters(data):

    #st.write("ALL FILTERS")

    st.session_state["optim_brand_filter"]=brand_filter
    st.session_state["Period_filter"]=Period_filter
    st.session_state["Tactic_Type_filter"]=Tactic_Type_filter
    st.session_state["Tactic_filter"]=Tactic_filter
    st.session_state["opt_type"]=opt_type
    st.session_state["target_mode"]=target_mode
    #st.write(data)
    data=data[data["Brand"]==brand_filter]
    #st.write("After Brand Filter:", data.shape)

    Annual_Actual_Data=data.groupby(['Brand', 'Tactic']).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index().rename(columns={'Actual Spend (Selected Months)': 'Actual Spend - Full Year'})
    st.session_state['Annual_data']= Annual_Actual_Data
    #st.write('Annual df', Annual_Actual_Data.shape)
    

    st.session_state.filter_data = data[data["Monthly"].isin(st.session_state["Month_filter"])]
    data = data[data["Monthly"].isin(st.session_state["Month_filter"])]
    #st.write('Month Filtered', data.shape)

    if st.session_state["Tactic_Type_filter"]=='Combo':
        grouped_data1= data.groupby(['Brand', 'Actual Tactic', st.session_state["Period_filter"]]).agg(
            {'Actual Spend (Selected Months)': 'sum'}).reset_index().rename(
            columns={st.session_state["Period_filter"]: 'PERIOD','Actual Tactic':'Tactic'}).sort_values(
            ['Brand', 'Tactic'])
        
        grouped_data2= data.groupby(['Brand', 'Group Tactic', st.session_state["Period_filter"]]).agg(
            {'Actual Spend (Selected Months)': 'sum'}).reset_index().rename(
            columns={st.session_state["Period_filter"]: 'PERIOD','Group Tactic':'Tactic'}).sort_values(
            ['Brand', 'Tactic'])

        grouped_data=pd.concat([grouped_data1,grouped_data2],ignore_index=True)
        
    else:
        # Group by brand, tactic, year, quarter, month and sum budget_spend
        grouped_data= data.groupby(['Brand','Level 1','Level 2','Level 3', 'Tactic', st.session_state["Period_filter"]]).agg(
            {'Actual Spend (Selected Months)': 'sum'}).reset_index().rename(
            columns={st.session_state["Period_filter"]: 'PERIOD'}).sort_values(
            ['Brand', 'Tactic'])
        
    st.session_state.ptr=grouped_data['PERIOD'].unique().tolist()
    if st.session_state["opt_type"]=="Target FEC":
        st.session_state['Exi_Total_value']=round(data['Actual FEC'].sum())
        st.session_state['Exi_Extra_value']=0
    
    
    st.session_state.grouped_byTactic=data.groupby(['Brand','Level 1','Level 2','Level 3', 'Tactic']).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index()
    st.session_state.grouped_byTactic_byPeriod=data.groupby(['Brand', 'Tactic',st.session_state["Period_filter"]]).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index()
    grouped_data['MIN_BOUND']=-30
    grouped_data['MAX_BOUND']=30
    grouped_data['COEF_ADJU']=1
    grouped_data['CPM_ADJU']=1
    grouped_data['MIN_BOUND_Spend']=grouped_data['MIN_BOUND']*grouped_data['Actual Spend (Selected Months)']
    grouped_data['MAX_BOUND_Spend']=grouped_data['MAX_BOUND']*grouped_data['Actual Spend (Selected Months)']
    
    st.session_state['grouped_data'] = grouped_data


if "apply_status" not in st.session_state:
    st.session_state.apply_status="FALSE"

if "optim_brand_filter" not in st.session_state:
    #st.write("Brand Filter not in session state", data.shape)
    all_filters(data)

if st.session_state["Tactic_Type_filter"]=='Tactic':
    data = data[data["Actual Tactic"].isin(st.session_state["Tactic_filter"])]
    data['Tactic']=data['Actual Tactic']

elif st.session_state["Tactic_Type_filter"]=='EmPlanner Tactic':
    data = data[data["Group Tactic"].isin(st.session_state["Tactic_filter"])]
    data['Tactic']=data['Group Tactic']

data=data[data["Brand"]==st.session_state["optim_brand_filter"]]

st.session_state['Brand_filter_Data']=data[~data["Monthly"].isin(st.session_state["Month_filter"])] 

data = data[data["Monthly"].isin(st.session_state["Month_filter"])]
    
st.session_state['for_lock_period']=data
            
def update_input():

    #st.write("Update Input Data")

    if st.session_state["Tactic_Type_filter"]=='Combo':
        input_data1=data[data["Monthly"].isin(Month_filter)].groupby(['Brand','Group Tactic', 'Actual Tactic',st.session_state["Period_filter"]]).agg({'Actual Spend (Selected Months)': 'sum',
                                                                                                                                        'MIN_BOUND':'mean', 'MAX_BOUND':'mean',"COEF_ADJU":"mean","CPM_ADJU":"mean"}).reset_index().rename(columns={'Actual Tactic':'Tactic'})
    
        input_data2=data[data["Monthly"].isin(Month_filter)].groupby(['Brand', 'Group Tactic',st.session_state["Period_filter"]]).agg({'Actual Spend (Selected Months)': 'sum', 
                                                              'MIN_BOUND':'mean', 'MAX_BOUND':'mean','Actual Tactic':'nunique',"COEF_ADJU":"mean","CPM_ADJU":"mean"}).reset_index().rename(columns={'Group Tactic':'Tactic'})
        input_data2=input_data2[input_data2['Actual Tactic']>1]
        input_data2.drop(['Actual Tactic'],axis=1,inplace=True)
        input_data2['Group Tactic']=input_data2['Tactic']
        
        input_data=pd.concat([input_data1,input_data2],ignore_index=True)
        input_data=input_data.sort_values(by=['Group Tactic','Actual Spend (Selected Months)'], ascending=[True,False])
        # input_data[["COEF_ADJU","CPM_ADJU"]]=1
    else :
        input_data=data[data["Monthly"].isin(Month_filter)].groupby(['Brand','Level 1','Level 2','Level 3', 'Tactic',st.session_state["Period_filter"]]).agg({'Actual Spend (Selected Months)': 'sum', 
                                                                          'MIN_BOUND':'mean', 'MAX_BOUND':'mean',"COEF_ADJU":"mean","CPM_ADJU":"mean"}).reset_index()



#         # import pandas as pd

        # --- Base grouped data ---
        base = (
            data[data["Monthly"].isin(Month_filter)]
            .groupby(['Brand', 'Level 1', 'Level 2', 'Level 3', 'Tactic', st.session_state["Period_filter"]])
            .agg({
                'Actual Spend (Selected Months)': 'sum',
                'MIN_BOUND': 'mean',
                'MAX_BOUND': 'mean',
                'COEF_ADJU': 'mean',
                'CPM_ADJU': 'mean'
            })
            .reset_index()
        )

        # --- Prepare subtotal dataframes ---
        lvl3_total = (
            base.groupby(['Brand', 'Level 1', 'Level 2', 'Level 3', st.session_state["Period_filter"]], as_index=False)
            .agg({
                'Actual Spend (Selected Months)': 'sum',
                'MIN_BOUND': 'mean',
                'MAX_BOUND': 'mean',
                'COEF_ADJU': 'mean',
                'CPM_ADJU': 'mean'
            })
        )
        lvl3_total["Level 3"] = lvl3_total["Level 3"].astype(str) + " total"
        lvl3_total["Tactic"] = ""

        lvl2_total = (
            base.groupby(['Brand', 'Level 1', 'Level 2', st.session_state["Period_filter"]], as_index=False)
            .agg({
                'Actual Spend (Selected Months)': 'sum',
                'MIN_BOUND': 'mean',
                'MAX_BOUND': 'mean',
                'COEF_ADJU': 'mean',
                'CPM_ADJU': 'mean'
            })
        )
        lvl2_total["Level 2"] = lvl2_total["Level 2"].astype(str) + " total"
        lvl2_total["Level 3"] = ""
        lvl2_total["Tactic"] = ""

        lvl1_total = (
            base.groupby(['Brand', 'Level 1', st.session_state["Period_filter"]], as_index=False)
            .agg({
                'Actual Spend (Selected Months)': 'sum',
                'MIN_BOUND': 'mean',
                'MAX_BOUND': 'mean',
                'COEF_ADJU': 'mean',
                'CPM_ADJU': 'mean'
            })
        )
        lvl1_total["Level 1"] = lvl1_total["Level 1"].astype(str) + " total"
        lvl1_total["Level 2"] = ""
        lvl1_total["Level 3"] = ""
        lvl1_total["Tactic"] = ""

#         # --- Build final_df in hierarchy order ---
#         final_rows = []

#         for brand in sorted(base["Brand"].unique()):
#             brand_df = base[base["Brand"] == brand]
#             lvl1_groups = brand_df["Level 1"].unique()
            
#             for lvl1 in lvl1_groups:
#                 # Level 1 total first
#                 final_rows.append(lvl1_total[(lvl1_total["Brand"] == brand) & (lvl1_total["Level 1"].str.startswith(lvl1))])
                
#                 lvl1_df = brand_df[brand_df["Level 1"] == lvl1]
#                 lvl2_groups = lvl1_df["Level 2"].unique()
                
#                 for lvl2 in lvl2_groups:
#                     # Level 2 total
#                     final_rows.append(lvl2_total[
#                         (lvl2_total["Brand"] == brand) &
#                         (lvl2_total["Level 1"] == lvl1) &
#                         (lvl2_total["Level 2"].str.startswith(lvl2))
#                     ])
                    
#                     lvl2_df = lvl1_df[lvl1_df["Level 2"] == lvl2]
#                     lvl3_groups = lvl2_df["Level 3"].unique()
                    
#                     for lvl3 in lvl3_groups:
#                         # Level 3 total
#                         final_rows.append(lvl3_total[
#                             (lvl3_total["Brand"] == brand) &
#                             (lvl3_total["Level 1"] == lvl1) &
#                             (lvl3_total["Level 2"] == lvl2) &
#                             (lvl3_total["Level 3"].str.startswith(lvl3))
#                         ])
                        
#                         # Level 3 details
#                         final_rows.append(lvl2_df[lvl2_df["Level 3"] == lvl3])

#         # Combine all pieces
#         final_df = pd.concat(final_rows, ignore_index=True)[
#     ['Brand', 'Level 1', 'Level 2', 'Level 3', 'Tactic', st.session_state["Period_filter"],
#      'Actual Spend (Selected Months)', 'MIN_BOUND', 'MAX_BOUND', 'COEF_ADJU', 'CPM_ADJU']
# ]

        final_rows = []

        for brand in sorted(base["Brand"].unique()):
            brand_df = base[base["Brand"] == brand]
            lvl1_groups = brand_df["Level 1"].unique()
            
            for lvl1 in lvl1_groups:
                lvl1_df = brand_df[brand_df["Level 1"] == lvl1]
                lvl2_groups = lvl1_df["Level 2"].unique()
                
                # Add Level 1 total only if more than one Level 2 exists
                if len(lvl2_groups) > 1:
                    final_rows.append(
                        lvl1_total[
                            (lvl1_total["Brand"] == brand) &
                            (lvl1_total["Level 1"].str.startswith(lvl1))
                        ]
                    )
                
                for lvl2 in lvl2_groups:
                    lvl2_df = lvl1_df[lvl1_df["Level 2"] == lvl2]
                    lvl3_groups = lvl2_df["Level 3"].unique()
                    
                    # Add Level 2 total only if more than one Level 3 exists
                    if len(lvl3_groups) > 1:
                        final_rows.append(
                            lvl2_total[
                                (lvl2_total["Brand"] == brand) &
                                (lvl2_total["Level 1"] == lvl1) &
                                (lvl2_total["Level 2"].str.startswith(lvl2))
                            ]
                        )
                    
                    for lvl3 in lvl3_groups:
                        lvl3_df = lvl2_df[lvl2_df["Level 3"] == lvl3]
                        
                        # Add Level 3 total only if more than one Tactic exists
                        if len(lvl3_df["Tactic"].unique()) > 1:
                            final_rows.append(
                                lvl3_total[
                                    (lvl3_total["Brand"] == brand) &
                                    (lvl3_total["Level 1"] == lvl1) &
                                    (lvl3_total["Level 2"] == lvl2) &
                                    (lvl3_total["Level 3"].str.startswith(lvl3))
                                ]
                            )
                        
                        # Add the detail rows under Level 3
                        final_rows.append(lvl3_df)

        # Combine everything
        final_df = pd.concat(final_rows, ignore_index=True)[
            ['Brand', 'Level 1', 'Level 2', 'Level 3', 'Tactic', st.session_state["Period_filter"],
            'Actual Spend (Selected Months)', 'MIN_BOUND', 'MAX_BOUND', 'COEF_ADJU', 'CPM_ADJU']
        ]

        input_data=final_df
        # import pandas as pd
        # import streamlit as st

        # period_col = st.session_state["Period_filter"] 

        # subtotal_levels = ['Level 2', 'Level 3']

        # # --- 1️⃣ Base aggregation ---
        # input_data = (
        #     data[data["Monthly"].isin(Month_filter)]
        #     .groupby(['Brand', 'Level 1', 'Level 2', 'Level 3', 'Tactic', period_col])
        #     .agg({
        #         'Actual Spend (Selected Months)': 'sum',
        #         'MIN_BOUND': 'mean',
        #         'MAX_BOUND': 'mean',
        #         'COEF_ADJU': 'mean',
        #         'CPM_ADJU': 'mean'
        #     })
        #     .reset_index()
        # )

        # # --- 2️⃣ Generic subtotal function ---
        # def add_subtotals(df, group_cols, label_col, parent_label, period_col):
        #     subtotals = (
        #         df.groupby(group_cols  + [period_col], dropna=False)
        #         .agg({
        #             'Actual Spend (Selected Months)': 'sum',
        #             'MIN_BOUND': 'mean',
        #             'MAX_BOUND': 'mean',
        #             'COEF_ADJU': 'mean',
        #             'CPM_ADJU': 'mean'
        #         })
        #         .reset_index()
        #     )
        #     # st.write(label_col,parent_label)
        #     subtotals[label_col] = ""
        #     subtotals[label_col] = subtotals[parent_label].astype(str) + " Total"
        #     deeper_cols = [c for c in ['Level 1', 'Level 2', 'Level 3', 'Tactic'] if c not in group_cols+[label_col]]
        #     # deeper_cols
        #     for c in deeper_cols:
        #         subtotals[c] = ''
        #     subtotals = subtotals[df.columns]
        #     # subtotals
        #     return subtotals

        # # --- 3️⃣ Dynamic subtotal builder ---
        # level_hierarchy = ['Brand', 'Level 1', 'Level 2', 'Level 3', 'Tactic']
        # subtotal_dfs = [input_data]

        # for level in subtotal_levels:
        #     idx = level_hierarchy.index(level)
        #     parent_label = level_hierarchy[idx - 1] if idx > 0 else 'Brand'
        #     group_cols = level_hierarchy[:idx] if idx > 0 else []
        #     group_cols = [c for c in group_cols if c in input_data.columns]
        #     # group_cols
        #     subtotal_df = add_subtotals(input_data, group_cols, level, parent_label, period_col)
        #     # subtotal_df
        #     subtotal_dfs.append(subtotal_df)

        # # --- 4️⃣ Combine all subtotals ---
        # final_df = pd.concat(subtotal_dfs, ignore_index=True)

        # # --- 5️⃣ Sort totals above details ---
        # def total_sorter(val):
          
        #     return ' '+val  if isinstance(val, str) and 'Total' in val else val



        # input_data = final_df.sort_values(
        #     ['Brand', 'Level 1', 'Level 2', 'Level 3', period_col],
        #     key=lambda col: col.map(total_sorter)
        # ).reset_index(drop=True)
        # mask = input_data['Level 2'].str.contains('Total', case=False, na=False)
        # input_data.loc[mask, 'Level 1'] = input_data.loc[mask, 'Level 1'] + ' Total'
        # input_data.loc[mask, 'Level 2'] = ''
        # mask = input_data['Level 3'].str.contains('Total', case=False, na=False)
        # input_data.loc[mask, 'Level 2'] = input_data.loc[mask, 'Level 2'] + ' Total'
        # input_data.loc[mask, 'Level 3'] = ''

        # mask = input_data['Level 1'].str.contains('Total', case=False, na=False)
        # input_data.loc[mask, 'Level 1'] = input_data.loc[mask, 'Level 1'] + ' Total'
        # input_data.loc[mask, 'Level 2'] = ''




    if st.session_state["Period_filter"]=="Monthly":
        input_data['Monthly'] = pd.Categorical(input_data['Monthly'], categories=["Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"], ordered=True)
    
        # Sort DataFrame
        input_data = input_data.sort_values(['Brand', 'Tactic','Monthly'])
    
    input_data["MIN_BOUND_Spend"]=((input_data["MIN_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"]
    input_data["MAX_BOUND_Spend"]=((input_data["MAX_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"]
    input_data["MIN_BOUND"]=np.where(round(input_data["Actual Spend (Selected Months)"]) == 0, 0, input_data["MIN_BOUND"])
    input_data["MAX_BOUND"]=np.where(round(input_data["Actual Spend (Selected Months)"]) == 0, 0, input_data["MAX_BOUND"])
    # input_data[["COEF_ADJU","CPM_ADJU"]]=1
    
    # st.write("Input Data:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # st.dataframe(input_data)
    
    st.session_state.input_data=input_data

if  "input_data" not in st.session_state:
        update_input()



if Apply_button:
    if new_or_exi=='Existing Scenario':
        # already passed var
        pass
    else:
        st.session_state.update_status="TRUE"
        st.session_state.apply_status="TRUE"
        all_filters(data)
        update_input()
        #st.dataframe(st.session_state['grouped_data'], use_container_width=True, hide_index=True)
    
    
if 'img' not in st.session_state:
        st.session_state.img=brand_logo()
   
    
if "grouped_data" not in st.session_state:
    
    # type_update(Type)
    all_filters(data)
    st.session_state.update_status="FALSE"
    st.session_state.filter_data = data[data["Monthly"].isin(Month_filter)]
    # Tactic_filter=tactic_list
    
    # add_Simulator_budget(st.session_state['grouped_data'])


col1, col2,col3 = st.columns([0.04,0.91,0.04], gap="small")
with col1:
    st.image(
            "https://www.c5i.ai/wp-content/themes/course5iTheme/new-assets/images/c5i-primary-logo.svg",
            width=60, # Manually Adjust the width of the image as per requirement
        )
with col2:
    st.header('Sonic Optimization Tool',divider=True)

with col3:
    st.image(st.session_state.img,width=100,)

refresh_radio = st.radio(r"$\textrm{\large Choose the refresh mode}$",options = ["Manual Refresh","Auto Refresh"],horizontal=True)

# if refresh_radio=="Manual Refresh":
#     update_data=st.button("Update Data",type="primary")



# s1,s2,s3,s4 =st.tabs(['Step 1','Step 2','Step 3','Step 4'])

# with s1:

with st.container(border=True):
    
    max_on=st.toggle(r"$\textrm{\large Click to On Max Iteration}$")
    if max_on:
        MAX_ITR=True
    else:
        MAX_ITR=False
    o1,o2,o3=st.columns([0.3,0.7,0.01])
    

        
        
    with o1:
        comments = st.text_input(r"$\textrm{\large Scenario *}$", value=st.session_state["comments_filter"], placeholder="Please enter the Scenario", max_chars=30,)
        
        if comments in comments_list:
            flag = True
            st.error("This scenario already exists. Please enter a unique name.")
        else:
            flag = False

    with o2:
        desc=st.text_input(r"$\textrm{\large Desc}$",placeholder="Please enter the Desc",max_chars=100)

# c1,c2,c3 = st.columns([0.3,0.6,0.1])
# with c1:
#     st.button("Back",key="s1b")
# with c3:
#     st.button("Next",key="s1n")

lock_period_input=st.session_state['for_lock_period']
# Period_filter2 =st.selectbox(   " Choose the Period Type for locking Spend:", ['Yearly','Quarterly','Monthly'])
# st.session_state["Period_filter2"]=Period_filter2

def upate_edit_lock_period():
    edit_lock_period_input=lock_period_input.groupby(['Brand', st.session_state["Period_filter2"]]).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index()
    edit_lock_period_input["Increase By"]=-0.0
    edit_lock_period_input["Optim_Allocate_Spend"]=edit_lock_period_input['Actual Spend (Selected Months)']+(edit_lock_period_input['Actual Spend (Selected Months)']*edit_lock_period_input["Increase By"])
    edit_lock_period_input["MIN_BOUND"]=-1
    edit_lock_period_input["MAX_BOUND"]=0
    edit_lock_period_input["MIN_BOUND_VALUE"]=(edit_lock_period_input["MIN_BOUND"]/100+1)*edit_lock_period_input['Actual Spend (Selected Months)']
    edit_lock_period_input["MAX_BOUND_VALUE"]=(edit_lock_period_input["MAX_BOUND"]/100+1)*edit_lock_period_input['Actual Spend (Selected Months)']
    edit_lock_period_input["BUDGET_LOCK"]=True
    st.session_state['edit_lock_period_input']=edit_lock_period_input[["Brand", st.session_state["Period_filter2"],'Actual Spend (Selected Months)',"Increase By","Optim_Allocate_Spend","BUDGET_LOCK"]]
    st.session_state['edit_lock_period_input2']=edit_lock_period_input[["BUDGET_LOCK","Brand", st.session_state["Period_filter2"],'Actual Spend (Selected Months)',"MIN_BOUND","MAX_BOUND","MIN_BOUND_VALUE","MAX_BOUND_VALUE"]]


def unpivot_edit_lock_period_input2(new_df: pd.DataFrame):
    if new_df is not None:
            if new_df.equals(st.session_state.edit_lock_period_input2):
                if st.session_state.apply_status=="TRUE":
                    edit_lock_period_input=st.session_state['edit_lock_period_input2']
                    edit_lock_period_input["MIN_BOUND_VALUE"]=((edit_lock_period_input["MIN_BOUND"]/100)+1)*edit_lock_period_input["Actual Spend (Selected Months)"]
                    edit_lock_period_input["MAX_BOUND_VALUE"]=((edit_lock_period_input["MAX_BOUND"]/100)+1)*edit_lock_period_input["Actual Spend (Selected Months)"]
    


                
                return
    
            st.session_state['edit_lock_period_input2'] = new_df

    
    if update_mode=="By Spend Value":
        
        edit_lock_period_input=st.session_state['edit_lock_period_input2']
        edit_lock_period_input["MIN_BOUND"]=((edit_lock_period_input["MIN_BOUND_VALUE"]/edit_lock_period_input["Actual Spend (Selected Months)"])-1)*100
        edit_lock_period_input["MAX_BOUND"]=((edit_lock_period_input["MAX_BOUND_VALUE"]/edit_lock_period_input["Actual Spend (Selected Months)"])-1)*100
       
    
    else:
        
        edit_lock_period_input=st.session_state['edit_lock_period_input2']
        edit_lock_period_input["MIN_BOUND_VALUE"]=((edit_lock_period_input["MIN_BOUND"]/100)+1)*edit_lock_period_input["Actual Spend (Selected Months)"]
        edit_lock_period_input["MAX_BOUND_VALUE"]=((edit_lock_period_input["MAX_BOUND"]/100)+1)*edit_lock_period_input["Actual Spend (Selected Months)"]
    
    st.rerun()

# with s3:
   
with st.container(border=True):
    if st.session_state["Tactic_Type_filter"]=="Combo":
        spend_type_list=["Only Total Spend/Revenue"]
    elif st.session_state["opt_type"]=="Target Revenue":
            spend_type_list=["Only Total Spend/Revenue"]   
    else:
        spend_type_list=["Only Total Spend/Revenue",
                         # "Total Spend + Lock Budget by Period"
                        ]
        
    
    def apply_custom_css():
        custom_css = """
        <style>
        div[role="radiogroup"] label {
            font-size: 200px; /* Adjust the font size as needed */
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
    
    apply_custom_css()

    spend_type = st.radio(r'$\textrm{\large Choose Spend Type}$', options=spend_type_list, horizontal=True)
    # st.write("Spend type")
    # st.dataframe(data)
    
    if st.session_state["opt_type"]=="Budget Adjustment":


        
       if spend_type=='Total Spend + Lock Budget by Period':
    
    
            c1,c2=st.columns([0.3,0.7])
            with c1:
                Period_filter2 =st.selectbox(   " Choose the Period Type for locking Spend:", ['Yearly','Quarterly','Monthly'])
                
                
                st.session_state["Period_filter2"]=Period_filter2
                if "edit_lock_period_input2" not in st.session_state:
                    upate_edit_lock_period()
                # Apply2=st.button("Apply",key="apply2")
                if Apply_button:
                    upate_edit_lock_period()
                # st.write(st.session_state['edit_lock_period_input2'].columns[2])
                if st.session_state["Period_filter2"]!=st.session_state['edit_lock_period_input2'].columns[2]:
                    upate_edit_lock_period()
                
                
                
            # upate_edit_lock_period()
            # method=st.radio("Chosoe the method to lock the period",["By Fixed Spend","By Range Spend"],horizontal=True)
            update_mode=st.radio("Input Mode to update Locking Spend",["By Percentage","By Spend Value"],horizontal=True)
            
    
            # elif method=="By Range Spend":
            if update_mode=="By Percentage":
                
                edis=["Brand", st.session_state["Period_filter2"],'Actual Spend (Selected Months)',"Optim_Allocate_Spend","MIN_BOUND_VALUE","MAX_BOUND_VALUE"]
                # col=[["Brand", st.session_state["Period_filter2"],'Actual Spend (Selected Months)',"Increase By","Optim_Allocate_Spend"]]
            else:
                edis=["Brand", st.session_state["Period_filter2"],'Actual Spend (Selected Months)',"Increase By","MIN_BOUND","MAX_BOUND"]
                
                
            ec1={ col: st.column_config.NumberColumn(step="1",) for col in ['Actual Spend (Selected Months)',"Optim_Allocate_Spend","MIN_BOUND_VALUE","MAX_BOUND_VALUE"]} 
            ec2={col:st.column_config.NumberColumn(format="%0.2f %% ") for col in ['MIN_BOUND','MAX_BOUND']}
            edi_df2=st.data_editor(st.session_state['edit_lock_period_input2'],column_config={**ec1, **ec2}
                                             ,
                                          disabled=edis,
                hide_index=True,)
            
            # if refresh_radio=="Manual Refresh":
            #     if update_data:
            #         unpivot_edit_lock_period_input2(edi_df2)
            # else:
            #     unpivot_edit_lock_period_input2(edi_df2)
            
            e1,e2=st.columns([0.3,0.70])
            with e1:
                With_Extra_Spend=st.number_input("Total Spend",value=round(data['Actual Spend (Selected Months)'].sum()),key="ex",step=1000000)
                st.session_state["Variable_Spend"]=With_Extra_Spend
            min_to_print=sum(st.session_state['edit_lock_period_input2']["MIN_BOUND_VALUE"])
            max_to_print=sum(st.session_state['edit_lock_period_input2']["MAX_BOUND_VALUE"])
            st.write(f"Total Optim Min Budget Spend {round(min_to_print)} and Max Budget Spend {round(max_to_print)} for selected period ")
            edit_lock_input=st.session_state['edit_lock_period_input2']
            edit_lock_input=edit_lock_input[edit_lock_input["BUDGET_LOCK"]==True]
            edit_lock_input=edit_lock_input[["Brand",st.session_state["Period_filter2"],"MIN_BOUND_VALUE","MAX_BOUND_VALUE"]]
            
            edit_lock_input=edit_lock_input.rename(columns={"MAX_BOUND_VALUE":"LOCK_MAX_BOUND","MIN_BOUND_VALUE":"LOCK_MIN_BOUND"})
            edit_lock_input["LOCK_PERIOD"]=st.session_state["Period_filter2"]
            
            data=data.merge(edit_lock_input,on=["Brand",st.session_state["Period_filter2"]],how="left")
                   
       else:
           e1,e2=st.columns([0.5,0.70])
           with e1:
                chg=st.toggle(r"$\textrm{\large By Change}$")
                if new_or_exi=="New Scenario":
                    if chg==False:
                        With_Extra_Spend2=st.number_input(r"$\textrm{\large Total Spend}$",value=round(data['Actual Spend (Selected Months)'].sum()),key="ex2",step=1000000)
                    else:
                        With_Extra_Spend2=round(data['Actual Spend (Selected Months)'].sum())+st.number_input("Additional Spend",value=0,key="ex2",step=1000000)
                    st.write(f"Total Optim Budget Spend: {round(With_Extra_Spend2 / 1_000_000, 2)}M and Extra Budget Spend: {round((With_Extra_Spend2 - round(data['Actual Spend (Selected Months)'].sum())) / 1_000_000, 2)}M for the selected period")

                else:
                    
                    if chg==False:
                        With_Extra_Spend2=st.number_input("Total Spend",value=st.session_state['Exi_Total_value'],key="ex2",step=1000000)
                        st.write(f"Total Optim Budget Spend: {round(With_Extra_Spend2 / 1_000_000, 2)}M and Extra Budget Spend: {round(((With_Extra_Spend2 - st.session_state['Exi_Total_value'])+st.session_state['Exi_Extra_value']) / 1_000_000, 2)}M for the selected period")

                    else:
                        With_Extra_Spend2=st.session_state['Exi_Total_value']+st.number_input("Additional Spend",value=st.session_state['Exi_Extra_value'],key="ex2",step=1000000)
                        st.write(f"Total Optim Budget Spend: {round(With_Extra_Spend2 / 1_000_000, 2)}M and Extra Budget Spend: {round((With_Extra_Spend2 - st.session_state['Exi_Total_value']) / 1_000_000, 2)}M for the selected period")

                    
                    
           st.session_state["Variable_Spend"]=With_Extra_Spend2
           data=data.assign(LOCK_PERIOD="Yearly",LOCK_MAX_BOUND=None,LOCK_MIN_BOUND=None)
           
           
           
       data["Target_LTE_Profit"]=0 
    elif st.session_state["opt_type"]=="Target FEC" and st.session_state["target_mode"]=="FEC":
        e1,e2=st.columns([0.5,0.70])
        
        with e1:
            chg=st.toggle("By Change (Media BB)")
            if new_or_exi=="New Scenario": 
                if chg==False:
                    Target_Profit=st.number_input("Total Target Media FEC (₩)",value=st.session_state['Exi_Total_value'],step=1000000)
                    # st.write(f"Total Optim Budget Spend: {round(Target_Profit / 1_000_000, 2)}M and Extra Budget Spend: {round(((Target_Profit - st.session_state['Exi_Total_value'])+st.session_state['Exi_Extra_value']) / 1_000_000, 2)}M for the selected period")

                else:
                    Target_Profit=st.session_state['Exi_Total_value']+st.number_input("Target Media BB (₩)",value=st.session_state['Exi_Extra_value'],step=1000000)
                    # st.write(f"Total Optim Budget Spend: {round(Target_Profit / 1_000_000, 2)}M and Extra Budget Spend: {round((Target_Profit - st.session_state['Exi_Total_value']) / 1_000_000, 2)}M for the selected period")
                    
            else:
                if chg==False:
                    Target_Profit=st.number_input("Total Target Media FEC (₩)",value=round(data['Actual FEC'].sum()),step=1000000)
                else:
                    Target_Profit=round(data['Actual FEC'].sum())+st.number_input("Target Media BB (₩)",value=0,step=1000000)
    
        
        st.session_state["Target_Profit"]=Target_Profit
        data=data.assign(LOCK_PERIOD="Yearly",LOCK_MAX_BOUND=None,LOCK_MIN_BOUND=None,Target_LTE_Profit=0)
        Extra_Spend=0
        st.write(f"Total Optim Budget FEC: {round(Target_Profit / 1_000_000, 2)}M and Extra Budget FEC: {round((Target_Profit -round(data['Actual FEC'].sum())) / 1_000_000, 2)}M for the selected period")


    elif st.session_state["opt_type"]=="Target FEC" and st.session_state["target_mode"]=="LT FEC":
        Target_LTE_Profit=st.number_input("Target Optimization LTE FEC (₩)",value=0)
        data["Target_LTE_Profit"]=Target_LTE_Profit
        data=data.assign(LOCK_PERIOD="Yearly",LOCK_MAX_BOUND=None,LOCK_MIN_BOUND=None,Target_Profit=0)
        Extra_Spend=0
    elif st.session_state["opt_type"]=="Target FEC" and st.session_state["target_mode"]=="MULTI KPIs":
        c1,c2 = st.columns(2)
        with c1:
            Target_Profit=st.number_input("Target Optimization FEC (₩)",value=0)
        with c2:
            Target_LTE_Profit=st.number_input("Target Optimization LTE FEC (₩)",value=0)
        st.session_state["Target_Profit"]=Target_Profit
        data["Target_LTE_Profit"]=Target_LTE_Profit
        data=data.assign(LOCK_PERIOD="Yearly",LOCK_MAX_BOUND=None,LOCK_MIN_BOUND=None)
        Extra_Spend=0

def unpivot_input_data(new_df: pd.DataFrame):
    if new_df is not None:
            if new_df.equals(st.session_state.input_data):
                # if st.session_state.apply_status=="TRUE":
            
                input_data=st.session_state['input_data']
                input_data["MIN_BOUND_Spend"]=(((input_data["MIN_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"])+1
                input_data["MAX_BOUND_Spend"]=(((input_data["MAX_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"])+1
                st.session_state['input_data']=input_data


                
                return
    
            st.session_state['input_data'] = new_df

    if spend_switch=="By Spend Value":
        input_data=st.session_state['input_data']
        input_data["MIN_BOUND"]=((input_data["MIN_BOUND_Spend"]/input_data["Actual Spend (Selected Months)"])-1)*100
        input_data["MAX_BOUND"]=((input_data["MAX_BOUND_Spend"]/input_data["Actual Spend (Selected Months)"])-1)*100
        st.session_state['input_data']=input_data
    else:
    
        input_data=st.session_state['input_data']
        input_data["MIN_BOUND_Spend"]=((input_data["MIN_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"]
        input_data["MAX_BOUND_Spend"]=((input_data["MAX_BOUND"]/100)+1)*input_data["Actual Spend (Selected Months)"]
        st.session_state['input_data']=input_data

    
 
    st.rerun()   


        
    



# with s2:
with st.container(border=True, height=650):
    col1, col2,col3,col4 = st.columns([0.4,.5,0.15,0.1])
    with col1:
        spend_switch=st.radio(r"$\textrm{\large Input Mode to update Spend}$",["By Percentage","By Spend Value"],horizontal=True)
    
    with col3:
        # download_template=st.button("Download Templete", use_container_width=True)
        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")
        gd=st.session_state['grouped_data']
        if st.session_state["Period_filter"]=="Monthly":
        
        
            gd['PERIOD'] = pd.Categorical(gd['PERIOD'], categories=["Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"], ordered=True)
        
            # Sort DataFrame
            gd = gd.sort_values(['Brand', 'Tactic','PERIOD'])
        csv = convert_df(gd)
        
        st.download_button(
            label="Download Templete",
            data=csv,
            file_name=str(st.session_state["optim_brand_filter"])+"_"+st.session_state["Period_filter"]+"_"+st.session_state["Tactic_Type_filter"]+"_"+str(datetime.now().strftime("%d_%m_%y %H_%M"))+".csv",
            mime="text/csv",
            type="secondary"
            ,use_container_width=True
        )
    with col4:
        reset_button=st.button("Reset", type="primary",use_container_width=True)
    
    if reset_button:
        all_filters(data)
        st.session_state.update_status="TRUE"
    if spend_switch=="By Spend Value":
        dis=["Brand",'Level 1','Level 2','Level 3', "Tactic",'Actual Spend - Full Year',st.session_state["Period_filter"],'Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)','MIN_BOUND',"MAX_BOUND"]
     
    else:
        dis=["Brand",'Level 1','Level 2','Level 3', "Tactic",'Actual Spend - Full Year',st.session_state["Period_filter"],'Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)',"MIN_BOUND_Spend","MAX_BOUND_Spend"]
    c1={ col: st.column_config.NumberColumn(step="1",) for col in ['Actual Spend (Selected Months)','MIN_BOUND_Spend','MAX_BOUND_Spend']} 
    c2={col:st.column_config.NumberColumn(format="%0.2f %% ") for col in ['MIN_BOUND','MAX_BOUND']}
    color_code = '#f0f0f0' 
    header_style = {
    'selector': 'th',  # Target the header (th element)
    'props': [('background-color', '{color_code}'), ('color', 'white'), ('font-weight', 'bold')]
    }



    def highlight_all_columns(col):
        # Base neutral background for all cells
        return ['background-color: #FAFAFA'] * len(col)

    def highlight_total_rows(df):
        n_rows, n_cols = df.shape
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        # Track active hierarchy levels by column index
        active_levels = []

        for i in range(n_rows):
            row_str = df.iloc[i].astype(str).str.lower()
            total_cols = [j for j, val in enumerate(row_str) if 'total' in val]
            is_total_row = bool(total_cols)

            # If we encounter a new total, update hierarchy stack
            if is_total_row:
                level_col = total_cols[0]
                # Remove any deeper (same or greater) levels
                active_levels = [lvl for lvl in active_levels if lvl < level_col]
                active_levels.append(level_col)

            # Apply hierarchical shading for all active levels
            for lvl_idx, lvl_start in enumerate(active_levels):
                base_gray = 250
                step = 15  # how much darker per level
                shade = max(170, base_gray - step * (lvl_idx + 1))
                color = f'rgb({shade},{shade},{shade})'

                # Apply shade from that level's column → right
                for j in range(lvl_start, n_cols):
                    styles.iloc[i, j] = f'background-color: {color};'

            # Apply bold style ONLY from "total" column → right
            if is_total_row:
                total_col_idx = total_cols[0]
                for j in range(total_col_idx, n_cols):
                    existing = styles.iloc[i, j]
                    styles.iloc[i, j] = existing + ' font-weight: bold;' if existing else 'font-weight: bold;'

        return styles


    # Apply to Streamlit DataFrame
    styled_df = (
        st.session_state['input_data']
        .style
        .apply(highlight_all_columns, axis=0)
        .apply(highlight_total_rows, axis=None)
    )

    # Define formatting for currency and percentage columns
    currency_format = {
        col: '₩ {:,.0f}' for col in [
            'Actual Spend (Selected Months)',
            'MIN_BOUND_Spend',
            'MAX_BOUND_Spend'
        ]
    }
    percentage_format = {
        col: "{:.2f}%" for col in ['MIN_BOUND', 'MAX_BOUND']
    }

    styled_df = styled_df.format({**percentage_format, **currency_format})

        

    edited_input_Data=st.data_editor(styled_df ,height=550
                                 ,column_config={**c1, **c2}
                                 ,use_container_width=True,
                              disabled=dis,
    hide_index=True,
                                 # on_change=unpivot_data(st.session_state.grouped_data)
                                                  )
 


if np.any((edited_input_Data['MIN_BOUND'] <= -200) | (edited_input_Data['MAX_BOUND'] >= 200)):
    st.warning('⚠️ Warning: Some values in MIN_BOUND or MAX_BOUND are out of the acceptable range (-200 to 200).')
# if refresh_radio=="Manual Refresh":
#     if update_data:
#         unpivot_input_data(edited_input_Data)
# else:
#     unpivot_input_data(edited_input_Data)

# unpivot_input_data(edited_input_Data)

# c1,c2,c3 = st.columns([0.3,0.6,0.1])
# with c1:
#     st.button("Back",key="s2b")
# with c3:
#     st.button("Next",key="s2n")




# def update_min_max(data):
column_list = ["Brand",'Level 1','Level 2','Level 3', "Actual Tactic", "BUDGET_WEEK_START_DT", "C1", "C2", "C3", "C4", "CPM", "Actual Spend (Selected Months)", "ADJUSTMENT_FACTOR", "REF_ADJ_FCTR", 
               "ECOMM_ROI", "TACTIC_ADJ_FCTR", "SEASONAL_ADJ_FCTR", "ADSTOCK_X", "CURVE_TYPE", "ADSTOCK_WEEK", "ADSTOCK", "MONTH_YEAR", "Group Tactic", "FEC_FCTR", 
               "Yearly", "Quarterly", "Monthly", "Tactic",'Actual FEC', 'Actual Profit','LOCK_PERIOD', 'LOCK_MIN_BOUND', 'LOCK_MAX_BOUND','LTE_FCTR', 
               'Actual LTE Profit', 'Actual LTE FEC',"Target_LTE_Profit"]

data=data[column_list]

# st.write("column list")
# st.dataframe(data)

input_cache=st.session_state['input_data'][['Brand','Level 1','Level 2','Level 3','Tactic',st.session_state["Period_filter"],'MIN_BOUND_Spend','MAX_BOUND_Spend',"Actual Spend (Selected Months)","COEF_ADJU","CPM_ADJU"]]
input_cache=input_cache.rename(columns={"Actual Spend (Selected Months)":"Actual_Spend_Agg"})
data=data.merge(input_cache,on=['Brand','Level 1','Level 2','Level 3',st.session_state["Period_filter"],'Tactic'] )

# st.write("aggregated")
# st.dataframe(data)

if st.session_state["Tactic_Type_filter"]=='Combo':
    
    G_data=data[column_list].merge(input_cache,left_on=['Brand',st.session_state["Period_filter"],'Group Tactic'],right_on=['Brand',st.session_state["Period_filter"],'Tactic'],how="left" ).rename(columns={'MIN_BOUND_Spend':'GRP_MIN_BOUND_SPEND','MAX_BOUND_Spend':'GRP_MAX_BOUND_SPEND'})
   
    data[['GRP_MIN_BOUND_SPEND','GRP_MAX_BOUND_SPEND']]= G_data[['GRP_MIN_BOUND_SPEND','GRP_MAX_BOUND_SPEND']]
    data['GRP_MIN_BOUND_SPEND'] = np.where(pd.isnull(data['GRP_MIN_BOUND_SPEND']), 
                                        data['MIN_BOUND_Spend'], 
                                        data['GRP_MIN_BOUND_SPEND'])
    data['GRP_MAX_BOUND_SPEND'] = np.where(pd.isnull(data['GRP_MAX_BOUND_SPEND']), 
                                        data['MAX_BOUND_Spend'], 
                                        data['GRP_MAX_BOUND_SPEND'])
    
else :
    data[['GRP_MIN_BOUND_SPEND','GRP_MAX_BOUND_SPEND']]= 0
 
data['MIN_BOUND']=data["Actual Spend (Selected Months)"]*(data["MIN_BOUND_Spend"]/data["Actual_Spend_Agg"])
data['MAX_BOUND']=data["Actual Spend (Selected Months)"]*(data["MAX_BOUND_Spend"]/data["Actual_Spend_Agg"])


    # ['MIN_BOUND']=data["Actual Spend (Selected Months)"]*(data["MIN_BOUND_Spend"]/data["Actual_Spend_Agg"])
    # data['MAX_BOUND']=data["Actual Spend (Selected Months)"]*(data["MAX_BOUND_Spend"]/data["Actual_Spend_Agg"])
# data.drop(columns=["MIN_BOUND_Spend", "MAX_BOUND_Spend", "Actual_Spend_Agg"], inplace=True)

agg_data=data.groupby(["Brand"]).agg({"Actual Spend (Selected Months)":'sum',"MIN_BOUND":'sum',"MAX_BOUND":'sum'})

# agg_data["Total Optimizing Spend"]=agg_data["Actual Spend (Selected Months)"]+Variable_Spend
#st.session_state["min_value"]=int(round(agg_data["MIN_BOUND"]-agg_data["Actual Spend (Selected Months)"])[0])
st.session_state["min_value"] = int(round((agg_data["MIN_BOUND"] - agg_data["Actual Spend (Selected Months)"]).iloc[0]))
#st.session_state["max_value"]=int(round(agg_data["MAX_BOUND"]-agg_data["Actual Spend (Selected Months)"])[0])  
st.session_state["max_value"] = int(round((agg_data["MAX_BOUND"] - agg_data["Actual Spend (Selected Months)"]).iloc[0]))



# t=data.groupby(['Brand']).agg(
#             {'Actual Spend (Selected Months)': 'sum','Actual Profit':'sum'})
# t




min_value=st.session_state["min_value"]
max_value=st.session_state["max_value"]
if "Target_Profit" not in st.session_state:
    st.session_state["Target_Profit"]=0
    st.session_state["Variable_Spend"]=0



# with s4:

# st.session_state['grouped_data']
# st.session_state['input_data']
if refresh_radio=="Manual Refresh":
    Manual_update=st.button(r"$\textrm{\large Manual Update}$")
    if Manual_update:
        unpivot_input_data(edited_input_Data)
        if st.session_state["opt_type"]=="Budget Adjustment":
            if spend_type=='Total Spend + Lock Budget by Period':
                unpivot_edit_lock_period_input2(edi_df2)
        
else:
    unpivot_input_data(edited_input_Data)
    if st.session_state["opt_type"]=="Budget Adjustment":
            if spend_type=='Total Spend + Lock Budget by Period':
                unpivot_edit_lock_period_input2(edi_df2)
    
# method=st.toggle("Switch to Old Method")
# save_scenario=st.button(r"$\textrm{\large Save Scenario}$")
# if save_scenario:
#     if len(comments)==0:
#         st.warning("⚠ Please enter the Scenario")
#     elif comments in [row["COMMENTS"] for row in session.sql('SELECT DISTINCT "comments" AS COMMENTS FROM ANALYTICS.UMM_OPTIM.PROD_SONIC_OPTIMIZATION_INPUT_TABLE').collect()]:
#         st.warning(f"{comments} Scenario name is already exisit so Please enter different Scenario name")
#     else:
#         with st.spinner("Please wait - Don't close the tab!"):
#             unpivot_input_data(edited_input_Data)
#             # input_values_to_save
#             s_df=st.session_state['input_data']
#             s_df=s_df.rename(columns={st.session_state["Period_filter"]:'PERIOD'})
#             s_df['TARGET_OPTIM_TYPE']=st.session_state["target_mode"]
#             s_df['OPTIM_TYPE']=st.session_state["opt_type"]
#             s_df['INPUT_TYPE']=st.session_state["Tactic_Type_filter"]
#             s_df['PERIOD_TYPE']=st.session_state["Period_filter"]
#             s_df['YEAR_LIST']=st.session_state["Yr_filter"]
#             s_df['MONTH_LIST']=', '.join(map(str, st.session_state["Month_filter"]))
#             s_df['comments']=comments
#             s_df["Total_value"]=np.where(st.session_state["opt_type"]=="Target FEC",st.session_state["Target_Profit"],st.session_state["Variable_Spend"])
#             s_df["Extra_value"]=np.where(st.session_state["opt_type"]=="Target FEC",st.session_state["Target_Profit"],st.session_state["Variable_Spend"])-np.where(st.session_state["opt_type"]=="Target FEC",round(data['Actual FEC'].sum()),round(data['Actual Spend (Selected Months)'].sum()))
#             session.create_dataframe(s_df).write.mode("append").save_as_table("ANALYTICS.UMM_OPTIM.PROD_SONIC_OPTIMIZATION_INPUT_TABLE")
#             st.success(f"{comments} - Saved Successfully")                    
# method=True
# st.write("Turning on the above toggle would run the old method of optimization")
Optimizing_button = st.button(r"$\textrm{\large 🚀 Start Optimization}$",use_container_width=False, disabled = flag)
# if st.session_state["opt_type"]=="Budget Adjustment" and not min_value<=Variable_Spend<=max_value:
#     st.warning(f"⚠ Change in Base Spend should between {min_value} and {max_value}")
# st.session_state['Brand_filter_Data']
# st.session_state['Brand_filter_Data']['COEF_ADJU']
# data

def anl_curv(df,edited_input_Data):
#Pre-defined function
    def log(x,C1,C2):
        return C1*np.log(x+C2)
    def hill(x,C1,C2,C3,C4):
        return C1 + ((C2-C1) * x**C3) / (C4**C3 + x**C3)
    def poly_fit(x,y):
        coefficients = np.polyfit(x, y, deg=2)
        trend_line = np.poly1d(coefficients)
        return trend_line
    def logistic(x,C1,C2,C3):
        return C1 / (1 + np.exp(C2*(x - C3)))
    def power(x,C1,C2):
        return np.exp(C1)*np.power(x,C2)
    def linear(x,C1,C2):
        return C1*x + C2
    def rms(y_pred,y_ori):
        return(np.sqrt(((y_pred - y_ori) ** 2).mean()))
    
    
    #Calculate the best curve based on RMSE. 
    def best_curve(profit,spend,i,j):
      # fig,ax=plt.subplots(figsize=(10, 5))
      # ax.plot(np.sort(spend),np.sort(profit),color='b',label = 'Original Profit Curve')
      #logistic Curve
      max_profit = np.max(profit)
      best_sig_coeff = None
      best_profit_sig = 0
      best_rmse_sig = float('inf')
      x1 = [max_profit,1]
      x2 = [0,1]
      x3 = [0,1]
      # Generate all combinations of x1, x2, x3
      combinations = itertools.product(x1, x2, x3)
      # Initialize variables to track the best combination
      # Iterate through all combinations
      for inital_param in combinations:
        try:
          popt_sig, pcov_sig = curve_fit(logistic, spend, profit, p0=inital_param)
          y_sig = logistic(spend, *popt_sig)
          # Calculate RMSE
          rmse_sig = rms(y_sig, profit)
          # Update best combination if this RMSE is lower
          if rmse_sig < best_rmse_sig:
            best_rmse_sig = rmse_sig
            best_profit_sig = y_sig
            best_sig_coeff = popt_sig
        except RuntimeError:
          # Handle any errors during curve fitting
          continue
      # ax.plot(np.sort(spend),np.sort(best_profit_sig),color='red',label="logistic curve")
      #Power Curve
      best_power_coeff = None
      best_profit_power = 0
      best_rmse_power = float('inf')
      x1 = [0,1]
      x2 = [0,1]
      # Generate all combinations of x1, x2
      combinations = itertools.product(x1, x2)
      # Initialize variables to track the best combination
      # Iterate through all combinations
      for inital_param in combinations:
        try:
          popt_power, pcov_power= curve_fit(power, spend, profit, p0=inital_param)
          y_power = power(spend, *popt_power)
          # Calculate RMSE
          rmse_power = rms(y_power, profit)
          # Update best combination if this RMSE is lower
          if rmse_power < best_rmse_power:
            best_rmse_power = rmse_power
            best_profit_power = y_power
            best_power_coeff = popt_power
        except RuntimeError:
          # Handle any errors during curve fitting
          continue
      # ax.plot(np.sort(spend),np.sort(best_profit_power),color='black',label='Power curve')
      #Hill curve
      best_hill_coeff = None
      best_profit_hill = 0
      best_rmse_hill = float('inf')
      x1 = [max_profit]
      x2 = [np.min(profit)]
      x3 = [0,1]
      x4 = [0,1]
      # Generate all combinations of x1, x2, x3
      combinations = itertools.product(x1, x2, x3, x4)
      # Initialize variables to track the best combination
      # Iterate through all combinations
      for inital_param in combinations:
        try:
          popt_hill, pcov_hill= curve_fit(hill, spend, profit, p0=inital_param)
          y_hill = hill(spend, *popt_hill)
          # Calculate RMSE
          rmse_hill = rms(y_hill, profit)
          # Update best combination if this RMSE is lower
          if rmse_hill< best_rmse_hill:
            best_rmse_hill = rmse_hill
            best_profit_hill = y_hill
            best_hill_coeff = popt_hill
        except RuntimeError:
          continue
      # ax.plot(np.sort(spend),np.sort(best_profit_hill),color='cyan',label="hill curve")
      # ax.legend(loc='upper right',bbox_to_anchor=(1.27, 1))
      # ax.set_xlabel("Spend")
      # ax.set_ylabel("Profit")
      # plt.title(i+" "+j)
      # plt.show()
      if best_rmse_power<best_rmse_hill and best_rmse_power<best_rmse_sig:
        curve_type = 'Power'
        return curve_type,best_power_coeff,best_profit_power
      elif best_rmse_hill<best_rmse_power and best_rmse_hill<best_rmse_sig:
        curve_type = 'Hill'
        return curve_type,best_hill_coeff,best_profit_hill
      elif best_rmse_sig<best_rmse_power and best_rmse_sig<best_rmse_hill:
        curve_type = 'Logistic'
        return curve_type,best_sig_coeff,best_profit_sig
    
    
    #Make a list of alll the curve values to be stored in DF
    def append_rows(brand_1,tactic_1,curve_type_1,coeff,base_budget,base_profit,predicted_profit):
      if curve_type_1 == 'Hill':
        row = [brand_1,tactic_1,curve_type_1,coeff[0],coeff[1],coeff[2],coeff[3],"((C1 + ((C2-C1) * x^C3) / (C4^C3 + x^C3)))",base_budget,base_profit,predicted_profit]
        return row
      elif curve_type_1 == 'Power':
        row = [brand_1,tactic_1,curve_type_1,coeff[0],coeff[1],0,0,"((e^(C1))*(x^C2))",base_budget,base_profit,predicted_profit]
        return row
      elif curve_type_1 == 'Logistic':
        row = [brand_1,tactic_1,curve_type_1,coeff[0],coeff[1],coeff[2],0,"C1/(1+exp(C2*(x-C3)))",base_budget,base_profit,predicted_profit]
        return row
    
    # import warnings
    
    # # Suppress all warnings
    # warnings.filterwarnings('ignore')
    
    #Main function to pass all the reqired values Run this code with the necessary updates to get the coefficients of each brand and tactic based on spend and profit 
    def curve(df):
      total_row = []
      brand_list = df['Brand'].unique()
      for i in brand_list:
        df_brand = df[df['Brand']==i]
        tactic_list = df_brand['Tactic'].unique()
        for j in tactic_list:
          df_tactic = df_brand[df_brand['Tactic']==j]
          df_2 = df_tactic.groupby('PERCENTAGE').agg({'BUDGET_SPEND': 'sum','FINAL_PROFIT': 'sum'}).reset_index()
          spent = df_2['BUDGET_SPEND']
          profit = df_2['FINAL_PROFIT']
          if spent.empty or profit.empty or spent.sum() == 0 or profit.sum() == 0:
            continue
          base_ROI = profit/spent
          curve_type,coeff,predicted_profit = best_curve(profit,spent,i,j)
          row = append_rows(i,j,curve_type,coeff,spent[9],profit[9],predicted_profit[9])
          total_row.append(row)
      return total_row
    
    total_row=curve(df) 
    column_name = ['Brand','Tactic','CURVE_TYPE','C1','C2','C3','C4','FUNCTION','BASE BUDGET','BASE ADJUSTED PROFIT','ANNUAL CALC PROFIT']
    anl_df= pd.DataFrame(total_row, columns=column_name)
    # print(total_row)  
    anl_df=anl_df.merge(edited_input_Data)

    # df
    # anl_df

    def objective_to_minimize(x, df):
        idx = df.index.values
        C1_values = df["C1"].values
        C2_values = df["C2"].values
        C3_values = df["C3"].values
        C4_values = df["C4"].values
        CURVE_TYPE_values = df["CURVE_TYPE"].values
        
        revenue = np.zeros(len(df))
        for i in range(len(df)):
            revenue[i] = calculate_revenue(
                x[idx[i]], 
                C1_values[i], 
                C2_values[i], 
                C3_values[i], 
                C4_values[i], 
                CURVE_TYPE_values[i]
            )
        return -np.sum(revenue)
    
    # Define revenue calculation based on curve type
    def calculate_revenue(x, c1, c2, c3, c4, curve_type, CPM = 1000, adjustment_factor = 1, ref_adj_fctr = 1, 
                      tactic_adj_fctr = 1, seasonal_adj_fctr = 1, ADSTOCK_X = 1, ECOMM_ROI = 1):
        if CPM == 0:
            return 0
        if round(x) <= 1:
            return 0
        if curve_type == "Hill":
            return hill(x, c1, c2, c3, c4, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
        elif curve_type == "Logistic":
            return logistic(x, c1, c2, c3, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
        elif curve_type == "Power":
            return power(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
        else:
            return NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
        
    def hill(x, c1, c2, c3, c4, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
        if x == 0:
            return 0
        #return ((adjustment_factor * ref_adj_fctr * (c1 + ((c2 - c1) * ((x / CPM) + ADSTOCK_X) ** c3) / (c4**c3 + ((x / CPM) + ADSTOCK_X) ** c3))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
        return ((adjustment_factor * ref_adj_fctr * c1 / (1 + expit(c2 * (x - c3)))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr


    # Logistic function
    def logistic(x, c1, c2, c3, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
        if x == 0:
            return 0
        #return ((adjustment_factor * ref_adj_fctr * (c1 / (1 + expit(c2 * (((x / CPM) + ADSTOCK_X) - c3))))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
        return ((adjustment_factor * ref_adj_fctr * (c1 / (1 + expit(c2 * (x - c3))))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr


    # Power function
    def power(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
        if x == 0:
            return 0
        return ((adjustment_factor * ref_adj_fctr * (expit(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr

    # NullCurve function
    def NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
        if x == 0:
            return 0
        return ((adjustment_factor * ref_adj_fctr * (expit(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (
            x * ECOMM_ROI
        )) * tactic_adj_fctr * seasonal_adj_fctr
    
    
    # Define a function to compute the gradient
    def custom_gradient(x, df):
        epsilon = np.sqrt(np.finfo(float).eps) * np.maximum(1, np.abs(x))
        return approx_fprime(x, objective_to_minimize, epsilon, df)
    
    initial_spend_values = anl_df["MIN_BOUND_Spend"].values
    bounds = tuple(
            zip( anl_df["MIN_BOUND_Spend"].values, anl_df["MAX_BOUND_Spend"].values)
        )
    CHANGE_SPEND=st.session_state["Variable_Spend"]
    
    def equality_constraint(x):
        return np.sum(x) - (CHANGE_SPEND 
                           )
    
    # def inequality_constra(x):
    #         # Ensure that the sum of x is within 95% and 100% of budget_spend
    #     return [np.sum(x) - 0.992 * (anl_df["Actual Spend (Selected Months)"].sum()+CHANGE_SPEND ), 1* (anl_df["Actual Spend (Selected Months)"].sum()+CHANGE_SPEND ) - np.sum(x)]
    
    
    constraints = [
                    {"type": "eq", "fun": equality_constraint},
                    ] 
    print("constraint")
    print(constraints)
    # st.write(bounds)
    # data.show()
    # start_time = datetime.now()
    def minimzer_fun(): 
        result = minimize(
                        fun=objective_to_minimize,
                        x0=initial_spend_values,
                        args=(anl_df,),
                        jac=custom_gradient,
                        constraints=constraints,
                        bounds=bounds,
                        options={"maxiter": 100}
                )
        return result
    result=minimzer_fun()
    # if result.message not in ("Optimization terminated successfully","Iteration limit reached"):
    #     def inequality_constra(x):
   
    #         return [np.sum(x) - 0.95 * (anl_df["BASE BUDGET"].sum()+CHANGE_SPEND  ), 1* (anl_df["BASE BUDGET"].sum()+CHANGE_SPEND  ) - np.sum(x)]

    #     constraints = [
    #         {"type": "ineq", "fun": inequality_constra}
    #     ]
    #     result=minimzer_fun()
    # if result.message not in ("Optimization terminated successfully","Iteration limit reached"):
    #       initial_spend_values = anl_df["MIN_BOUND_Spend"].values*0.95
    #       bounds = tuple(
    #           zip( anl_df["MIN_BOUND_Spend"].values*0.95,anl_df["MAX_BOUND_Spend"].values)
    #       )
    #       result=minimzer_fun()
    # print(anl_df["BASE BUDGET"].sum())
    # print(sum(result.x))
    # print((result))
    anl_df["OPTIMZED_SPEND"]=result.x
    anl_df["OPTIMZED_SPEND_110"]=anl_df["OPTIMZED_SPEND"]*1.1
    anl_df["OPTIMIZED_PROFIT"]=[calculate_revenue(row["OPTIMZED_SPEND"], row["C1"], row["C2"], row["C3"], row["C4"], row["CURVE_TYPE"]) for i, row in anl_df.iterrows()] #Original
    #anl_df["OPTIMIZED_PROFIT"]=[(row["BASE BUDGET"]/row["OPTIMZED_SPEND"]) for i, row in anl_df.iterrows()]
    anl_df["OPTIMIZED_PROFIT_110"]=[calculate_revenue(row["OPTIMZED_SPEND_110"], row["C1"], row["C2"], row["C3"], row["C4"], row["CURVE_TYPE"]) for i, row in anl_df.iterrows()]
    anl_df["Result"]=result.message
    # st.write(result.message)
    # st.write("optim_spend total:",sum(anl_df["OPTIMZED_SPEND"]),"actual_spend_total:",anl_df["BASE BUDGET"].sum()+CHANGE_SPEND)
    return anl_df

# edited_input_Data
# st.write()
                           
# st.write(edited_input_Data.groupby(["Brand","Tactic"]).agg({"Actual Spend (Selected Months)":"sum","MIN_BOUND":"mean","MAX_BOUND":"mean","COEF_ADJU":"mean","CPM_ADJU":"mean","MIN_BOUND_Spend":"sum","MAX_BOUND_Spend":"sum"}).reset_index())
# st.write(edited_input_Data_Grp)                         
# edited_input_Data
data=data[(data["MIN_BOUND"]>1)|(data["MAX_BOUND"]>1)].reset_index(drop=True)
# data
# edited_input_Data
if Optimizing_button: # and min_value<=Variable_Spend<=max_value
    if len(comments)==0:
        st.warning("⚠ Please enter the Scenario")
    else:
        
        with st.container(border=True):
                with st.spinner(text="Please wait.....its Optimizing ..."):

                    
                    unpivot_input_data(edited_input_Data)
                    if st.session_state["opt_type"]=="Budget Adjustment":
                            if spend_type=='Total Spend + Lock Budget by Period':
                                unpivot_edit_lock_period_input2(edi_df2)
                                
                    data=pd.concat([data,st.session_state['Brand_filter_Data']],ignore_index=True)
                    data=data[(data['MIN_BOUND_Spend'] > 1) & (data['MAX_BOUND_Spend'] > 1)].reset_index(drop=True)

                    # # data=data.merge(st.session_state['input_data'][['Brand',Tactic_Type_filter,'MIN_BOUND','MAX_BOUND']],on=['Brand',Tactic_Type_filter] )
                    # # data
                    # data=data.merge(st.session_state['input_data'][['Brand',Tactic_Type_filter,Period_filter,'MIN_BOUND','MAX_BOUND']],on=['Brand',Period_filter,Tactic_Type_filter] )
                    # # data
                    # data = session.sql(sql).collect()
                    
                    JOB_ID = datetime.now().strftime("%Y%m%d%H%M%S")
                    # st.write(JOB_ID)
                    #USER_EMAIL= st.experimental_user.email
                    USER_EMAIL = "test@gmail.com"
                    data['JOB_ID']=JOB_ID
                    data['USER_EMAIL']=USER_EMAIL
                    data['COMMENTS']=comments
                    data["DESCRIPTION"]=desc
                    data['DATE']=datetime.now().strftime("%d-%m-%Y %H:%M")
                    data['PERIOD_TYPE']=st.session_state["Period_filter"]
                    data['MONTH_LIST']=', '.join(map(str, Month_filter))
                    data['CHANNEL_TYPE']=st.session_state["Tactic_Type_filter"]
                    data['TACTIC_LIST']=', '.join(map(str, data['Tactic'].unique()))
                    data['BRAND_LIST']=st.session_state["optim_brand_filter"]
                    # update input
                    # Append dummy data
            
                    data=data.assign(OPTIMZED_SPEND=0, ADSTOCK_X_Sim=0.0, OPTIMIZED_PROFIT=0, OPTIMIZED_FEC=0,
                                     OPTIMZED_SPEND_110=0,ADSTOCK_X_Sim_110=0,OPTIMIZED_PROFIT_110=0)
                    data=data.assign(OPTIMZED_SPEND_TIME=0,OPTIMZED_SPEND_ITR=0)
                    data=data.assign(OPTIMZED_SPEND_MAX=0, ADSTOCK_X_Sim_MAX=0.0, OPTIMIZED_PROFIT_MAX=0, OPTIMIZED_FEC_MAX=0,
                                     OPTIMZED_SPEND_MAX_110=0,ADSTOCK_X_Sim_MAX_110=0,OPTIMIZED_PROFIT_MAX_110=0)
                    data=data.assign(OPTIMZED_SPEND_MAX_TIME=0,OPTIMZED_SPEND_MAX_ITR=0)
                    
                    data['STATUS']='In Progress'
                    data["BATCH_ID"]=1
                    data["SELECTED_MONTHS"]= np.where(data["Monthly"].isin(Month_filter), True, False)
                    data["OPT_TYPE"]=st.session_state["opt_type"]
                    # st.session_state["opt_type"]
                    data["OPT_TYPE_VALUE"]=np.where(st.session_state["opt_type"]=="Target FEC",st.session_state["Target_Profit"],st.session_state["Variable_Spend"])
                    
                    column_list = [
                        "Brand",'Level 1','Level 2','Level 3', "Actual Tactic", "BUDGET_WEEK_START_DT", "C1", "C2", "C3", "C4", "CPM", "Actual Spend (Selected Months)", "ADJUSTMENT_FACTOR", "REF_ADJ_FCTR", "ECOMM_ROI",
                        "TACTIC_ADJ_FCTR", "SEASONAL_ADJ_FCTR", "ADSTOCK_X", "CURVE_TYPE", "ADSTOCK_WEEK", "ADSTOCK", "MONTH_YEAR", "Group Tactic", "FEC_FCTR", "Yearly", "Quarterly", 
                        "Monthly", "Tactic", "Actual FEC", "Actual Profit", "GRP_MIN_BOUND_SPEND", "GRP_MAX_BOUND_SPEND", "MIN_BOUND", "MAX_BOUND", "JOB_ID", "USER_EMAIL", "COMMENTS", 
                        "DESCRIPTION", "DATE", "PERIOD_TYPE", "MONTH_LIST", "CHANNEL_TYPE", "TACTIC_LIST", "OPTIMZED_SPEND", "ADSTOCK_X_Sim", "OPTIMIZED_PROFIT", "OPTIMIZED_FEC", 
                        "OPTIMZED_SPEND_110", "ADSTOCK_X_Sim_110", "OPTIMIZED_PROFIT_110", "OPTIMZED_SPEND_TIME", "OPTIMZED_SPEND_ITR", "OPTIMZED_SPEND_MAX", "ADSTOCK_X_Sim_MAX", 
                        "OPTIMIZED_PROFIT_MAX", "OPTIMIZED_FEC_MAX", "OPTIMZED_SPEND_MAX_110", "ADSTOCK_X_Sim_MAX_110", "OPTIMIZED_PROFIT_MAX_110", "OPTIMZED_SPEND_MAX_TIME", 
                        "OPTIMZED_SPEND_MAX_ITR", "STATUS", "BATCH_ID", "SELECTED_MONTHS", "OPT_TYPE", "OPT_TYPE_VALUE", "LOCK_PERIOD", "LOCK_MIN_BOUND", "LOCK_MAX_BOUND",
                        "BRAND_LIST","COEF_ADJU","CPM_ADJU","LTE_FCTR","Actual LTE Profit","Actual LTE FEC",
                        "TARGET","OPTIM/SIM_LTE_PROFIT","OPTIM/SIM_LTE_FEC","Target_LTE_Profit"

                        
                    ]
                    data[["OPTIM/SIM_LTE_PROFIT","OPTIM/SIM_LTE_FEC"]]=0
                    data["TARGET"]=st.session_state["target_mode"]
                    # data["COEF_ADJU"]=1
                    # data["CPM_ADJU"]=1
                    data=data[column_list]
                    # st.write(st.session_state["target_mode"])
                    mapping = {"Hill": 1, "Power": 2, "Logistic": 3}
                    data["CURVE_TYPE"] = data["CURVE_TYPE"].map(mapping).fillna(0).astype(int)
                    # data
                    def objective_to_minimize(x, df):
                        idx = df.index.values
                        C1_values = df["C1"].values
                        C2_values = df["C2"].values
                        C3_values = df["C3"].values
                        C4_values = df["C4"].values
                        CURVE_TYPE_values = df["CURVE_TYPE"].values
                        CPM_values = df["CPM"].values
                        adjustment_factor_values = df["ADJUSTMENT_FACTOR"].values
                        ref_adj_fctr_values = df["REF_ADJ_FCTR"].values
                        tactic_adj_fctr_values = df["TACTIC_ADJ_FCTR"].values
                        seasonal_adj_fctr_values = df["SEASONAL_ADJ_FCTR"].values
                        ADSTOCK_X_values = df["ADSTOCK_X"].values
                        ECOMM_ROI_values = df["ECOMM_ROI"].values
                        CPM_ADJU=df["CPM_ADJU"].values
                        COEF_ADJU=df["COEF_ADJU"].values
                        
                        revenue = np.zeros(len(df))
                        for i in range(len(df)):
                            revenue[i] = calculate_revenue(x[idx[i]], C1_values[i], C2_values[i], C3_values[i], C4_values[i], CURVE_TYPE_values[i], CPM_values[i]*CPM_ADJU[i], adjustment_factor_values[i], ref_adj_fctr_values[i], tactic_adj_fctr_values[i]*COEF_ADJU[i], seasonal_adj_fctr_values[i], ADSTOCK_X_values[i], ECOMM_ROI_values[i])
                        # print("Profit :",np.sum(revenue),"Spend :",np.sum(x))
                        return -np.sum(revenue)
                  
                   

                    # ---------------------- HILL FUNCTION ----------------------
                    def hill_tensor(x, c1, c2, c3, c4, CPM,
                                    adjustment_factor, ref_adj_fctr,
                                    tactic_adj_fctr, seasonal_adj_fctr,
                                    ADSTOCK_X, ECOMM_ROI):

                        # Avoid divide-by-zero errors
                        x_safe = torch.where(CPM == 0, torch.zeros_like(CPM), x / CPM)

                        # Core Hill equation
                        hill_val = c1 + ((c2 - c1) * torch.pow(x_safe, c3)) / (torch.pow(c4, c3) + torch.pow(x_safe, c3))

                        # Apply adjustment, ref, ROI, and tactic/seasonal factors (uncommented full formula)
                        hill_val = ((adjustment_factor * ref_adj_fctr * hill_val) + (x_safe * ECOMM_ROI)) \
                                    * tactic_adj_fctr * seasonal_adj_fctr

                        # Zero-out invalids (x <= 0)
                        hill_val = torch.where(x <= 0, torch.zeros_like(hill_val), hill_val)

                        return hill_val


                    # ---------------------- LOGISTIC FUNCTION ----------------------
                    def logistic_tensor(x, c1, c2, c3, CPM,
                                        adjustment_factor, ref_adj_fctr,
                                        tactic_adj_fctr, seasonal_adj_fctr,
                                        ADSTOCK_X, ECOMM_ROI):

                        x_safe = torch.where(CPM == 0, torch.zeros_like(CPM), x / CPM)

                        # Core Logistic equation
                        logistic_val = c1 / (1 + torch.exp(c2 * (x_safe - c3)))

                        # Apply adjustment, ref, ROI, and tactic/seasonal factors
                        logistic_val = ((adjustment_factor * ref_adj_fctr * logistic_val) + (x_safe * ECOMM_ROI)) \
                                        * tactic_adj_fctr * seasonal_adj_fctr

                        logistic_val = torch.where(x <= 0, torch.zeros_like(logistic_val), logistic_val)

                        return logistic_val


                    # ---------------------- POWER FUNCTION ----------------------
                    def power_tensor(x, c1, c2, CPM,
                                    adjustment_factor, ref_adj_fctr,
                                    tactic_adj_fctr, seasonal_adj_fctr,
                                    ADSTOCK_X, ECOMM_ROI):

                        x_safe = torch.where(CPM == 0, torch.zeros_like(CPM), x / CPM)

                        # Core Power equation
                        power_val = torch.exp(c1) * torch.pow(x_safe, c2)

                        # Apply adjustment, ref, ROI, and tactic/seasonal factors
                        power_val = ((adjustment_factor * ref_adj_fctr * power_val) + (x_safe * ECOMM_ROI)) \
                                    * tactic_adj_fctr * seasonal_adj_fctr

                        power_val = torch.where(x <= 0, torch.zeros_like(power_val), power_val)

                        return power_val

                    def calculate_revenue_tensor(
                        x, c1, c2, c3, c4, curve_type,
                        CPM, adj, ref_adj, tactic_adj, seasonal_adj,
                        ADSTOCK, ROI
                    ):
                        # --- Compute all curve types in one shot ---
                        h = hill_tensor(x, c1, c2, c3, c4, CPM, adj, ref_adj, tactic_adj, seasonal_adj, ADSTOCK, ROI)
                        l = logistic_tensor(x, c1, c2, c3, CPM, adj, ref_adj, tactic_adj, seasonal_adj, ADSTOCK, ROI)
                        p = power_tensor(x, c1, c2, CPM, adj, ref_adj, tactic_adj, seasonal_adj, ADSTOCK, ROI)
                        # n = custom(x, tactic_adj, seasonal_adj, ROI)  # optional
                        # n = p

                        # --- Blend by curve type (1=hill, 3=logistic, 2=power, else=power) ---
                        ct = curve_type
                        rev = torch.where(ct == 1, h,
                                torch.where(ct == 3, l,
                                torch.where(ct == 2, p, p)))

                        # --- Handle invalid values (same logic as NumPy version) ---
                        invalid_mask = (x.round() <= 1) | (CPM == 0)

                        # Zero out invalid rows safely
                        rev = torch.where(invalid_mask, torch.zeros_like(rev), rev)

                        return rev

                        # return rev.masked_fill(invalid, 0.0)
                    def objective_to_minimize(x, df):

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        # Fix for x_idx
                        x_tensor = torch.tensor(x, dtype=torch.float64, device=device)

                        # Convert dataframe columns to tensors (same labels and order as in reference)
                        idx                     = torch.tensor(df.index.values, dtype=torch.int64, device=device)
                        C1_values               = torch.tensor(df["C1"].values, dtype=torch.float32, device=device)
                        C2_values               = torch.tensor(df["C2"].values, dtype=torch.float32, device=device)
                        C3_values               = torch.tensor(df["C3"].values, dtype=torch.float32, device=device)
                        C4_values               = torch.tensor(df["C4"].values, dtype=torch.float32, device=device)
                        CURVE_TYPE_values       = torch.tensor(df["CURVE_TYPE"].values, dtype=torch.int64, device=device)
                        CPM_values              = torch.tensor(df["CPM"].values, dtype=torch.float64, device=device)
                        adjustment_factor_values = torch.tensor(df["ADJUSTMENT_FACTOR"].values, dtype=torch.float32, device=device)
                        ref_adj_fctr_values     = torch.tensor(df["REF_ADJ_FCTR"].values, dtype=torch.float32, device=device)
                        tactic_adj_fctr_values  = torch.tensor(df["TACTIC_ADJ_FCTR"].values, dtype=torch.float32, device=device)
                        seasonal_adj_fctr_values = torch.tensor(df["SEASONAL_ADJ_FCTR"].values, dtype=torch.float32, device=device)
                        ADSTOCK_X_values        = torch.tensor(df["ADSTOCK_X"].values, dtype=torch.float32, device=device)
                        ECOMM_ROI_values        = torch.tensor(df["ECOMM_ROI"].values, dtype=torch.float32, device=device)
                        CPM_ADJU_values         = torch.tensor(df["CPM_ADJU"].values, dtype=torch.float64, device=device)
                        COEF_ADJU_values        = torch.tensor(df["COEF_ADJU"].values, dtype=torch.float32, device=device)

                        # Compute adjusted parameters
                        CPM_eff = CPM_values * CPM_ADJU_values
                        tactic_eff = tactic_adj_fctr_values * COEF_ADJU_values

                        # Compute revenue per row using your tensor version of revenue function
                        revenue = calculate_revenue_tensor(
                            x_tensor[idx],
                            C1_values,
                            C2_values,
                            C3_values,
                            C4_values,
                            CURVE_TYPE_values,
                            CPM_eff,
                            adjustment_factor_values,
                            ref_adj_fctr_values,
                            tactic_eff,
                            seasonal_adj_fctr_values,
                            ADSTOCK_X_values,
                            ECOMM_ROI_values
                        )

                        # Optional: print like original if you want to debug
                        # print("Profit :", torch.sum(revenue).item(), "Spend :", torch.sum(x_tensor).item())

                        # Return negative total revenue (objective to minimize)
                        return -float(torch.sum(revenue))

                    initial = data["MIN_BOUND"].to_numpy()
                    bounds = tuple(zip(data["MIN_BOUND"].to_numpy(), data["MAX_BOUND"].to_numpy()))
                    def custom_gradient(x, data):
                        epsilon = np.sqrt(np.finfo(float).eps) * np.maximum(1, np.abs(x))
                        return approx_fprime(x, objective_to_minimize, epsilon, data)
                    
                    def equality_constraint(x):
                        return (np.sum(x)) - data["Actual Spend (Selected Months)"].sum()+1e-2
                    
                    constraints = [{
                        'type': 'eq',
                        'fun': equality_constraint,
                    }]
                    # data
                    input_cache=st.session_state['input_data'][['Brand','Level 1','Level 2','Level 3','Tactic',st.session_state["Period_filter"],'MIN_BOUND_Spend','MAX_BOUND_Spend',"Actual Spend (Selected Months)","COEF_ADJU","CPM_ADJU"]]

                    def group_tactic(x, group_name):
                                               
                        lb = input_cache.loc[input_cache["Level 1"] == group_name, "Actual Spend (Selected Months)"].sum().astype(int)
                        ub = input_cache.loc[input_cache["Level 1"] == group_name, "Actual Spend (Selected Months)"].sum().astype(int)
                        idx_list = data[data["Level 1"] == group_name.replace(" total", "")].index.tolist()
                        x_sum = np.sum([x[i] for i in idx_list])
                        # st.write(x_sum , lb, ub , x_sum)
                        return x_sum - lb, ub - x_sum
                    group_names = [input_cache for input_cache in input_cache["Level 1"].unique().tolist() if "total" in str(input_cache).lower()]

                    # for group_name in group_names:
                    #     st.write("Adding constraint for group:", group_name)
                    #     constraints.append({"type": "ineq", "fun": lambda x, group_name=group_name: group_tactic(x, group_name)})
                    
                    
                    def group_tactic(x, group_name, level_2_value, level_3_value):
                        # Filter by 'Level 1', 'Level 2', and 'Level 3' for the lower and upper bounds
                        lb = input_cache.loc[
                            (input_cache["Level 1"] == group_name) &
                            (input_cache["Level 2"] == level_2_value) &
                            (input_cache["Level 3"] == level_3_value),
                            "MIN_BOUND_Spend"
                        ].sum()

                        ub = input_cache.loc[
                            (input_cache["Level 1"] == group_name) &
                            (input_cache["Level 2"] == level_2_value) &
                            (input_cache["Level 3"] == level_3_value),
                            "MAX_BOUND_Spend"
                        ].sum()

                        # Filter for matching indices in 'Level 1', 'Level 2', and 'Level 3'
                        # idx_list = data[
                        #     (data["Level 1"] == group_name.replace(" total", "")) &
                        #     (data["Level 2"] == level_2_value.replace(" total", "")) &
                        #     (data["Level 3"] == level_3_value.replace(" total", "")) 
                        # ].index.tolist()
                        # Prepare the conditions dynamically based on the non-blank values of levels
                        conditions = (data["Level 1"] == group_name.replace(" total", ""))  # Always filter by Level 1

                        # Add condition for Level 2 if it's not blank
                        if level_2_value:
                            conditions &= (data["Level 2"] == level_2_value.replace(" total", ""))

                        # Add condition for Level 3 if it's not blank
                        if level_3_value:
                            conditions &= (data["Level 3"] == level_3_value.replace(" total", ""))

                        # Apply the conditions and get the index list
                        idx_list = data[conditions].index.tolist()

                        # Calculate the sum for the selected indices in x
                        x_sum = np.sum([x[i] for i in idx_list])
                        # st.write(f"Level 1 {group_name}, Level 2: {level_2_value}, Level 3: {level_3_value},{lb},{ub},{x_sum}")
                        return x_sum - lb, ub - x_sum
                        
                                     
                    for group_name, level_2_value, level_3_value in input_cache[['Level 1', 'Level 2', 'Level 3']].drop_duplicates().values:
                        # Dynamically add the constraint for each group_name, level_2_value, and level_3_value
                        
                        constraints.append({
                            "type": "ineq",  # This means we're creating an inequality constraint
                            "fun": lambda x, group_name=group_name, level_2_value=level_2_value, level_3_value=level_3_value: group_tactic(x, group_name, level_2_value, level_3_value)
                        })

                    # st.write(constraints)
                    result = minimize(
                        fun=objective_to_minimize,
                        x0=initial,
                        method='SLSQP',
                        args=(data,),
                        jac='2-point',
                        constraints=constraints,
                        bounds=bounds,
                        options={
                            'disp': True,
                            "maxiter": 10000,'eps':1e-6,'ftol':1e1
                        }
                        )
                    data['OPTIMZED_SPEND']=result.x
                    mapping = {"Hill": 1, "Power": 2, "Logistic": 3}
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    data["CURVE_TYPE"] = data["CURVE_TYPE"].map(reverse_mapping).fillna("Unknown")
                    # st.write(result)
                    # st.dataframe(data)
                    # data['OPTIMZED_SPEND'] = data['Actual Spend (Selected Months)']

                    
                    def update_optimized_profit_by_tactic(data, df_compare):
                        data = data.copy()
                        # st.write("data before update_optimized_profit_by_tactic")
                        # data
                        # df_compare 
                        df_compare = df_compare.copy()
    
                        # Round spend_diff for matching
                        data['spend_diff'] = (data['OPTIMZED_SPEND'] / data['Actual Spend (Selected Months)']).round(2)
                        df_compare['spend_multiplier'] = df_compare['spend_multiplier'].round(2)
    
                        # Initialize OPTIMIZED_PROFIT as NaN
                        data['OPTIMIZED_PROFIT'] = float('nan')
    
                        # Loop over unique Actual Tactic values in data
                        for tactic in data['Actual Tactic'].unique():
                            # Filter rows in data for this tactic
                            data_idx = data['Actual Tactic'] == tactic
                            spend_diff_subset = data.loc[data_idx, 'spend_diff']
        
                            # Filter df_compare rows for matching channel
                            df_subset = df_compare[df_compare['channel'] == tactic]
        
                            # Create lookup for this channel
                            lookup = dict(zip(df_subset['spend_multiplier'], df_subset['incremental_outcome']))
        
                            # Map only for this tactic
                            data.loc[data_idx, 'OPTIMIZED_PROFIT'] = spend_diff_subset.map(lookup)
    
                        # Drop temporary spend_diff column if you don't want it
                        data = data.drop(columns=['spend_diff'])
    
                        return data

                    df_compare=pd.read_excel(r'final_response_curves.xlsx',sheet_name='response_curves')
                    df_compare = df_compare[df_compare['metric'] == 'mean']
                    # df_compare['channel']=df_compare['channel'].str.replace(r'\btotal\b', '', regex=True)
                    data=update_optimized_profit_by_tactic(data,df_compare)

                    # data.to_csv("Optimization Output Files\compare.csv")
                    

                    # st.write(result)

                    # data['OPTIMIZED_PROFIT']=[calculate_revenue(row['OPTIMZED_SPEND'], 
                    #                          row["C1"], row["C2"], row["C3"], row["C4"], 
                    #                          row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], 
                    #                          row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], row["SEASONAL_ADJ_FCTR"], 
                    #                          row["ADSTOCK_X"], row["ECOMM_ROI"]) for i, row in data.iterrows()]
                    
                    #data['OPTIMIZED_PROFIT'] = data['OPTIMIZED_PROFIT'] / 20
                    data['OPTIMIZED_FEC'] = data['OPTIMIZED_PROFIT']*data['FEC_FCTR']
                    data['Actual ROI'] = data['Actual Profit'] / data['Actual Spend (Selected Months)']
                    data['OPTIMIZED_ROI'] = data['OPTIMIZED_PROFIT'] / data['OPTIMZED_SPEND']
                    #data['OPTIMIZED_PROFIT']=data['OPTIMZED_SPEND'] / data['Actual Spend (Selected Months)']
                    # st.dataframe(data[['Actual Tactic', 'Actual Spend (Selected Months)', 'OPTIMZED_SPEND', 'CPM', 'CURVE_TYPE', 
                    #                    'Actual ROI', 'OPTIMIZED_ROI', 
                    #                    'Actual FEC' ,'OPTIMIZED_FEC']])

                    # def update_optimized_profit(data, df_compare):
                    #     # Compute spend_diff temporarily
                    #     spend_diff = (data['OPTIMZED_SPEND'] / data['Actual Spend (Selected Months)']).round(2)
    
                    #     # Round df_compare spend_multiplier too (to ensure same precision)
                    #     df_compare['spend_multiplier'] = df_compare['spend_multiplier'].round(2)

                    #     # Create a lookup dictionary for faster mapping
                    #     lookup = dict(zip(df_compare['spend_multiplier'], df_compare['incremental_outcome']))
    
                        

                    #     # Map matching incremental_outcome from df_compare to data
                    #     data['OPTIMIZED_PROFIT'] = spend_diff.map(lookup)
                        

                    #     return data

                    # st.dataframe(data)
                    # st.write(result['message'])                 
                    output_path = "Output_data.csv"
                    job = data['JOB_ID'].iloc[0]
                    scenario = data['COMMENTS'].iloc[0]

                    with st.success("Optimization Task successfully initiated!"):
                        data['STATUS'] = result['message']

                        st.page_link("pages/3_Output.py", 
                                    label=f"👉 Click here to view the report. 'Optim-{job}' & {scenario}")
                    first_row_for_base = data.iloc[[0]].copy()
                    first_row_for_base['Actual Tactic'] = 'Base Total'
                    first_row_for_base['Tactic'] = 'Base Total'
                    first_row_for_base['Level 1'] = 'Base Total'
                    first_row_for_base['Level 2']= 'Base Total'
                    first_row_for_base['Level 3']= 'Base Total'
                    # df_compare
                    # df_compare[df_compare['channel'] == 'BASE']['spend']
                    first_row_for_base['Actual Spend (Selected Months)'] = df_compare[df_compare['channel'] == 'BASE']['spend'].iloc[0]
                    first_row_for_base['Actual Profit'] = df_compare[df_compare['channel'] == 'BASE']['incremental_outcome'].iloc[0]
                    first_row_for_base['Actual FEC'] =df_compare[df_compare['channel'] == 'BASE']['incremental_outcome'].iloc[0]
                    first_row_for_base['OPTIMZED_SPEND'] = df_compare[df_compare['channel'] == 'BASE']['spend'].iloc[0]
                    first_row_for_base['OPTIMIZED_PROFIT'] = df_compare[df_compare['channel'] == 'BASE']['incremental_outcome'].iloc[0]
                    first_row_for_base['OPTIMIZED_FEC'] =df_compare[df_compare['channel'] == 'BASE']['incremental_outcome'].iloc[0]
                    # first_row_for_base
                    data = pd.concat([first_row_for_base, data], ignore_index=True)

                    data.to_csv(output_path, mode = 'a', header = False ,index=False)
                    # data.to_csv(output_path, mode = 'a', header = False ,index=False)
###################################################################################################

                    # Aggregate the data for each metric
                    spend_aggregated_df = data.groupby('Actual Tactic')[['OPTIMZED_SPEND', 'Actual Spend (Selected Months)']].sum().reset_index()
                    fec_aggregated_df = data.groupby('Actual Tactic')[['OPTIMIZED_FEC', 'Actual FEC']].sum().reset_index()
                    roi_aggregated_df = data.groupby('Actual Tactic')[['OPTIMIZED_ROI', 'Actual ROI']].sum().reset_index()

                    
                    def format_number(n):
                        if n >= 1_000_000_000:
                            return f"{n / 1_000_000_000:.1f}B"
                        elif n >= 1_000_000:
                            return f"{n / 1_000_000:.1f}M"
                        elif n >= 1_000:
                            return f"{n / 1_000:.1f}K"
                        else:
                            return f"{n:.2f}"
                        
                    
                    spend_aggregated_df['Formatted Ac_Spend'] = spend_aggregated_df['Actual Spend (Selected Months)'].apply(format_number)
                    spend_aggregated_df['Formatted Op_Spend'] = spend_aggregated_df['OPTIMZED_SPEND'].apply(format_number)

                    fec_aggregated_df['Formatted Op_FEC'] = fec_aggregated_df['OPTIMIZED_FEC'].apply(format_number)
                    fec_aggregated_df['Formatted Ac_FEC'] = fec_aggregated_df['Actual FEC'].apply(format_number)

                    roi_aggregated_df['Formatted Op_ROI'] = roi_aggregated_df['OPTIMIZED_ROI'].apply(format_number)
                    roi_aggregated_df['Formatted Ac_ROI'] = roi_aggregated_df['Actual ROI'].apply(format_number)


                    # Define bar colors
                    bar_colors = {
                        'Yearly': '#1f77b4',  # Dark blue
                        'Optimized': '#aec7e8'  # Light blue
                    }

                    # Create subplots
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Spend", "Revenue", "ROI"),
                        horizontal_spacing=0.07
                    )

                    # Add traces for Spend
                    fig.add_trace(
                        go.Bar(
                            y=spend_aggregated_df['Actual Tactic'],
                            x=spend_aggregated_df['Actual Spend (Selected Months)'],
                            name='Yearly',
                            orientation='h',
                            #text=spend_aggregated_df['Actual Spend (Selected Months)'],
                            text=spend_aggregated_df['Formatted Ac_Spend'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Yearly'],
                            insidetextanchor='middle'
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Bar(
                            y=spend_aggregated_df['Actual Tactic'],
                            x=spend_aggregated_df['OPTIMZED_SPEND'],
                            name='Optimized',
                            orientation='h',
                            text=spend_aggregated_df['Formatted Op_Spend'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Optimized'],
                            insidetextanchor='middle'
                        ),
                        row=1, col=1
                    )

                    # Add traces for FEC
                    fig.add_trace(
                        go.Bar(
                            y=fec_aggregated_df['Actual Tactic'],
                            x=fec_aggregated_df['Actual FEC'],
                            name='Yearly',
                            orientation='h',
                            text=fec_aggregated_df['Formatted Ac_FEC'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Yearly'],
                            insidetextanchor='middle',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                    fig.add_trace(
                        go.Bar(
                            y=fec_aggregated_df['Actual Tactic'],
                            x=fec_aggregated_df['OPTIMIZED_FEC'],
                            name='Optimized',
                            orientation='h',
                            text=fec_aggregated_df['Formatted Op_FEC'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Optimized'],
                            insidetextanchor='middle',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                    # Add traces for ROI
                    fig.add_trace(
                        go.Bar(
                            y=roi_aggregated_df['Actual Tactic'],
                            x=roi_aggregated_df['Actual ROI'],
                            name='Yearly',
                            orientation='h',
                            text=roi_aggregated_df['Formatted Ac_ROI'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Yearly'],
                            insidetextanchor='middle',
                            showlegend=False
                        ),
                        row=1, col=3
                    )

                    fig.add_trace(
                        go.Bar(
                            y=roi_aggregated_df['Actual Tactic'],
                            x=roi_aggregated_df['OPTIMIZED_ROI'],
                            name='Optimized',
                            orientation='h',
                            text=roi_aggregated_df['Formatted Op_ROI'],
                            #texttemplate="%{x:₩,.2f}",
                            textposition='auto',
                            textfont=dict(
                                        color='black',      # White text for readability
                                        size=19,
                                        family='Arial, sans-serif'
                                    ),
                            marker_color=bar_colors['Optimized'],
                            insidetextanchor='middle',
                            showlegend=False
                        ),
                        row=1, col=3
                    )

                    # Update layout
                    fig.update_layout(
                        title='Yearly vs Optimized',
                        barmode='group',
                        height=1000,
                        width=1800,
                        plot_bgcolor='white',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12, color='black'),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.05,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=14),
                            traceorder="reversed"
                        ),
                        margin=dict(l=150, r=50, t=150, b=50)
                    )

                    # Update axes for each subplot
                    fig.update_xaxes(
                        title='Spend',
                        tickprefix="₩",
                        tickformat=",.0f",
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        row=1, col=1
                    )

                    fig.update_xaxes(
                        title='Revenue',
                        tickprefix="₩",
                        tickformat=",.0f",
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        row=1, col=2
                    )

                    fig.update_xaxes(
                        title='ROI',
                        tickprefix="₩",
                        tickformat=",.0f",
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        row=1, col=3
                    )

                    fig.update_yaxes(
                        title='Tactic',
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        tickfont=dict(size=14),
                        row=1, col=1
                    )

                    fig.update_yaxes(
                        showticklabels=False,
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        tickfont=dict(size=14),
                        row=1, col=2
                    )

                    fig.update_yaxes(
                        showticklabels=False,
                        showgrid=True,
                        gridcolor='lightgrey',
                        gridwidth=1,
                        tickfont=dict(size=14),
                        row=1, col=3
                    )

                    # Show the chart
                    # st.plotly_chart(fig)


    # st.success(f"Optimization Task successfully initiated  ---- Please check the Output page to view the report")

    # st.page_link("pages/3_Scenario Output.py", label="Go to Output Page")













    