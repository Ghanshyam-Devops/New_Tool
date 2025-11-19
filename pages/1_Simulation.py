import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import random
from datetime import timedelta, datetime
from scipy.special import expit


st.set_page_config(layout="wide")
#from snowflake.snowpark.context import get_active_session
#session = get_active_session() 
# rollup_name_qry=f"""removed query """
# rollup_name=session.sql(rollup_name_qry).to_pandas()

# st.session_state['rollup_name']=rollup_name
BUDGET_BATCH_ID=205
BUDGET_YEAR=2024
# st.set_page_config(layout="wide")

t=60
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
            div.st-emotion-cache-1d8vwwt.e1lln2w84
            {
            background-color: #ffffff;
            }
            div.st-emotion-cache-1cvtqh0.e1f1d6gn0
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
            button.st-emotion-cache-wasuqa.ef3psqc13
            {
            background-color: #A6C9EC;
            
            
            color: black;
            }
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
            div.st-emotion-cache-15hroa0.e1nzilvr5
            {
            # color: white;
            }

   # div.st-emotion-cache-1629p8f.e1nzilvr2
   #          {
                           
                
   #               display: flex;
   #              justify-content: center; /* Horizontally center */
   #              align-items: center;display: grid;
   #              place-items: center;
   #              padding-bottom: 5px;
               

   #          }

            [data-testid=stSidebarNavItems] {
                font-size: 18px;
                background-color: #ffffff;
                padding: 20px;
                margin-top: 40px; /* Adjust this value as needed */
            }

             
            
   
#     [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p {
#         /* Styles for the label text for checkbox and toggle */
        
#         width: 200px;
#         margin-top: 1rem;
#     }
    
#     [data-baseweb="checkbox"] div {
#         /* Styles for the slider container */
#         height: 1.5rem;
#         width: 4rem;
#     }
#     [data-baseweb="checkbox"] div div {
#         /* Styles for the slider circle */
#         height: 1.5rem;
#         width: 1.5rem;
#         transform: translateX(0px);

        
        
#     }
# [data-baseweb="checkbox"] input:checked + div div div {
#     /* Styles for the slider circle when checkbox is checked */
#     transform: matrix(1, 0, 0, 1, 16, 0); /* Updated transform when checked */
# }

#     [data-testid="stCheckbox"] label span {
#         /* Styles the checkbox */
#         height: 8rem;
#         width: 8rem;
#     }
    
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
    elif curve_type == 'Linear':
        return linear(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    else:
        return NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI)
    
def hill(x, c1, c2, c3, c4, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    #return ((adjustment_factor * ref_adj_fctr * (c1 + ((c2 - c1) * ((x / CPM) + ADSTOCK_X) ** c3) / (c4**c3 + ((x / CPM) + ADSTOCK_X) ** c3))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    x = x/CPM
    #return ((adjustment_factor * ref_adj_fctr * c1 / (1 + expit(c2 * (x - c3)))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return c1 + ((c2 - c1) * (np.power(x, c3))) / (np.power(c4, c3) + np.power(x, c3))


# Logistic function
def logistic(x, c1, c2, c3, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    
    x = x/CPM
    #return ((adjustment_factor * ref_adj_fctr * (c1 / (1 + expit(c2 * (((x / CPM) + ADSTOCK_X) - c3))))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return c1 / (1 + np.exp(c2 * (x - c3)))


# Power function
def power(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    
    x = x/CPM
    #return ((adjustment_factor * ref_adj_fctr * (expit(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (x * ECOMM_ROI)) * tactic_adj_fctr * seasonal_adj_fctr
    return np.exp(c1) * (x ** c2) 

# NullCurve function
def NullCurve(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    if x == 0:
        return 0
    return ((adjustment_factor * ref_adj_fctr * (expit(c1) * np.power(((x / CPM) + ADSTOCK_X), c2))) + (
        x * ECOMM_ROI
    )) * tactic_adj_fctr * seasonal_adj_fctr

def linear(x, c1, c2, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI):
    x = x/CPM
    return c1 + c2 * x


def get_snowflake_data():

    #data = pd.read_csv(r'C:\Users\Prakhar.saxena\Downloads\Amout_Calc_22-7-25.csv')
    data = pd.read_excel(r'https://simtooladls.blob.core.windows.net/simt/Input_data.xlsx')
    # data.drop('BRAND', inplace=True, axis=1)
    
    # column_lengths = {col: len(col) for col in data.columns}
    # st.write(data.head())

    # data['MONTH_YEAR'] = pd.to_datetime(data['MONTH_YEAR'])
    # data['Yearly'] = data['Yearly'].astype(str)
    # data['Quarterly'] = data['Quarterly'].astype(str)
    # data['Monthly'] = data['Monthly'].dt.strftime('%b-%y')
    # data['Brand'] = 'C5i'

    data['MONTH_YEAR'] = pd.to_datetime(data['BUDGET_WEEK_START_DT'])
    #st.write(data.head())
    data['Yearly'] = data['MONTH_YEAR'].dt.year.astype(str)  # → "2025"
    data['Monthly'] = data['MONTH_YEAR'].dt.strftime('%b-%y')  # → "Jan-25"
    data['Quarterly'] = 'Q' + data['MONTH_YEAR'].dt.quarter.astype(str) + '-' + data['MONTH_YEAR'].dt.year.astype(str).str[-2:]
    #st.write(data.head())
    data.rename(columns={'Spend': 'Actual Spend (Selected Months)',
                         'TACTIC' : 'Tactic'}, inplace=True)
    
    data['Subbrand'] = 'Brand1'
    data['FEC'] = data['Actual Profit']
    data['Publishername'] = 'Custom_Publish'
    data['Channel'] = "Channel1"
    data['ADSTOCK_WEEK'] = 1
    data['ADSTOCK'] = 0
    data["ADJUSTMENT_FACTOR"] = 1
    data["REF_ADJ_FCTR"] = 1
    data["ECOMM_ROI"] = 1
    data["TACTIC_ADJ_FCTR"] = 1
    data["SEASONAL_ADJ_FCTR"] = 1
    data['FEC_FCTR'] = 1
    data['LTE_FCTR'] = 1
    data['CPM_Original'] = data['CPM']
    data['Group Tactic'] = 'Group'
    # data['Actual Profit'] = data['Calculated']

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

    # st.write(data.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                'Actual Profit' : 'sum'}))
    
    # st.write("TEMPLATE")
    # st.dataframe(data[['Brand', 'Tactic', 'BUDGET_WEEK_START_DT', 'Yearly', 'Monthly', 'Quarterly', 'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'TACTIC_ADJ_FCTR',
    #                    'SEASONAL_ADJ_FCTR']])

    data = data[['Brand', 'Subbrand', 'Tactic', 'Actual Tactic', 'Group Tactic','Publishername', 'Channel', 'BUDGET_WEEK_START_DT', 'Actual Spend (Selected Months)','CPM', 'CURVE_TYPE',
                       'C1', 'C2', 'C3', 'C4', 'Actual Profit', 'Impression', 'MONTH_YEAR', "Yearly", 'Monthly', 'Quarterly', 'ADSTOCK', 'ADSTOCK_WEEK', 'ADSTOCK_X_Ana',
                       'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'ECOMM_ROI', 'TACTIC_ADJ_FCTR', 'SEASONAL_ADJ_FCTR', "FEC_FCTR", "LTE_FCTR", 'CPM_Original', 'Actual FEC']]
    # st.dataframe(data)

    data = data[data['Monthly'].isin(['Jan-25','Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25'])]
    
    #st.write("SNOWFLAKE")
    #st.dataframe(data)

    # st.dataframe(data.groupby('Tactic').agg({'Actual Spend (Selected Months)' : 'sum',
    #                                          'Actual FEC' : 'sum'}))

    st.session_state.snapshot_data = data
    st.session_state.sim_snowflake_data = data




def old_get_snowflake_data():

    mmm_df = pd.read_csv(r'02_MODELING_HBR_LEVEL1_SK_M_MS_output.csv')
    curve_df = pd.read_csv(r'Best_Curve_Coefficients.csv')

    data = pd.merge(mmm_df, curve_df, left_on= 'key' ,right_on='TACTIC', how='left')
    data = data[data['key'] != 'Paid Traditional']
    data.rename(columns={'spends': 'Actual Spend (Selected Months)',
                             'nsv_value' : 'FEC',
                             'brand' : 'Brand',
                             'sub_brand_x' : 'Subbrand',
                             'TACTIC' : 'Actual Tactic'}, inplace=True)
    
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'] + DateOffset(years=1)
    data['Yearly'] = data['date'].dt.year.astype(str)  # → "2025"
    data['Monthly'] = data['date'].dt.strftime('%b-%y')  # → "Jan-25"
    data['Quarterly'] = 'Q' + data['date'].dt.quarter.astype(str) + '-' + data['date'].dt.year.astype(str).str[-2:]
    data['Channel'] = data['Actual Tactic']
    data['Publishername'] = 'Custom_Publish'
    data['BUDGET_WEEK_START_DT'] = data['date']
    data['ADSTOCK_WEEK'] = 1
    data['CPM'] = 50
    data['ADSTOCK'] = 0
    data["ADJUSTMENT_FACTOR"] = 1
    data["REF_ADJ_FCTR"] = 1
    data["ECOMM_ROI"] = 1
    data["TACTIC_ADJ_FCTR"] = 1
    data["SEASONAL_ADJ_FCTR"] = 1
    data['FEC_FCTR'] = 1
    data['LTE_FCTR'] = 1
    data['MONTH_YEAR'] = data['date']
    data['Tactic'] = data['Actual Tactic']
    data['CPM_Original'] = data['CPM'] * 0.6
    data['Group Tactic'] = 'Group'
    data = data[data['date'].dt.year == 2025]


    data.reset_index(drop=True, inplace=True)
    data=data.sort_values(by=['Brand', 'Actual Tactic', 'BUDGET_WEEK_START_DT'], ascending=[True, True, False])
    
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
    
    data['Actual Profit']=[calculate_revenue(row['Actual Spend (Selected Months)'], 
                                             row["C1"], row["C2"], row["C3"], row["C4"], 
                                             row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], 
                                             row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], row["SEASONAL_ADJ_FCTR"], 
                                             row["ADSTOCK_X_Ana"], row["ECOMM_ROI"]) for i, row in data.iterrows()]

    data['Actual Profit'] = data['Actual Profit'] / 100

     
    data["Actual FEC"] = data['Actual Profit']*data['FEC_FCTR']
    
    data["Actual LTE Profit"]=data["Actual Profit"]*data["LTE_FCTR"]
    data["Actual LTE FEC"]=data["Actual LTE Profit"]*data["FEC_FCTR"]
    
    # st.write("SNOWFLAKE")
    # st.dataframe(data)
    st.session_state.snapshot_data = data
    st.session_state.sim_snowflake_data = data
  
    

if 'sim_snowflake_data' not in st.session_state:
    get_snowflake_data()
    
data=st.session_state.sim_snowflake_data  

with st.sidebar.container(border=True):

    new_or_exi =st.selectbox(   " New or Existing Scenario:", ['New Scenario'
                                                               # ,'Existing Scenario'
                                                               ])  

if new_or_exi=='Existing Scenario':
    from snowflake.snowpark.context import get_active_session
    session = get_active_session() 
    user_email=str(st.experimental_user.email)
    query_exi_scn=f"""    removed query"""
    exi_sce = session.sql(query_exi_scn).collect()
    for_filter=pd.DataFrame(exi_sce)
    # st.session_state.exi_sce=exi_sce
    # for_filter=st.session_state.exi_sce
    # for_filter
    with st.sidebar.container(border=True):

        exi_brand =st.selectbox(   " Brand:",for_filter['Brand'].unique()

    )
    
    with st.sidebar.container(border=True):

        comments_filter =st.selectbox(   " Choose the Scenario:",for_filter[for_filter["Brand"]==exi_brand]['COMMENTS'].unique()

    )
    Tactic_Type_filter=for_filter['INPUT_TYPE'].iloc[0]
    if Tactic_Type_filter=='Actual':
        subbrand_filter=for_filter['Subbrand']
        channel_filter=for_filter['Channel']
        publisher_filter=for_filter['Publishername']
    elif  Tactic_Type_filter=='UMM Tactic Only':
        
      Tactic_filter=for_filter['Tactic']

    elif Tactic_Type_filter=='EmPlanner Tactic':
        Tactic_filter=for_filter['Group Tactic']
    
    brand_filter=for_filter['Brand'].iloc[0]
   # st.write(for_filter)
    Period_filter=for_filter['Period_filter'].iloc[0]
    Type=for_filter['type'].iloc[0]
    Yr_filter=for_filter['Yr_filter'].iloc[0]
    Month_filter=ast.literal_eval(for_filter['Month_filter'].iloc[0])
    st.session_state.exi_sce=for_filter[(for_filter['Brand']==exi_brand)&(for_filter['COMMENTS']==comments_filter)]
    
else:
    with st.sidebar.container(border=True):
        # st.write(data["Brand"].unique().tolist())
        brand_list=data["Brand"].dropna().unique().tolist()
        
        brand_list.sort()
        brand_filter =st.selectbox(   " Brand Name:",brand_list
                                      # ,index=brand_list.index("Aleve")
        
        )
    
    with st.sidebar.container(border=True):
        #Tactic_Type_filter =st.selectbox(   " Input Type :", ["Actual",'UMM Tactic x Subbrand','UMM Tactic Only','EmPlanner Tactic','EmPlanner x Subbrand'])
        Tactic_Type_filter =st.selectbox(   " Input Type :", ['UMM Tactic Only'])

    
    with st.sidebar.container(border=True):
        if Tactic_Type_filter=='UMM Tactic Only':
            tactic_list=data[data["Brand"]==brand_filter]["Actual Tactic"].unique().tolist()
            Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
            if "Select All" in Tactic_filter:
                Tactic_filter=tactic_list
    
        elif Tactic_Type_filter=='EmPlanner Tactic':
            tactic_list=data[data["Brand"]==brand_filter]["Group Tactic"].unique().tolist()
            # [x for x in tactic_list if x != 0]
            # st.write(tactic_list)
            Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
            if "Select All" in Tactic_filter:
                Tactic_filter=tactic_list
                
        elif Tactic_Type_filter=='Actual':
            with st.sidebar.container(border=True):
                
                subbrand_filter =st.multiselect(   " SubBrand:", ["Select All"]+data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist(),default="Select All")
                if "Select All" in subbrand_filter:
                    subbrand_filter=data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist()
            
            with st.sidebar.container(border=True):
        
                channel_filter =st.multiselect(   " Channel:", ["Select All"]+data[data["Brand"]==brand_filter]['Channel'].unique().tolist(),default="Select All")
                if "Select All" in channel_filter:
                    channel_filter=data[data["Brand"]==brand_filter]['Channel'].unique().tolist()
            
            with st.sidebar.container(border=True):
        
                publisher_filter =st.multiselect(   " Publishername:", ["Select All"]+data[data["Brand"]==brand_filter]['Publishername'].unique().tolist(),default="Select All")
                if "Select All" in publisher_filter:
                    publisher_filter=data[data["Brand"]==brand_filter]['Publishername'].unique().tolist()
        elif Tactic_Type_filter=='UMM Tactic x Subbrand':
            with st.sidebar.container(border=True):
                tactic_list=data[data["Brand"]==brand_filter]["Actual Tactic"].unique().tolist()
                Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
                if "Select All" in Tactic_filter:
                    Tactic_filter=tactic_list
            with st.sidebar.container(border=True):
                
                subbrand_filter =st.multiselect(   " SubBrand:", ["Select All"]+data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist(),default="Select All")
                if "Select All" in subbrand_filter:
                    subbrand_filter=data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist() 
        elif Tactic_Type_filter=='EmPlanner x Subbrand':
            with st.sidebar.container(border=True):
                tactic_list=data[data["Brand"]==brand_filter]["Group Tactic"].unique().tolist()
                Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
                if "Select All" in Tactic_filter:
                    Tactic_filter=tactic_list
            with st.sidebar.container(border=True):
                
                subbrand_filter =st.multiselect(   " SubBrand:", ["Select All"]+data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist(),default="Select All")
                if "Select All" in subbrand_filter:
                    subbrand_filter=data[data["Brand"]==brand_filter]['Subbrand'].unique().tolist() 
            
    with st.sidebar.container(border=True):
        
        Period_filter =st.selectbox(   " Period Type:", ['Yearly' ,'Quarterly','Monthly']
    )
    
    with st.sidebar.container(border=True):
        
        Yr_filter =st.selectbox(   " Year list:", data['Yearly'].unique().tolist())
    
    with st.sidebar.container(border=True):
    
        # month_list=["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
        if Yr_filter=='2022':
            month_list=["Jan-22", "Feb-22", "Mar-22", "Apr-22", "May-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22", "Oct-22", "Nov-22", "Dec-22"]
        elif Yr_filter=='2023':
            month_list=["Jan-23", "Feb-23", "Mar-23", "Apr-23", "May-23", "Jun-23", "Jul-23", "Aug-23", "Sep-23", "Oct-23", "Nov-23", "Dec-23"]
        elif Yr_filter=='2024':
            month_list=["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
        else:
            month_list = ["Jan-25","Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"]
    
        Month_filter =st.multiselect(   " Month list:",["Select All"]+ month_list,default="Select All")
        # Tactic_filter =st.multiselect(   " Tactic Name:", ["Select All"]+tactic_list,default="Select All")
        if "Select All" in Month_filter:
            Month_filter=month_list
    
        # container = st.container()
        # all = st.checkbox("Select all")
         
        # if all:
        #     Tactic_filter = container.multiselect("Select one or more Tactic:",
        #         tactic_list,tactic_list)
        # else:
        #     Tactic_filter =  container.multiselect("Select one or more Tactic:",
        #        tactic_list)
    
    
    with st.sidebar.container(border=True):
        
        Type =st.selectbox(   " Budget Entry Type:", ['Total Spend','Spend Change','Percentage Change']
    
    )
        
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

if Tactic_Type_filter=='UMM Tactic Only':
    data = data[data["Actual Tactic"].isin(Tactic_filter)]
    data['Tactic']=data['Actual Tactic']
elif Tactic_Type_filter=='EmPlanner Tactic':
    data = data[data["Group Tactic"].isin(Tactic_filter)]
    data['Tactic']=data['Group Tactic']
elif Tactic_Type_filter=='EmPlanner x Subbrand':
    data = data[data["Group Tactic"].isin(Tactic_filter)]
    data['Tactic']=data['Group Tactic']
elif Tactic_Type_filter=='Actual':
    data = data[data["Subbrand"].isin(subbrand_filter)]
    data = data[data["Channel"].isin(channel_filter)]
    data = data[data["Publishername"].isin(publisher_filter)]

if 'Yr_filter' not in st.session_state:
    st.session_state.Yr_filter=Yr_filter

def all_filters(data):
    

    # st.write("CHECKKK")
    # st.write(data)
    # st.write(data.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                'Actual Profit' : 'sum'}))
    
    
    #st.write(Yr_filter)
  
    data=data[data["MONTH_YEAR"].dt.year == int(Yr_filter)]

    # st.write("CHECKKK")
    # st.write(data.groupby('Tactic').agg({'Actual Spend (Selected Months)' : 'sum'}))

    # st.write(data['Monthly'].unique())
    
    #data=data[data["MONTH_YEAR"].dt.year == 2024]
    #st.write(data)

    data2=st.session_state.snapshot_data
    #st.write("CHECKKKK")
    # st.dataframe(st.session_state.snapshot_data)
    
    # data2['FEC']
    # data2=data2[data2["MONTH_YEAR"].dt.year==Yr_filter]
    st.session_state.Yr_filter=Yr_filter

    data=data[data["Brand"]==brand_filter]
    data2=data2[data2["Brand"]==brand_filter]
    # st.session_state['Brand_sim_filter_data']=data[~data["Monthly"].isin(Month_filter)]
    st.session_state['Brand_sim_filter_data']=data2[~data2["Monthly"].isin(Month_filter)]


    if new_or_exi=='Existing Scenario':

        exi_sce=st.session_state.exi_sce

        st.session_state.agg_level_list=ast.literal_eval(exi_sce['agg_level'].iloc[0])
        st.session_state.type=exi_sce['type'].iloc[0]
        
        exi_sce = exi_sce.melt(id_vars= st.session_state.agg_level_list+[ 'PERIOD','Coef adjustment','CPM adjustment'], value_vars=[st.session_state.type], var_name='Measure', value_name='values')
        exi_sce = exi_sce.pivot(index= st.session_state.agg_level_list+['Coef adjustment','CPM adjustment'], columns='PERIOD', values='values').reset_index()
    

    else:
        if Tactic_Type_filter=="Actual":
            st.session_state.agg_level_list=['Brand', 'Subbrand','Channel','Publishername','Tactic']
        elif Tactic_Type_filter=='UMM Tactic x Subbrand':
            st.session_state.agg_level_list=['Brand', 'Subbrand','Tactic']    
        elif Tactic_Type_filter=='EmPlanner x Subbrand':
            st.session_state.agg_level_list=['Brand', 'Subbrand','Tactic']  
        else:
             st.session_state.agg_level_list=['Brand','Tactic']
    

        
    Annual_Actual_Data=data.groupby( st.session_state.agg_level_list).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index().rename(columns={'Actual Spend (Selected Months)': 'Actual Spend - Full Year'})
    st.session_state['Sim_Annual_data']= Annual_Actual_Data
    
    
    
    st.session_state.sim_filter_data = data2
    #data
    # [data2["Monthly"].isin(Month_filter)]
    # st.write("MONTH FILTER DATA")
    # st.dataframe(data)

    data = data[data["Monthly"].isin(Month_filter)]
    # Group by brand, tactic, year, quarter, month and sum budget_spend


    # st.write("ALL FILTER CHECK")
    # st.dataframe(data.head())
    # st.write(data.groupby('Tactic').agg({'Actual Spend (Selected Months)' : 'sum'}))
    
    sim_grouped_data= data.groupby( st.session_state.agg_level_list+[ Period_filter]).agg({'Actual Spend (Selected Months)': 'sum', 'MONTH_YEAR':'min'}).reset_index().rename(columns={Period_filter: 'PERIOD', 'MONTH_YEAR':'sort_order'}).sort_values( st.session_state.agg_level_list+['sort_order'])
    
    
    # sim_grouped_data=sim_grouped_data.rename(columns ={"BUDGET_SPEND":"Budget Spend"})
    sim_grouped_data['Sim_Per']=0.00
    sim_grouped_data['Delta']=0.00

    
    # sim_grouped_data['Spend_value']=0
    # adding existing data
    
    # exi_sce
    # st.write([x for x in exi_sce.columns if x not in  st.session_state.agg_level_list])
    if new_or_exi=='Existing Scenario':
        sim_grouped_data= sim_grouped_data.loc[:, ~sim_grouped_data.columns.isin([x for x in exi_sce.columns if x not in  st.session_state.agg_level_list])].merge(exi_sce,on=st.session_state.agg_level_list)
        sim_grouped_data['Simulation Spend (Selected Months)']=sim_grouped_data[st.session_state.ptr].sum(axis=1)
    else:
        sim_grouped_data['Simulation Spend (Selected Months)']=sim_grouped_data['Actual Spend (Selected Months)']
        # sim_grouped_data['Simulation Spend (Selected Months)']=sim_grouped_data[st.session_state.ptr].sum(axis=1)
    
    st.session_state.ptr=sim_grouped_data['PERIOD'].unique().tolist()


    st.session_state.grouped_byTactic=data.groupby( st.session_state.agg_level_list).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index()
    st.session_state.sim_grouped_byTactic_byPeriod=data.groupby( st.session_state.agg_level_list+[Period_filter]).agg({'Actual Spend (Selected Months)': 'sum'}).reset_index()
    # data = {'Brand': ['Afrin'], 'Tactic': ['Social'], Period_filter: ['Q2-24'], 'Budget Spend': [194221]}
    # new_row = pd.DataFrame(data)
    # st.session_state.sim_grouped_byTactic_byPeriod=pd.concat([st.session_state.sim_grouped_byTactic_byPeriod, new_row], ignore_index=True)
    st.session_state.ptr=sim_grouped_data['PERIOD'].unique().tolist()
    
    sim_grouped_data = sim_grouped_data.melt(id_vars= st.session_state.agg_level_list+[ 'PERIOD'], value_vars=[st.session_state.type], var_name='Measure', value_name='values')
    sim_grouped_data = sim_grouped_data.pivot(index= st.session_state.agg_level_list, columns='PERIOD', values='values').reset_index()
    sim_grouped_data=sim_grouped_data.merge(st.session_state.grouped_byTactic, on= st.session_state.agg_level_list)
    # sim_grouped_data['Simulation Spend (Selected Months)']=sim_grouped_data[st.session_state.ptr].sum(axis=1) #Not sure why i did this but keeping for now
    sim_grouped_data['Simulation Spend (Selected Months)']=sim_grouped_data['Actual Spend (Selected Months)']
    sim_grouped_data=sim_grouped_data.merge(st.session_state['Sim_Annual_data'], on= st.session_state.agg_level_list)
   
    # sim_grouped_data['Simulation Spend (Selected Months)']= sim_grouped_data['Simulation Spend (Selected Months)'].round(0)
    sim_grouped_data['Simulation Spend - Full Year']=sim_grouped_data['Actual Spend - Full Year']+(sim_grouped_data['Simulation Spend (Selected Months)']-sim_grouped_data['Actual Spend (Selected Months)'])
    sim_grouped_data[['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']]=sim_grouped_data[['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']].round(0)
    sim_grouped_data=sim_grouped_data[ st.session_state.agg_level_list+['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']+st.session_state.ptr]   
    
    sim_grouped_data['Coef adjustment']=1.00
    sim_grouped_data['CPM adjustment']=1.00
    st.session_state['sim_grouped_data'] = sim_grouped_data

    # st.write("SIM GROUPED DATA")
    # st.dataframe(sim_grouped_data)
    
# st.session_state['Sim_Annual_data'] 
# st.write(st.session_state['sim_grouped_data'])
if "apply_status" not in st.session_state:
    st.session_state.apply_status="FALSE"
# st.session_state['input_values_to_save']
if 'type'not in st.session_state:
    st.session_state.type='Simulation Spend (Selected Months)'


# if 'un_sim_grouped_data'not in st.session_state:
#     st.session_state.type='Simulation Spend (Selected Months)'

def unpivot_data(new_df: pd.DataFrame):

    # st.write("UNPIVOT ENTERRR")
    if new_df is not None:
            if new_df.equals(st.session_state.sim_grouped_data):
                if st.session_state.apply_status=="TRUE":
                    
                    
                    sim_grouped_data=st.session_state['sim_grouped_data'] 

                    un_sim_grouped_data=sim_grouped_data[st.session_state.agg_level_list+['Coef adjustment','CPM adjustment']+st.session_state.ptr].melt(id_vars=st.session_state.agg_level_list+['Coef adjustment','CPM adjustment'],
                                                                                               value_vars=st.session_state.ptr, var_name='PERIOD', value_name=st.session_state.type) 
                    
                    st.session_state['input_values_to_save'] =un_sim_grouped_data
                    
                    un_sim_grouped_data=un_sim_grouped_data.pivot_table(index=st.session_state.agg_level_list+['PERIOD','Coef adjustment','CPM adjustment'], aggfunc='sum', values=st.session_state.type).reset_index()
                    # st.session_state['test5'] =un_sim_grouped_data
                    un_sim_grouped_data=un_sim_grouped_data.merge(st.session_state.sim_grouped_byTactic_byPeriod, left_on=st.session_state.agg_level_list+['PERIOD'],right_on=st.session_state.agg_level_list+[Period_filter])
                           
                    if st.session_state.type=="Sim_Per":
                        un_sim_grouped_data['Simulation Spend (Selected Months)']=un_sim_grouped_data['Actual Spend (Selected Months)']*(un_sim_grouped_data['Sim_Per']/100+1)
                        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']-un_sim_grouped_data['Actual Spend (Selected Months)']
                        un_sim_grouped_data['Spend_value']=0 
                        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
                    elif st.session_state.type=="Delta":
                        
                        un_sim_grouped_data['Simulation Spend (Selected Months)']=un_sim_grouped_data['Actual Spend (Selected Months)']+un_sim_grouped_data['Delta']
                        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Delta']
                        un_sim_grouped_data['Sim_Per']=(un_sim_grouped_data['Delta']/un_sim_grouped_data['Actual Spend (Selected Months)'])*100
                        un_sim_grouped_data['Spend_value']=0   
                        
                        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Sim_Per','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
                    # un_sim_grouped_data=un_sim_grouped_data[['Brand', 'Tactic', 'PERIOD','Actual Spend (Selected Months)','Spend_Delta_value', st.session_state.type]]
                    elif st.session_state.type=='Simulation Spend (Selected Months)':
                        un_sim_grouped_data['Sim_Per']=((un_sim_grouped_data['Simulation Spend (Selected Months)']/un_sim_grouped_data['Actual Spend (Selected Months)'])-1)*100
                        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']-un_sim_grouped_data['Actual Spend (Selected Months)']
                        un_sim_grouped_data['Spend_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']
                
                        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Sim_Per','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
                    
                        
                    st.session_state['exceded_spend_limit']=np.any(np.where(un_sim_grouped_data['Sim_Per'] < -100, True, False))
                    if st.session_state['exceded_spend_limit']==True:
                        exceded_spend_limit_data=un_sim_grouped_data[un_sim_grouped_data['Sim_Per'] < -100]
                        exceded_spend_limit_data=exceded_spend_limit_data.rename(columns={'Actual Spend (Selected Months)':'Maximum Budget can decrease'})
                        exceded_spend_limit_data=exceded_spend_limit_data[['Tactic',Period_filter,'Maximum Budget can decrease']]
                        exceded_spend_limit_data['Maximum Budget can decrease']=exceded_spend_limit_data['Maximum Budget can decrease'].astype(int)
                        st.session_state['exceded_spend_limit_data']=exceded_spend_limit_data
                    st.session_state.updated_grouped_byTactic=un_sim_grouped_data.groupby(st.session_state.agg_level_list+['Coef adjustment','CPM adjustment']).agg({'Actual Spend (Selected Months)': 'sum','Simulation Spend (Selected Months)': 'sum'}).reset_index()
                    
                    un_sim_grouped_data=un_sim_grouped_data[col]
                    
                    
                    st.session_state['un_sim_grouped_data']=un_sim_grouped_data
                    #un_sim_grouped_data['Simulation Spend (Selected Months)']= un_sim_grouped_data['Simulation Spend (Selected Months)'].round(0)
                    st.session_state['test2'] =un_sim_grouped_data
                    sim_grouped_data = un_sim_grouped_data.melt(id_vars=st.session_state.agg_level_list+[ 'PERIOD'], value_vars=[st.session_state.type], var_name='Measure', value_name='values')
                    st.session_state['test3'] =sim_grouped_data
                    sim_grouped_data = sim_grouped_data.pivot(index=st.session_state.agg_level_list, columns='PERIOD', values='values').reset_index()
                    st.session_state['test4'] =sim_grouped_data
                    sim_grouped_data=sim_grouped_data.merge(st.session_state.updated_grouped_byTactic, on=st.session_state.agg_level_list)
                    sim_grouped_data=sim_grouped_data.merge(st.session_state['Sim_Annual_data'], on=st.session_state.agg_level_list)
                    
                    sim_grouped_data['Simulation Spend - Full Year']=sim_grouped_data['Actual Spend - Full Year']+(sim_grouped_data['Simulation Spend (Selected Months)']-sim_grouped_data['Actual Spend (Selected Months)'])
                    sim_grouped_data=sim_grouped_data[st.session_state.agg_level_list+['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']+st.session_state.ptr+['Coef adjustment','CPM adjustment']]         
                    st.session_state['sim_grouped_data'] = sim_grouped_data
                    
                    
                    st.session_state.apply_status="FALSE"


                
                return
    
            st.session_state['sim_grouped_data'] = new_df

    
    st.session_state.update_status="TRUE"
    sim_grouped_data=st.session_state['sim_grouped_data'] 
    
    un_sim_grouped_data=sim_grouped_data[st.session_state.agg_level_list+['Coef adjustment','CPM adjustment']+st.session_state.ptr].melt(id_vars=st.session_state.agg_level_list+['Coef adjustment','CPM adjustment'],
                                                                               value_vars=st.session_state.ptr, var_name='PERIOD', value_name=st.session_state.type) 
    
    st.session_state['input_values_to_save'] =un_sim_grouped_data
    
    un_sim_grouped_data=un_sim_grouped_data.pivot_table(index=st.session_state.agg_level_list+['PERIOD','Coef adjustment','CPM adjustment'], aggfunc='sum', values=st.session_state.type).reset_index()
    # st.session_state['test5'] =un_sim_grouped_data
    un_sim_grouped_data=un_sim_grouped_data.merge(st.session_state.sim_grouped_byTactic_byPeriod, left_on=st.session_state.agg_level_list+['PERIOD'],right_on=st.session_state.agg_level_list+[Period_filter])
           
    if st.session_state.type=="Sim_Per":
        un_sim_grouped_data['Simulation Spend (Selected Months)']=un_sim_grouped_data['Actual Spend (Selected Months)']*(un_sim_grouped_data['Sim_Per']/100+1)
        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']-un_sim_grouped_data['Actual Spend (Selected Months)']
        un_sim_grouped_data['Spend_value']=0 
        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
    elif st.session_state.type=="Delta":
        un_sim_grouped_data['Simulation Spend (Selected Months)']=un_sim_grouped_data['Actual Spend (Selected Months)']+un_sim_grouped_data['Delta']
        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Delta']
        un_sim_grouped_data['Sim_Per']=(un_sim_grouped_data['Delta']/un_sim_grouped_data['Actual Spend (Selected Months)'])*100
        un_sim_grouped_data['Spend_value']=0   
        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Sim_Per','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
    # un_sim_grouped_data=un_sim_grouped_data[['Brand', 'Tactic', 'PERIOD','Actual Spend (Selected Months)','Spend_Delta_value', st.session_state.type]]
    elif st.session_state.type=='Simulation Spend (Selected Months)':
        un_sim_grouped_data['Sim_Per']=((un_sim_grouped_data['Simulation Spend (Selected Months)']/un_sim_grouped_data['Actual Spend (Selected Months)'])-1)*100
        un_sim_grouped_data['Spend_Delta_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']-un_sim_grouped_data['Actual Spend (Selected Months)']
        un_sim_grouped_data['Spend_value']=un_sim_grouped_data['Simulation Spend (Selected Months)']

        col=st.session_state.agg_level_list+['PERIOD','Actual Spend (Selected Months)','Spend_Delta_value','Sim_Per','Spend_value', st.session_state.type,'Coef adjustment','CPM adjustment']
    
        
    st.session_state['exceded_spend_limit']=np.any(np.where(un_sim_grouped_data['Sim_Per'] < -100, True, False))
    if st.session_state['exceded_spend_limit']==True:
        exceded_spend_limit_data=un_sim_grouped_data[un_sim_grouped_data['Sim_Per'] < -100]
        exceded_spend_limit_data=exceded_spend_limit_data.rename(columns={'Actual Spend (Selected Months)':'Maximum Budget can decrease'})
        exceded_spend_limit_data=exceded_spend_limit_data[['Tactic',Period_filter,'Maximum Budget can decrease']]
        exceded_spend_limit_data['Maximum Budget can decrease']=exceded_spend_limit_data['Maximum Budget can decrease'].astype(int)
        st.session_state['exceded_spend_limit_data']=exceded_spend_limit_data
    st.session_state.updated_grouped_byTactic=un_sim_grouped_data.groupby(st.session_state.agg_level_list+['Coef adjustment','CPM adjustment']).agg({'Actual Spend (Selected Months)': 'sum','Simulation Spend (Selected Months)': 'sum'}).reset_index()
    
    un_sim_grouped_data=un_sim_grouped_data[col]
    
    
    st.session_state['un_sim_grouped_data']=un_sim_grouped_data
    #un_sim_grouped_data['Simulation Spend (Selected Months)']= un_sim_grouped_data['Simulation Spend (Selected Months)'].round(0)
    st.session_state['test2'] =un_sim_grouped_data
    sim_grouped_data = un_sim_grouped_data.melt(id_vars=st.session_state.agg_level_list+[ 'PERIOD'], value_vars=[st.session_state.type], var_name='Measure', value_name='values')
    st.session_state['test3'] =sim_grouped_data
    sim_grouped_data = sim_grouped_data.pivot(index=st.session_state.agg_level_list, columns='PERIOD', values='values').reset_index()
    st.session_state['test4'] =sim_grouped_data
    sim_grouped_data=sim_grouped_data.merge(st.session_state.updated_grouped_byTactic, on=st.session_state.agg_level_list)
    sim_grouped_data=sim_grouped_data.merge(st.session_state['Sim_Annual_data'], on=st.session_state.agg_level_list)
    
    sim_grouped_data['Simulation Spend - Full Year']=sim_grouped_data['Actual Spend - Full Year']+(sim_grouped_data['Simulation Spend (Selected Months)']-sim_grouped_data['Actual Spend (Selected Months)'])
    sim_grouped_data=sim_grouped_data[st.session_state.agg_level_list+['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']+st.session_state.ptr+['Coef adjustment','CPM adjustment']]         
    st.session_state['sim_grouped_data'] = sim_grouped_data

    # st.write("UNPIVOT")
    # st.dataframe(sim_grouped_data)

    if refresh_radio=="Manual Refresh":
      st.session_state.update_status="FALSE"

    st.rerun()

def profit_cal(un_sim_grouped_data):

    # st.write("PROFITT")
    # st.dataframe(un_sim_grouped_data)

    # st.write("CALCULATION")
    # st.dataframe(data)
    # st.dataframe(data.groupby('Actual Tactic').agg({'Actual Profit' : 'sum'}))

    # st.session_state.sim_filter_data
    un_sim_grouped_data['Simulation Spend (Selected Months)']=un_sim_grouped_data['Actual Spend (Selected Months)']+un_sim_grouped_data['Spend_Delta_value']
    if Tactic_Type_filter=="UMM Tactic x Subbrand" or Tactic_Type_filter=='EmPlanner x Subbrand':
        # un_sim_grouped_data
        # st.session_state.MTA_data
        un_sim_group_col=un_sim_grouped_data.columns
        # st.session_state.MTA_data
        un_sim_grouped_data=un_sim_grouped_data.merge(st.session_state.MTA_data, left_on=['Brand','Subbrand'],right_on=['BRAND','SUBBRAND'],how='left')
        # mean_for_null_value=st.session_state.MTA_data[st.session_state.MTA_data["BRAND"]==brand_filter]["MTA_ROI"].mean()
        # un_sim_grouped_data['MTA_ROI']=un_sim_grouped_data['MTA_ROI'].fillna(mean_for_null_value)

        un_sim_grouped_data['BRAND_MTA_ROI']=un_sim_grouped_data['BRAND_MTA_ROI'].fillna(un_sim_grouped_data['BRAND_MTA_ROI'].mean())
        # un_sim_grouped_data['NPD']=un_sim_grouped_data['NPD'].fillna('Null')
        
        un_sim_grouped_data['FINAL_MTA_ROI'] = np.where(un_sim_grouped_data['NPD']=='False',un_sim_grouped_data['MTA_ROI'],np.where(un_sim_grouped_data['NPD']=='True',un_sim_grouped_data['NPD_ASSUMPTION']*un_sim_grouped_data['BRAND_MTA_ROI'],un_sim_grouped_data['BRAND_MTA_ROI']))
        # un_sim_grouped_data
        un_sim_grouped_data['base_spend_per'] = un_sim_grouped_data["FINAL_MTA_ROI"]*(un_sim_grouped_data['Actual Spend (Selected Months)']/un_sim_grouped_data.groupby('Tactic')['Actual Spend (Selected Months)'].transform('sum'))
        un_sim_grouped_data['BASE_AVG_ROI']=un_sim_grouped_data.groupby(['Tactic'])['base_spend_per'].transform('sum')
        
        un_sim_grouped_data['Scenario_spend_per'] = un_sim_grouped_data["FINAL_MTA_ROI"]*(un_sim_grouped_data['Simulation Spend (Selected Months)']/un_sim_grouped_data.groupby('Tactic')['Simulation Spend (Selected Months)'].transform('sum'))
        un_sim_grouped_data['Scenario_AVG_ROI']=un_sim_grouped_data.groupby(['Tactic'])['Scenario_spend_per'].transform('sum')
        un_sim_grouped_data["Subbrand_Coef_adj"]=un_sim_grouped_data['Scenario_AVG_ROI']/un_sim_grouped_data["BASE_AVG_ROI"]
        # un_sim_grouped_data
        
        un_sim_grouped_data.groupby('Subbrand')['Scenario_spend_per'].sum()
        un_sim_grouped_data["Coef adjustment"]=un_sim_grouped_data["Coef adjustment"]*un_sim_grouped_data["Subbrand_Coef_adj"]
        un_sim_grouped_data=un_sim_grouped_data[un_sim_group_col]
        # un_sim_grouped_data
        
    
    # un_sim_grouped_data
    Cal_un_sim_grouped_data=un_sim_grouped_data.groupby(['Brand','Tactic','PERIOD']).agg({'Sim_Per':'sum','Spend_Delta_value':'sum','Spend_value':'sum','Coef adjustment':'mean','CPM adjustment':'mean'}).reset_index()
    # [['Brand','Tactic','PERIOD','Sim_Per','Spend_Delta_value','Spend_value','Coef adjustment','CPM adjustment']]
    Cal_un_sim_grouped_data['Spend_Delta_value']=round(Cal_un_sim_grouped_data['Spend_Delta_value'])
    if Tactic_Type_filter=="EmPlanner Tactic"  or Tactic_Type_filter=='EmPlanner x Subbrand':
        left_on_list=['Brand','Group Tactic',Period_filter]
        Cal_un_sim_grouped_data=Cal_un_sim_grouped_data.rename(columns={"Tactic":"Group Tactic"})
        right_on_list=['Brand','Group Tactic','PERIOD']
        input_colu_list=['Brand','Group Tactic','PERIOD','Sim_Per','Spend_Delta_value','Spend_value','Coef adjustment','CPM adjustment']
    else:
        left_on_list=['Brand','Tactic',Period_filter]
        right_on_list=['Brand','Tactic','PERIOD']
        input_colu_list=['Brand','Tactic','PERIOD','Sim_Per','Spend_Delta_value','Spend_value','Coef adjustment','CPM adjustment']

    
    # st.write('LEFT DF')
    # st.dataframe(st.session_state.sim_filter_data)


    output_df=st.session_state.sim_filter_data.merge(Cal_un_sim_grouped_data[input_colu_list], left_on=left_on_list, right_on=right_on_list)

    output_df['SELECTED_MONTHS'] = output_df["Monthly"].isin(Month_filter)
    # output_df=output_df[output_df['SELECTED_MONTHS']].reset_index(drop=True)
    output_df['Wk_Cnt']=output_df.groupby(['Brand','Tactic',Period_filter,'SELECTED_MONTHS']).transform('count').iloc[:, 0]
    
    # st.write("OUTPUT")
    # st.dataframe(output_df)

    output_df['CPM']=output_df['CPM_Original']+output_df['CPM_Original']*(st.session_state.inflation/100)
    output_df['Simulation Spend_WK']=(output_df['Spend_Delta_value']/output_df['Wk_Cnt'])+output_df['Actual Spend (Selected Months)']

    # output_df['Simulation Spend_WK'] = output_df['Spend_value']/output_df['Wk_Cnt']

    output_df.loc[output_df['SELECTED_MONTHS'] == False, 'Simulation Spend_WK']=output_df.loc[output_df['SELECTED_MONTHS'] == False, 'Actual Spend (Selected Months)']
    # output_df['Simulation Spend_WK'].fillna(0,inplace=True)
    
    negative_spend=output_df[output_df['Simulation Spend_WK']<0]
    
    # i=0
    # st.write(output_df.groupby(['Brand','Tactic',Period_filter])['Simulation Spend_WK'].sum())
    # negative_spend
    if len(negative_spend)>0:
        
        while len(negative_spend)!=0:
            
            # negative_spend
            # output_df
            # while len(negative_spend)==0:
            # negative_spend['Simulation Spend_WK']=abs(negative_spend['Simulation Spend_WK'])
            negative_spend=negative_spend.groupby(['Brand','Tactic',Period_filter]).agg({'Simulation Spend_WK':'sum','Wk_Cnt':'mean'}).reset_index()
            remaining_wk=output_df[output_df['Simulation Spend_WK']>0].groupby(['Brand','Tactic',Period_filter]).agg({'MONTH_YEAR':'count'}).reset_index()
            
            negative_spend=negative_spend.merge(remaining_wk, on=['Brand','Tactic',Period_filter],how='left')
            
            negative_spend['reamining_Wk_Cnt']=negative_spend['MONTH_YEAR']
            negative_spend['reamining_Sim_Spend']=negative_spend['Simulation Spend_WK']/negative_spend['reamining_Wk_Cnt']
            # negative_spend
            output_df['Simulation Spend_WK'] = np.where(output_df['Simulation Spend_WK'] < 0, 0, output_df['Simulation Spend_WK'])
            negative_spend=negative_spend[['Brand','Tactic',Period_filter,'reamining_Sim_Spend']]
            output_df= output_df.merge(negative_spend, on=['Brand','Tactic',Period_filter],how='left')
            output_df['reamining_Sim_Spend']=output_df['reamining_Sim_Spend'].fillna(0)
            output_df['Simulation Spend_WK']=np.where(output_df['Simulation Spend_WK'] == 0, 0,output_df['Simulation Spend_WK']+output_df['reamining_Sim_Spend'])
            
            # st.write("cal")
            # st.dataframe(output_df)

            # output_df=output_df[['Brand', 'Actual Tactic', 'BUDGET_WEEK_START_DT', 'C1', 'C2', 'C3', 'C4', 'CPM', 'Actual Spend (Selected Months)',
            #                      'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'ECOMM_ROI', 'TACTIC_ADJ_FCTR', 'SEASONAL_ADJ_FCTR', 'ADSTOCK_X', 'CURVE_TYPE', 
            #                      'ADSTOCK_WEEK', 'ADSTOCK', 'MONTH_YEAR', 'Group Tactic', 'FEC_FCTR','LTE_FCTR', 'Yearly', 'Quarterly', 'Monthly', 'Tactic',
            #                      'PERIOD', 'Sim_Per', 'Spend_Delta_value', 'Spend_value', 'Wk_Cnt', 'Simulation Spend_WK','Coef adjustment','CPM adjustment']]
            
            output_df=output_df[['Brand', 'Actual Tactic', 'BUDGET_WEEK_START_DT', 'C1', 'C2', 'C3', 'C4', 'CPM', 'Actual Spend (Selected Months)',
                        'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'ECOMM_ROI', 'TACTIC_ADJ_FCTR', 'SEASONAL_ADJ_FCTR', 'CURVE_TYPE', 
                        'ADSTOCK_WEEK', 'ADSTOCK', 'MONTH_YEAR', 'Group Tactic', 'FEC_FCTR','LTE_FCTR', 'Yearly', 'Quarterly', 'Monthly', 'Tactic',
                        'PERIOD', 'Sim_Per', 'Spend_Delta_value', 'Spend_value', 'Wk_Cnt', 'Simulation Spend_WK','Coef adjustment','CPM adjustment']]
            
            negative_spend=output_df[output_df['Simulation Spend_WK']<0]
            # negative_spend
            # i=i+1
            
            
            

    
    # negative_spend
    # output_df['FEC']
    # st.write(i)
    # st.write(output_df.groupby(['Brand','Tactic',Period_filter])['Simulation Spend_WK'].sum())
    
    
    
    remaining_months_data=st.session_state['Brand_sim_filter_data']
    remaining_months_data["PERIOD"]=st.session_state['Brand_sim_filter_data'][Period_filter]
    remaining_months_data[['ADSTOCK_X_Sim', 'ADSTOCK_X_Ana','Sim_Per','Spend_Delta_value','Spend_value','Wk_Cnt']]=0
    remaining_months_data['Simulation Spend_WK']=remaining_months_data['Actual Spend (Selected Months)']
    # remaining_months_data['Simulation Profit']=remaining_months_data['Actual Profit']
    # remaining_months_data["Simulation FEC"]=remaining_months_data['Actual FEC']
    remaining_months_data[['Coef adjustment','CPM adjustment']]=1
    remaining_months_data['SELECTED_MONTHS']=False
    remaining_months_data=remaining_months_data[output_df.columns]
    remaining_months_data=remaining_months_data[~remaining_months_data['Monthly'].isin(Month_filter)]
   
    
    # output_df['SELECTED_MONTHS']=True
    output_df['SELECTED_MONTHS'] = output_df["Monthly"].isin(Month_filter)

    # remaining_months_data['SELECTED_MONTHS']
    output_df=pd.concat([output_df,remaining_months_data],ignore_index=True) #removing for now- double check the logic
    
    output_df=output_df.sort_values(by=['Brand', 'Tactic', 'BUDGET_WEEK_START_DT'], ascending=[True, True, False])
    output_df=output_df.reset_index(drop=True)
 
    output_df["ADSTOCK_X_Sim"] = 0.0

    for key, group in output_df.groupby(["Brand", "Tactic", "CURVE_TYPE"]):
        idx = group.index
        n, max_wk = len(group), int(group["ADSTOCK_WEEK"].max())
        spend = group["Simulation Spend_WK"].to_numpy()
        cpm = group["CPM"].replace(0, np.nan).to_numpy()
        adstock = group["ADSTOCK"].to_numpy()
        safe_spend = np.nan_to_num(spend / cpm, nan=0.0, posinf=0.0, neginf=0.0)
        
        res, decay = np.zeros(n), np.ones(n)
        for _ in range(1, max_wk + 1):
            rolled = np.roll(safe_spend, -_) 
            if len(rolled) >_:
                rolled[-_] = 0
            res += rolled * decay * adstock
            decay *= adstock
        
        output_df.loc[idx, "ADSTOCK_X_Sim"] = res

    # output_df

    # st.write("TAG 1")
    # st.write(output_df.groupby(['Actual Tactic']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                    'Simulation Spend_WK' : 'sum'}))

    output_df['Simulation Profit']=[calculate_revenue(row["Simulation Spend_WK"], row["C1"], row["C2"], row["C3"], row["C4"], row["CURVE_TYPE"], row["CPM"]*row["CPM adjustment"], row["ADJUSTMENT_FACTOR"], row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"]*row["Coef adjustment"], row["SEASONAL_ADJ_FCTR"], row["ADSTOCK_X_Sim"], row["ECOMM_ROI"]) for i, row in output_df.iterrows()]

    # output_df['Simulation Profit'] =  output_df['Simulation Profit'] * 2.5
        
    # multipliers = {
    #     'Traditional': 28.0,   # To bring 0.03 into 0.80–0.87
    #     'Paid Social': 25.0,   # To bring 0.05 into 1.00–1.30
    #     'Paid Search': 0.85    # To bring 1.11 below 1.00
    # }
    
    # output_df['Simulation Profit'] = output_df.apply(
    #     lambda row: round(row['Simulation Profit'] * multipliers[row['Actual Tactic']], 2),
    #     axis=1
    # )

    
    # st.write("AFTER SIM PROFIT")
    # st.write(output_df.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                              'Actual Profit' : 'sum',
    #                                                              'Simulation Profit' : 'sum'}))

    # st.write("DATA CHECK")
    # st.dataframe(output_df[output_df['Tactic'] == 'Paid Search'][['Tactic', 'Actual Spend (Selected Months)', 'CPM', 'Impression', 'CURVE_TYPE', 'C1', 'C2', 'C3', 'C4', 'Actual Profit', 'Simulation Profit']])
    
    output_df["ADSTOCK_X_Ana"] = 0.0

    for key, group in output_df.groupby(["Brand", "Tactic", "CURVE_TYPE"]):
        idx = group.index
        n, max_wk = len(group), int(group["ADSTOCK_WEEK"].max())
        spend = group["Actual Spend (Selected Months)"].to_numpy()
        cpm = group["CPM"].replace(0, np.nan).to_numpy()
        adstock = group["ADSTOCK"].to_numpy()
        safe_spend = np.nan_to_num(spend / cpm, nan=0.0, posinf=0.0, neginf=0.0)
        
        res, decay = np.zeros(n), np.ones(n)
        for _ in range(1, max_wk + 1):
            rolled = np.roll(safe_spend, -_) 
            if len(rolled) >_:
                rolled[-_] = 0
            res += rolled * decay * adstock
            decay *= adstock
        
        output_df.loc[idx, "ADSTOCK_X_Ana"] = res

    output_df.loc[:, "Cal_X0"] = 0.0

    for (brand, tactic), group in output_df.groupby(["Brand", "Tactic"]):
        max_week = int(group["ADSTOCK_WEEK"].max())
    
        # for i in range(1, max_week + 1):
        group["Cal_X0"] += (
            ((group['Actual Spend (Selected Months)'] / group["CPM"]).fillna(0))
           
            )
    
        # Update the 'ADSTOCK_X' column in the original dataframe for this group
        output_df.loc[group.index, "Cal_X0"] = group["Cal_X0"] 

    ########################### 
    # output_df["ADSTOCK_X_Ana"]=output_df["ADSTOCK_X"]
    ###########################
    
    # st.write("TEST-1")
    # st.write(output_df.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                              'Actual Profit' : 'sum',
    #                                                              'Simulation Profit' : 'sum'}))
    
    # output_df['Actual Profit']=[calculate_revenue(row['Actual Spend (Selected Months)'], row["C1"], row["C2"], row["C3"], row["C4"], 
    #                                               row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], 
    #                                               row["SEASONAL_ADJ_FCTR"], row["ADSTOCK_X_Ana"], row["ECOMM_ROI"]) for i, row in output_df.iterrows()]
    
    # output_df['Actual Profit'] = output_df['Actual Profit'] * 2.5
    # output_df['Actual Profit'] = output_df.apply(
    #     lambda row: round(row['Actual Profit'] * multipliers[row['Actual Tactic']], 2),
    #     axis=1
    # )

    #output_df['Actual Profit'] =   output_df['Actual Profit'] / 20 

    # st.write("TEST-2")
    # st.write(output_df.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                'Actual Profit' : 'sum'}))
    
    # output_df=output_df[output_df['Simulation Spend_WK']>1]
    # output_df["BUDGET_WEEK_START_DT"]=output_df["BUDGET_WEEK_START_DT"].dt.strftime('%d-%m-%y')

    output_df['Actual Profit']=[calculate_revenue(row["Actual Spend (Selected Months)"], row["C1"], row["C2"], row["C3"], row["C4"], 
                                                  row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"]*row["Coef adjustment"], 
                                                  row["SEASONAL_ADJ_FCTR"], row['ADSTOCK_X_Ana'], row["ECOMM_ROI"]) for i, row in output_df.iterrows()]

    # output_df['Actual Profit']=[calculate_revenue(row['Actual Spend (Selected Months)'], row["C1"], row["C2"], row["C3"], row["C4"], 
    #                                               row["CURVE_TYPE"], row["CPM"], row["ADJUSTMENT_FACTOR"], row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], 
    #                                               row["SEASONAL_ADJ_FCTR"], row["ADSTOCK_X_Ana"], row["ECOMM_ROI"]) for i, row in output_df.iterrows()]

    # st.write("INSIDE PROFIT CAL")
    # st.write(output_df.groupby("Tactic").agg({'Actual Profit': 'sum',
    #                                           'Simulation Profit': 'sum',
    #                                           'Actual Spend (Selected Months)' : 'sum',
    #                                           'Simulation Spend_WK' : 'sum'}))
    # st.write(output_df)

    output_df[['Actual Spend (Selected Months)','Simulation Spend_WK']]=output_df[['Actual Spend (Selected Months)','Simulation Spend_WK']].round(2)


    output_df["Actual FEC"]=output_df['Actual Profit']*output_df['FEC_FCTR']
    output_df["Simulation FEC"]=output_df['Simulation Profit']*output_df['FEC_FCTR']
    
    output_df["Actual LTE Profit"]=output_df["Actual Profit"]*output_df["LTE_FCTR"]
    output_df["Actual LTE FEC"]=output_df["Actual LTE Profit"]*output_df["FEC_FCTR"]
    output_df["Simulation LTE Profit"]=output_df["Simulation Profit"]*output_df["LTE_FCTR"]
    output_df["Simulation LTE FEC"]=output_df["Simulation LTE Profit"]*output_df["FEC_FCTR"]

    # st.write("PRE YEAR PROFT")
    # st.write(output_df.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
    #                                                'Actual Profit' : 'sum'}))
    # output_df['cal_x0']=output_df['Actual Spend (Selected Months)'] / output_df["CPM"]
    # output_df[output_df['Tactic'].isin(['Audio'])]
    
     
    
    # output_df=output_df[output_df["Monthly"].isin(Month_filter)]
    
    
    ######################
    # test=output_df[["Tactic","BUDGET_WEEK_START_DT",'Actual Spend (Selected Months)',"CPM","Actual Profit","FEC","Actual FEC","FEC_FCTR",'Cal_X0',"ADSTOCK_X","ADSTOCK_X_Ana"]]
    # test['FEC_Diff']=((test['Actual FEC']/test['FEC'])-1)
    # test['FEC_Diff']=test['FEC_Diff']
    # # test=test[test['Tactic']=='Audio']
    # # test
    # t2=output_df[['Tactic','BUDGET_WEEK_START_DT',"ADSTOCK_X_Ana","ADSTOCK_X"]]
    # t2['Diff']=t2['ADSTOCK_X']-t2['ADSTOCK_X_Ana']
    # t2
#     calculate_revenue(
#     x, c1, c2, c3, c4, curve_type, CPM, adjustment_factor, ref_adj_fctr, tactic_adj_fctr, seasonal_adj_fctr, ADSTOCK_X, ECOMM_ROI
# )
    # st.write(calculate_revenue(30866.68,8.013804476,1.602576056,0,0,'Power',12000,1,4, 0.67,1,1.102326773,0 ) )
    # st.session_state.Yearly_view=True

    if st.session_state.Yearly_view==True:
        # grp_output_df = output_df[output_df["Yearly"].isin(['2023']) ].groupby(['Brand', 
        #                                                                         'Tactic']).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit': 'sum', 
        #                                                                                         'Simulation Spend_WK': 'sum', 'Simulation Profit' : 'sum',
        #                                                                                         'Actual FEC' : 'sum', 'Simulation FEC' : 'sum', 
        #                                                                                         'Actual LTE FEC' : 'sum', "Simulation LTE FEC" : 'sum'}).reset_index()
        
        # st.write("2023")
        # st.dataframe(grp_output_df)

        # st.write("PROFIT CALC YEAR")
        # st.write(output_df.groupby(['Actual Tactic', 'Yearly']).agg({'Actual Spend (Selected Months)' : 'sum',
        #                                            'Actual Profit' : 'sum'}))

        grp_output_df = output_df[output_df["Yearly"].isin(['2025']) ].groupby(['Brand', 
                                                                                'Tactic']).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit': 'sum', 
                                                                                                'Simulation Spend_WK': 'sum', 'Simulation Profit' : 'sum',
                                                                                                'Actual FEC' : 'sum', 'Simulation FEC' : 'sum', 
                                                                                                'Actual LTE FEC' : 'sum', "Simulation LTE FEC" : 'sum'}).reset_index()
        # st.write("POST AGG PROFIT CALC")
        # st.write(grp_output_df.head())

        #output_df=output_df[output_df["Monthly"].isin(["Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"])]
        output_df = output_df[output_df["Monthly"].isin(Month_filter)]

        # st.write("YEARLYY")
        # st.write(output_df)
        
        output_df=output_df[output_df['Wk_Cnt']>0]
        output_df.reset_index(drop=True)
        
    else:
        # output_df=output_df[output_df['SELECTED_MONTHS']==True]
        # output_df[output_df['Monthly'].str[:3].isin([m[:3] for m in Month_filter]) &(output_df["Yearly"].isin(['2024'])) ]
        grp_output_df = output_df[output_df['Monthly'].str[:3].isin([m[:3] for m in Month_filter]) &(output_df["Yearly"].isin(['2025'])) ].groupby(['Brand', 'Tactic']).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit': 'sum', 'Simulation Spend_WK': 'sum', 'Simulation Profit': 'sum','Actual FEC':'sum','Simulation FEC':'sum', 'Actual LTE FEC':'sum',"Simulation LTE FEC":'sum'}).reset_index()
        output_df=output_df[output_df["Monthly"].isin(Month_filter)]
    # output_df[output_df['Quarterly']=='Q1-24']
    # Calculating the grand total
    grand_total = grp_output_df[['Actual Spend (Selected Months)', 'Actual Profit', 'Simulation Spend_WK', 'Simulation Profit','Actual FEC',
                                 'Simulation FEC',"Actual LTE FEC","Simulation LTE FEC"]].sum()
    
    # st.write("GRAND")
    # st.dataframe(grand_total)
    
    # Creating a DataFrame for the grand total row
    _2024_grand_total_row = pd.DataFrame({'Tactic': ['Grand_Total_2024'], 
                                    'Actual Spend (Selected Months)': [grand_total['Actual Spend (Selected Months)']], 
                                    'Actual Profit': [grand_total['Actual Profit']], 
                                    'Simulation Spend_WK': [grand_total['Simulation Spend_WK']], 
                                    'Simulation Profit': [grand_total['Simulation Profit']],
                                    'Actual FEC': [grand_total['Actual FEC']],
                                    'Simulation FEC':[grand_total['Simulation FEC']],
                                    "Actual LTE FEC": [grand_total["Actual LTE FEC"]],
                                    "Simulation LTE FEC":[grand_total["Simulation LTE FEC"]]
                                   })
    
    # st.write("BEFORE")
    # st.dataframe(output_df.shape)

    #output_df=output_df[output_df["Yearly"].astype(int).isin([Yr_filter])]
    output_df = output_df[output_df['Yearly'] == (Yr_filter)]

    # st.write("CAL")
    # st.dataframe(output_df)
    
    st.session_state['output_df']=output_df
    grp_output_df = output_df.groupby(['Brand', 'Tactic']).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit': 'sum', 'Simulation Spend_WK': 'sum', 'Simulation Profit': 'sum','Actual FEC':'sum','Simulation FEC':'sum', 'Actual LTE FEC':'sum',"Simulation LTE FEC":'sum'}).reset_index()
    
    # st.write("GRP OUT DF")
    # st.dataframe(grp_output_df)

    # Calculating the grand total
    grand_total = grp_output_df[['Actual Spend (Selected Months)', 'Actual Profit', 'Simulation Spend_WK', 'Simulation Profit','Actual FEC','Simulation FEC',"Actual LTE FEC","Simulation LTE FEC"]].sum()
    
    # Creating a DataFrame for the grand total row
    grand_total_row = pd.DataFrame({'Tactic': ['Grand Total'], 
                                    'Actual Spend (Selected Months)': [grand_total['Actual Spend (Selected Months)']], 
                                    'Actual Profit': [grand_total['Actual Profit']], 
                                    'Simulation Spend_WK': [grand_total['Simulation Spend_WK']], 
                                    'Simulation Profit': [grand_total['Simulation Profit']],
                                    'Actual FEC': [grand_total['Actual FEC']],
                                    'Simulation FEC':[grand_total['Simulation FEC']],
                                    "Actual LTE FEC": [grand_total["Actual LTE FEC"]],
                                    "Simulation LTE FEC":[grand_total["Simulation LTE FEC"]]
                                                                                                                                                                
                                   
                                   })
    
    # Appending the grand total row to grp_output_df
    # grp_output_df = pd.concat([grp_output_df, grand_total_row,_2024_grand_total_row], ignore_index=True)

    grp_output_df = pd.concat([grp_output_df, grand_total_row], ignore_index=True)

    grp_output_df['Spend_DIFF'] = ((grp_output_df['Simulation Spend_WK'] / grp_output_df['Actual Spend (Selected Months)'])-1)
    grp_output_df['Profit_DIFF'] = (grp_output_df['Simulation Profit'] / grp_output_df['Actual Profit'])-1
    grp_output_df['Original ROI']=(grp_output_df['Actual Profit']/grp_output_df['Actual Spend (Selected Months)']).round(2)
    grp_output_df['Simulation ROI']=(grp_output_df['Simulation Profit']/grp_output_df['Simulation Spend_WK']).round(2)

    # st.write("Original ROI", grp_output_df)

    # grp_output_df.rename(columns={'Budget Spend': 'Original Spend',})
    # grp_output_df["Actual FEC"]=grp_output_df['Actual Profit']*grp_output_df['FEC_FCTR']
    # grp_output_df["Simulation FEC"]=grp_output_df['Simulation Profit']*grp_output_df['FEC_FCTR']
    #need to update this later
    grp_output_df['FEC_Diff']=(grp_output_df['Simulation FEC']-grp_output_df['Actual FEC'])
    
    grp_output_df['Spend_Diff_value']=(grp_output_df['Simulation Spend_WK'] - grp_output_df['Actual Spend (Selected Months)'])
    grp_output_df['Classification'] = np.where(grp_output_df['FEC_Diff'] > 0, 'P', 'N')
    grp_output_df['Spend_Classification'] = np.where(grp_output_df['Spend_Diff_value'] > 0, 'P', 'N')
    grp_output_df=grp_output_df.sort_values(['Classification','FEC_Diff'], ascending=False)

    # st.write("GRP OUTPUTTT")
    # st.write(grp_output_df)

    
    FECdelta_grp=grp_output_df[['Tactic','FEC_Diff','Classification']].rename(columns={'FEC_Diff':'Measure'})
    FECdelta_grp['Flag']='FEC'
    
    Spenddelta_grp=grp_output_df[['Tactic','Spend_Diff_value','Spend_Classification']].rename(columns={'Spend_Diff_value':'Measure','Spend_Classification':'Classification'})
    Spenddelta_grp['Flag']='SPEND'
    # Delta=pd.concat([FECdelta_grp,Spenddelta_grp])
    # Delta=Delta.sort_values(['Classification','Measure'], ascending=False)
   
       # pd.merge()
    
    
    st.session_state['grp_output_df']=grp_output_df


    
    
    grp_output_diff = grp_output_df[['Spend_DIFF','Profit_DIFF']].copy()
    actual_grp_output_df=grp_output_df[['Brand', 'Tactic','Actual Spend (Selected Months)','Actual Profit',
                                        'Original ROI','Actual FEC','Actual LTE FEC']].rename(columns={'Actual Spend (Selected Months)': 'SPEND',
                                                                                                       'Actual Profit':'PROFIT', 
                                                                                                       'Original ROI': 'ROI',
                                                                                                       'Actual FEC':'FEC',
                                                                                                       'Actual LTE FEC':'LTE FEC'})
    actual_grp_output_df['Flag']='Actual'


    sim_grp_output_df=grp_output_df[['Brand', 'Tactic','Simulation Spend_WK','Simulation Profit',
                                     'Simulation ROI',"Simulation FEC",'Simulation LTE FEC']].rename(columns={'Simulation Spend_WK': 'SPEND',
                                                                                                              'Simulation Profit':'PROFIT', 
                                                                                                              'Simulation ROI': 'ROI',
                                                                                                              "Simulation FEC":'FEC',
                                                                                                              'Simulation LTE FEC':'LTE FEC'})
    sim_grp_output_df['Flag']='Simulation'

    # st.write("SIM GRP")
    # st.dataframe(sim_grp_output_df)

    # st.write("ACTUAL GRP")
    # st.dataframe(actual_grp_output_df)


    final_grp_output_df=pd.concat([sim_grp_output_df,actual_grp_output_df])
    # final_grp_output_df['FEC']=final_grp_output_df['PROFIT']
    

    final_grp_output_df=final_grp_output_df.melt(id_vars=['Brand', 'Tactic', 'Flag'], 
                                     value_vars=['SPEND', 'ROI',"FEC",'LTE FEC'], 
                                     var_name='Measure', 
                                     value_name='Value')
    # final_grp_output_df['Value'].astype((int))
    # final_grp_output_df=final_grp_output_df.sort_values(['Flag','Value'], ascending=True)


    final_grp_output_df['Measure']=pd.Categorical(final_grp_output_df['Measure'], categories=['SPEND', 'ROI',"FEC",'LTE FEC'], ordered=True)
    final_grp_output_df=final_grp_output_df.sort_values(by=['Measure','Value'])

    for_order=final_grp_output_df[final_grp_output_df['Measure']=="SPEND"]
    # for_order['Rank'] = for_order['Value'].rank(method='min')
    
    
    # for_order['Flag']=pd.Categorical(for_order['Flag'], categories=['Orignal', 'Simulation'], ordered=True)
    for_order=for_order[for_order['Flag']=='Simulation']
    for_order=for_order.sort_values(by=['Value'])

    for_order.reset_index(drop=True, inplace=True)
    for_order['row_numbers'] = for_order.index + 1 
    for_order=for_order[['Brand','Tactic','row_numbers']]

    # st.write("FOR ORDER")
    # st.write(for_order)

    final_grp_output_df=final_grp_output_df.merge(for_order, on=['Brand','Tactic'])
    final_grp_output_df['Value']=np.where(final_grp_output_df['Measure'] =='SPEND', final_grp_output_df['Value'].fillna(0).astype(int), final_grp_output_df['Value'])
    final_grp_output_df['Value']=np.where(final_grp_output_df['Measure'] =='FEC', final_grp_output_df['Value'].fillna(0).astype(int), final_grp_output_df['Value'])
    
    st.session_state['final_grp_output_df']=final_grp_output_df

    st.session_state['Delta']=pd.concat([FECdelta_grp,Spenddelta_grp]).merge(for_order, on='Tactic').sort_values(by='row_numbers')

    # st.write("FINAL-1")
    # st.dataframe(final_grp_output_df[final_grp_output_df['Tactic'] == 'Paid Search'])
        
    # st.write("FINAL-2")
    # st.dataframe(st.session_state.Delta)
    

def type_update(Type):
    if Type=="Percentage Change":
        st.session_state.type="Sim_Per"
    elif Type=="Total Spend":
        st.session_state.type='Simulation Spend (Selected Months)'
    elif Type=="Spend Change":
        st.session_state.type="Delta"
        

if Apply_button:
    # Group by brand, tactic, year, quarter, month and sum Budget Spend
    st.session_state.update_status="TRUE"
    st.session_state.apply_status="TRUE"
    st.session_state['exceded_spend_limit']=False
    
    type_update(Type)
    
    all_filters(data)
    
    
    st.session_state.img=brand_logo()

if 'un_sim_grouped_data'not in st.session_state:
    all_filters(data)


    
if 'img' not in st.session_state:
        st.session_state.img=brand_logo()
   
    
if "sim_grouped_data" not in st.session_state:
    
    type_update(Type)
    all_filters(data)
    st.session_state.update_status="FALSE"
    # st.session_state.sim_filter_data = data[data["Monthly"].isin(Month_filter)]
    # Tactic_filter=tactic_list
    
    # add_Simulator_budget(st.session_state['sim_grouped_data'])


col1, col2,col3 = st.columns([0.04,0.91,0.04], gap="small")
with col1:
    st.image(
            "https://www.c5i.ai/wp-content/themes/course5iTheme/new-assets/images/c5i-primary-logo.svg",
            width=60, # Manually Adjust the width of the image as per requirement
        )
with col2:
    st.header('Sonic Simulation Tool',divider=True)
with col3:
    st.image(st.session_state.img,width=100,)


with st.container(border=True, height=650):

    
        
    col1, col2,col3,col4 = st.columns([0.30,0.15,0.15,0.1])

    # with col1:
    #     st.markdown("Auto Refresh")
    with col1:
        refresh_radio = st.radio(r"$\textrm{\large Choose the refresh mode}$",options = [ "Manual Refresh","Auto Refresh"],horizontal=True)
        
        # cmd_on=st.toggle("Switch to Manual Refresh"
        #                 )
    # with col2:
        
    
            
      

    
    with col3:
        # download_template=st.button("Download Templete", use_container_width=True)
        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")
        
        csv = convert_df(st.session_state['sim_grouped_data'])
        
        st.download_button(
            label="Download Templete",
            data=csv,
            file_name=str(brand_filter)+"_"+Period_filter+"_"+Tactic_Type_filter+"_"+str(datetime.now().strftime("%d_%m_%y %H_%M"))+".csv",
            mime="text/csv",
            type="secondary"
            ,use_container_width=True
        )
    with col4:
        reset_button=st.button("Reset", type="primary",use_container_width=True)
    
    if reset_button:
        all_filters(data)
        st.session_state.update_status="TRUE"
        st.session_state['exceded_spend_limit']=False
    # unpivot_data(editable_df) 

    #st.write("checkkk")
    c1={ col: st.column_config.NumberColumn(step=1,) for col in ['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']} #,format="₩ %d"
    # c2={col:st.column_config.NumberColumn(format='%.2f %%') for col in st.session_state.ptr}
    # if Apply_button:
    if st.session_state.type=="Sim_Per":
        c2={col:st.column_config.NumberColumn(step=0.001,format='%.2f %%') for col in st.session_state.ptr}

    elif st.session_state.type=="Simulation Spend (Selected Months)":
        c2={col:st.column_config.NumberColumn(min_value=0,step=1) for col in st.session_state.ptr}

    else:
        c2={col:st.column_config.NumberColumn(step=1) for col in st.session_state.ptr}

    color_code = '#f0f0f0' 
    header_style = {
    'selector': 'th',  # Target the header (th element)
    'props': [('background-color', '{color_code}'), ('color', 'white'), ('font-weight', 'bold')]
    }
    
    # Function to apply background color to all columns (using custom color)
    def highlight_all_columns(x):
        return [f'background-color: {color_code}' for _ in range(len(x))]  # Apply color code to all rows in a column

    # Apply the styling to all columns
    styled_df  = st.session_state['sim_grouped_data'].style.apply(highlight_all_columns, axis=0)
    currency_format = {col: '₩ {:,.0f}' for col in ['Actual Spend - Full Year','Simulation Spend - Full Year','Actual Spend (Selected Months)','Simulation Spend (Selected Months)']}

    # Apply the dynamic currency format to the specified columns
    
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3], gap="small")

    if Type=='Total Spend':
        with col3:
            st.markdown(f"$\\textrm{{\\small \\textbf{{{Type}}}: Simulate the data on total spent.}}$")

    elif Type=='Spend Change':
        with col3:
            st.markdown(f"$\\textrm{{\\small \\textbf{{{Type}}}: Simulate the data on certain spend change.}}$")

    else:
        with col3:
            st.markdown(f"$\\textrm{{\\small \\textbf{{{Type}}}: Simulate the data on certain percent change.}}$")

    styled_df = styled_df.format(currency_format)
    editable_df = st.data_editor(styled_df 
                                 ,height=550
                                 ,column_config={**c1, **c2}
                                 ,use_container_width=True,
                              disabled=["Brand", "Tactic",'Actual Spend - Full Year','Subbrand','Channel','Publishername','Simulation Spend - Full Year',
                                        'Actual Spend (Selected Months)','Simulation Spend (Selected Months)'],
                              hide_index=True,
                                                  )

iconname = "fas fa-asterisk"
lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'

def cards_html(header_input, input1, input2, input3,input4, wch_colour_box, wch_colour_font, fontsize1, fontsize2, iconname,Actual_2024,analplan_2024_diff,analplan_2024_diff_value):
    return f"""<style>
                .card-text {{
                    font-size: {fontsize2}px;
                    color: '#212121';
                    line-height: 19px;
                }}
                .bold-text {{
                    font-size: {fontsize1}px;
                    font-weight: bold;
                }}
                .underlined-text {{
                    text-decoration: underline;
                    font-size: {fontsize1 + 2}px; /* Increase font size by 2px */
                }}
                </style>
                <div style='background:{wch_colour_box}; 
                            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.90); 
                            font-size: {fontsize2}px; 
                            border-radius: 7px; 
                            padding-left: 12px; 
                            padding-top: 2px; 
                            padding-bottom: 2px; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>
                    <p class="card-text">
                        <span class="underlined-text bold-text" style='text-align: center; display: block;'>{header_input}</span>
                        <br><br>
                        <span class="bold-text">Yearly      : {input2} </span>
                        <span style='font-size: {fontsize2}px; padding-right: 12px;padding-top: 25px; float: right;'> Delta : {input4} (<span class="bold-text">{input3}</span>)</span>
                        <br><br>
                        <span class="bold-text">Simulation : {input1}</span>
                    </p>
                </div>"""

def fec_cards_html(header_input, input1, input2, input3,input4, wch_colour_box, wch_colour_font, fontsize1, fontsize2, iconname,Actual_2024,analplan_2024_diff,analplan_2024_diff_value):
    return f"""<style>
                .card-text {{
                    font-size: {fontsize2}px;
                    color: '#212121';
                    line-height: 19px;
                }}
                .bold-text {{
                    font-size: {fontsize1}px;
                    font-weight: bold;
                }}
                .underlined-text {{
                    text-decoration: underline;
                    font-size: {fontsize1 + 2}px; /* Increase font size by 2px */
                }}
                </style>
                <div style='background:{wch_colour_box}; 
                            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.90); 
                            font-size: {fontsize2}px; 
                            border-radius: 7px; 
                            padding-left: 12px; 
                            padding-top: 2px; 
                            padding-bottom: 2px; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>
                    <p class="card-text">
                        <span class="underlined-text bold-text" style='text-align: center; display: block;'>{header_input}</span>
                        <span class="bold-text"></span>
                        <br><br>
                        <span class="bold-text">Yearly      : {input2} </span>
                        <span style='font-size: {fontsize2}px; padding-right: 12px;padding-top: 25px; float: right;'> Delta : {input4} (<span class="bold-text">{input3}</span>)</span>
                        <br><br>
                        <span class="bold-text">Simulation : {input1}</span>
                    </p>
                </div>"""


# css = '''
# <style>
#     .stButton [data-testid="baseButton-secondary"] p {
#         # font-size: 1.5rem;
#         background-color: rgb(0, 104, 201);
#         color:rgb(255, 255, 255);
        
#     }
# </style>
# '''
# editable_df
# st.markdown(css, unsafe_allow_html=True)
# method_type=st.selectbox("",["Method 1","Method 2"])

if refresh_radio=="Manual Refresh":
    update_sim_spend=st.button(r"$\textrm{\large Update Simulation Spend}$")
if refresh_radio=="Manual Refresh":
    if update_sim_spend:
        unpivot_data(editable_df)
else:
    unpivot_data(editable_df)


       
if 'exceded_spend_limit' not in st.session_state:
    st.session_state['exceded_spend_limit']=False


c1,c2=st.columns([0.3,0.7])
with c1:
    
    # if st.session_state.Yr_filter==2025:
        
    #     with st.container(border=True):
    #         st.session_state.inflation=st.number_input('Inflation %',step=1.00,format='%0.2f')
    #         st.write(st.session_state.inflation)

    # else:
    #     st.session_state.inflation=0
    st.session_state.inflation=0
    pass

if st.session_state['exceded_spend_limit']==True:
    st.warning('SIMULATION SPEND HAS BEEN REDUCED MORE THAN Actual SPEND - Edit Simulation Spend to Start Simulation', icon="⚠️")
    with st.container():

        st.dataframe(st.session_state['exceded_spend_limit_data'],hide_index=True)
else:
    prfot_button=st.button(r"$\textrm{\large Start Simulation}$" )
    
    if prfot_button:
                with st.spinner(text="Calculating ..."):
                    st.session_state.apply_status="TRUE"
                    
                    unpivot_data(editable_df)  
                    #mulitply inflation
                    
                    profit_cal(st.session_state['un_sim_grouped_data'])
                    
                    st.session_state.update_status="FALSE"
                    st.session_state.save_enable=False

    if "grp_output_df" not in st.session_state:
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
        st.session_state.Yearly_view=False

    elif st.session_state.update_status=="TRUE":
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")

    elif "un_sim_grouped_data" not in st.session_state:
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")

    else:
    
        
        # selected = option_menu("", ["Comparison Chart","FEC Delta Chart", "Detail Report"],icons=['bar-chart', 'graph-up-arrow','database fill'], menu_icon="cast"
        # ,orientation="horizontal"
        # )
        # @st.experimental_dialog("Enter the scenario name to save",width="large")
        # def save_fun():
        if 'save_enable' not in st.session_state:
            st.session_state.save_enable=False

        # if st.button("Save this Scenario"):
        #     with st.container(border=True):
        #         st.session_state.save_enable=True

        st.session_state.save_enable=False

        with st.container(border=True):
            if st.session_state.save_enable==True:  
                c1,c2,c3=st.columns(3)
                with c1:
                    Scenario_name=st.text_input("Enter the Scenario name to save")
                with c2:
                    Description=st.text_input("Enter the Description to save")
                save_bt=st.button("Save")
                if save_bt:
                    if Scenario_name=="":
                        st.warning("Please fill the name")
                    else:   
                        with st.spinner("Please Don't close this tab/window - its Saving...."):
                            from snowflake.snowpark.exceptions import SnowparkSQLException
                            try:
                                save_data=st.session_state['output_df']
                                save_data["CURVE_TYPE"]=save_data["CURVE_TYPE"].astype(str)
                                save_data['COMMENTS']=Scenario_name
                                save_data['DATE']=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                JOB_ID=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                                save_data['JOB_ID']=JOB_ID
                                save_data['USER_EMAIL']=st.experimental_user.email
                                
                                save_data['BRAND_LIST'] = save_data['Brand']
                                
                                save_data['STATUS'] = 'Completed'
                                save_data['PERIOD_TYPE'] = Period_filter
                                save_data['MONTH_LIST'] = ', '.join(map(str, Month_filter))
                                save_data['CHANNEL_TYPE'] = Tactic_Type_filter
                                save_data['TACTIC_LIST'] = 'NA'
                                save_data["BATCH_ID"]=BUDGET_BATCH_ID
                                save_data['DESCRIPTION'] = Description
                                
                                
                                from snowflake.snowpark.context import get_active_session
                                session = get_active_session() 
                                
                                save_data=save_data[['Brand', 'Actual Tactic', 'BUDGET_WEEK_START_DT', 'C1', 'C2', 'C3', 'C4', 'CPM', 'Actual Spend (Selected Months)', 'ADJUSTMENT_FACTOR', 'REF_ADJ_FCTR', 'ECOMM_ROI', 'TACTIC_ADJ_FCTR', 'SEASONAL_ADJ_FCTR', 'ADSTOCK_X', 'CURVE_TYPE', 'ADSTOCK_WEEK', 'ADSTOCK', 'MONTH_YEAR', 'Group Tactic', 'FEC_FCTR', 'LTE_FCTR', 'Yearly', 'Quarterly', 'Monthly', 'Tactic', 'PERIOD', 'Sim_Per', 'Spend_Delta_value', 'Spend_value', 'Wk_Cnt', 'Simulation Spend_WK', 'Coef adjustment', 'CPM adjustment', 'SELECTED_MONTHS', 'ADSTOCK_X_Sim', 'Simulation Profit', 'ADSTOCK_X_Ana', 'Actual Profit', 'Actual FEC', 'Simulation FEC', 'Actual LTE Profit', 'Actual LTE FEC', 'Simulation LTE Profit', 'Simulation LTE FEC', 'COMMENTS', 'DATE', 'JOB_ID', 'USER_EMAIL', 'BRAND_LIST', 'STATUS', 'PERIOD_TYPE', 'MONTH_LIST', 'CHANNEL_TYPE', 'TACTIC_LIST', 'BATCH_ID', 'DESCRIPTION']]
                                session.create_dataframe(save_data).write.mode("append").save_as_table("ANALYTICS.UMM_OPTIM.PROD_SONIC_SIMULATION_TABLE")
                                # save_data
                                input_values_to_save=st.session_state.input_values_to_save
                                input_values_to_save['COMMENTS']=Scenario_name
                                input_values_to_save['JOB_ID']=JOB_ID
                                input_values_to_save['agg_level']=str(st.session_state.agg_level_list)
                                input_values_to_save['type']=st.session_state.type
                                input_values_to_save['Period_filter']=Period_filter
                                input_values_to_save['Spend_Type']="Total Spend"
                                input_values_to_save['Yr_filter']=Yr_filter
                                input_values_to_save['Month_filter']=str(Month_filter)
                                input_values_to_save['INPUT_TYPE']=Tactic_Type_filter
                                input_values_to_save['user_email']=st.experimental_user.email
        
                                req_column=["Brand", "Tactic", "Subbrand", "Channel", "Publishername","Coef adjustment", "CPM adjustment", "PERIOD",
                                 "Simulation Spend (Selected Months)", "COMMENTS", "JOB_ID", "agg_level",'type','Period_filter','Spend_Type',
                                            'Yr_filter','Month_filter','INPUT_TYPE','user_email']
                                # input_values_to_save.reindex(columns=req_column, fill_value=np.nan)
                                # input_values_to_save.assign(**{col: None for col in req_column if col not in input_values_to_save})
                                for col in req_column:
                                    if col not in input_values_to_save.columns:
                                        input_values_to_save[col] = None
                                input_values_to_save=input_values_to_save[req_column]
                                # input_values_to_save
                                session.create_dataframe(input_values_to_save).write.mode("append").save_as_table("ANALYTICS.UMM_OPTIM.PROD_SONIC_SIMULATION_INPUT_TABLE")
                                
                                st.success(f"Scenario '{Scenario_name}' Saved Successfully")
                                st.session_state.save_enable=False
                            
                            except SnowparkSQLException as e:
                                st.error(f"SQL Exception Occurred: {e}")
                                if "Insufficient privileges" in str(e):
                                    st.error("⚠️ Please use the Analytics_UserRole. If you do not have access, reach out to admin!.")
                                
            
        with st.container(border=True):
            yly=st.radio(r"$\textrm{\large Select the option to view the Report}$",["By Entire Year","By Selected Month",],horizontal=True)

   
        # with st.container(border=True):
        #     lte_fec=st.toggle("Enable LTE FEC")
        lte_fec=False
        if yly=="By Entire Year":
            with st.spinner(text="Calculating ..."):
                st.session_state.Yearly_view=True
                

                profit_cal(st.session_state['un_sim_grouped_data'])
              
    
        else:
            with st.spinner(text="Calculating ..."):
                st.session_state.Yearly_view=False
                
                  
                profit_cal(st.session_state['un_sim_grouped_data'])
        chart_df=st.session_state['final_grp_output_df']

        # st.write("CHARTTTT")
        # st.write(chart_df)

        gt_df=chart_df[ (chart_df['Tactic'] == "Grand Total")]


        #gt_df_2024=chart_df[ (chart_df['Tactic'] == "Grand_Total_2024")]

        gt_df_2024 = gt_df.copy()
        
        def format_currency(value):
                        if abs(value) >= 1e9:
                            return '₩{:.2f}B'.format(value / 1e9)
                        else:
                        # elif abs(value) >= 1e6:
                            return '₩{:.2f}M'.format(value / 1e6)
                        # # else:
                        # elif abs(value) >= 1e3:
                        #     return '₩{:.2f}K'.format(value / 1e3)
                        # else:
                        #    return '₩{:.2f}'.format(value)

        if lte_fec==True:
            col1,col2,col3,col4=st.columns(4)
        else:
            col1,col2,col3=st.columns(3)     
    
        with col1:
            
            O_Spend=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'SPEND')].reset_index()['Value'][0]
            O_Spend_2024=gt_df_2024[(gt_df_2024['Flag'] == 'Actual') & (gt_df_2024['Measure'] == 'SPEND')].reset_index()['Value'][0]
            S_Spend=gt_df[(gt_df['Flag'] == 'Simulation') & (gt_df['Measure'] == 'SPEND')].reset_index()['Value'][0]
            
            
            html_output = cards_html("Spend",format_currency(S_Spend), format_currency(O_Spend), str(round(((S_Spend/O_Spend) - 1)*100,2))+"%",
                                     format_currency(S_Spend-O_Spend),'#d1e5f0',(21, 73, 128), 18, 20, "example-icon",
                                     format_currency(O_Spend_2024),str(round(((O_Spend/O_Spend_2024) - 1)*100,2))+"%",format_currency(O_Spend-O_Spend_2024) )
           
            st.markdown(lnk + html_output, unsafe_allow_html=True)

        with col2:
            
            O_ROI=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'ROI')].reset_index()['Value'][0]
            O_ROI_2024=gt_df_2024[(gt_df_2024['Flag'] == 'Actual') & (gt_df_2024['Measure'] == 'ROI')].reset_index()['Value'][0]
            S_ROI=gt_df[(gt_df['Flag'] == 'Simulation') & (gt_df['Measure'] == 'ROI')].reset_index()['Value'][0]
            # Call the function with example values provided inline
            html_output = cards_html("ROI","₩"+str(S_ROI), "₩"+str(O_ROI), str(round((S_ROI/O_ROI)-1,2))+"%",
                                     "₩"+str(round((S_ROI-O_ROI),2)),'#e7d4e8', (78, 27, 101), 18, 20, "example-icon",
                                      "₩"+str(O_ROI_2024), str(round((O_ROI/O_ROI_2024)-1,2))+"%","₩"+str(round((O_ROI-O_ROI_2024),2)) )
            
            st.markdown(lnk + html_output, unsafe_allow_html=True)
    
        with col3:
                              
            O_FEC=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'FEC')].reset_index()['Value'][0]
            O_FEC_2024=gt_df_2024[(gt_df_2024['Flag'] == 'Actual') & (gt_df_2024['Measure'] == 'FEC')].reset_index()['Value'][0]
            S_FEC=gt_df[(gt_df['Flag'] == 'Simulation') & (gt_df['Measure'] == 'FEC')].reset_index()['Value'][0]

            html_output = fec_cards_html("Revenue",format_currency(S_FEC),format_currency(O_FEC),str(round(((S_FEC/O_FEC) - 1)*100,2))+"%" ,
                                     format_currency(S_FEC-O_FEC),'#d9f0d3', (44, 118, 130), 18, 20, "example-icon",
                                     format_currency(O_FEC_2024),str(round(((O_FEC/O_FEC_2024) - 1)*100,2))+"%" ,
                                     format_currency(O_FEC-O_FEC_2024))
            
            st.markdown(lnk + html_output, unsafe_allow_html=True)
        if lte_fec==True:
            with col4:
                              
                O_FEC=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'LTE FEC')].reset_index()['Value'][0]
                O_FEC_2024=gt_df_2024[(gt_df_2024['Flag'] == 'Actual') & (gt_df_2024['Measure'] == 'LTE FEC')].reset_index()['Value'][0]
                S_FEC=gt_df[(gt_df['Flag'] == 'Simulation') & (gt_df['Measure'] == 'LTE FEC')].reset_index()['Value'][0]
                html_output = cards_html('LTE FEC',format_currency(S_FEC),format_currency(O_FEC),str(round(((S_FEC/O_FEC) - 1)*100,2))+"%" ,
                                         format_currency(S_FEC-O_FEC),'#f0d3d9', (102, 51, 51), 18, 20, "example-icon",
                                         format_currency(O_FEC_2024),str(round(((O_FEC/O_FEC_2024) - 1)*100,2))+"%" ,
                                         format_currency(O_FEC-O_FEC_2024) )
                
                st.markdown(lnk + html_output, unsafe_allow_html=True)
        st.markdown("")
        tab1, tab2,tab3 = st.tabs(["📊Comparison Chart","📈 Revenue Delta Chart", "🛢 Detail Report"])
    
        css = '''
            <style>
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                    font-size: 1.3rem;
                   
                }
            </style>
        '''
    
        st.markdown(css, unsafe_allow_html=True)
    
        # st.write("CHARTTT")
        # st.dataframe(st.session_state['final_grp_output_df'])

        # st.session_state['grp_output_df']
        with tab1.subheader(" "):
        # if selected == "Comparison Chart":
            if "grp_output_df" not in st.session_state:
                    st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            elif st.session_state.update_status=="TRUE":
                st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            else:
                
                chart_df=st.session_state['final_grp_output_df']

                gt_df=chart_df[ (chart_df['Tactic'] == "Grand Total")]

                if lte_fec==True:
                    subplot_list=("Spend", "ROI", "Revenue",'LTE FEC')
                    no_columns=4
                    measures = ['SPEND', 'ROI', 'FEC','LTE FEC']
                else:
                    subplot_list=("Spend", "ROI", "Revenue")
                    no_columns=3
                    measures = ['SPEND', 'ROI', 'FEC']

                fig = make_subplots(rows=1, cols=no_columns, subplot_titles=subplot_list,
                              horizontal_spacing= 0.02
                            )
                
                
                flags = [ 'Simulation','Actual']
                
                # Define background colors for each column
                background_colors = ['#e8ebf1']
                
                # Define bar colors for each flag
                bar_colors = {
                    'Simulation': '#f5a968',
                    'Actual': '#6686a6'
                }
                
                color_discrete_sequence = ['#f5a968', '#6686a6']
                for i, measure in enumerate(measures):
                    for j, flag in enumerate(flags):
                          # Sort by  column
                        
                        # Determine the texttemplate based on the column index
                        if i == 1:  # Column 2 (ROI)
                            text_template = "%{x:₩,.2f}"  # Format as dollars with 2 decimal places
                        else:  # Column 1 (Spend) and Column 3 (FEC)
                            text_template = "%{x:₩,.2s}"  # Format as '0.2s' with dollar symbol
                      
                        chart_data = chart_df[(chart_df['Flag'] == flag) & (chart_df['Measure'] == measure) & (chart_df['Tactic'] != "Grand Total")  & (chart_df['Tactic'] != "Grand_Total_2024")].sort_values(by='row_numbers')
                        fig.add_trace(
                        go.Bar(y=chart_data['Tactic'], 
                               x=chart_data['Value'],  # Scale down the value for better visibility
                               name = 'Actual' if flag == 'Actual' else 'Simulation',
                               orientation='h',
                               text=chart_data['Value'],  # Enable value labels
                               texttemplate=text_template,  # Apply the custom text template
                               textposition='auto', 
                               textfont=dict(
                                        color='black',      # White text for readability
                                        size=16,
                                        family='Arial, sans-serif'
                                    ),
                               insidetextanchor='middle',
                               marker_color=bar_colors[flag] 
                              ), 
                        row=1, col=i+1
                    )
                        
                        # Update x-axis tick format
                        if i == 1:  # Column 2 (ROI)
                            #fig.update_xaxes(tickprefix="$", tickformat=".2f", row=1, col=i+1)  # Format as dollars with 2 decimal places #original
                            fig.update_xaxes(tickprefix="₩", tickformat=".2f", row=1, col=i+1)
                        else:  # Column 1 (Spend) and Column 3 (FEC)
                            #fig.update_xaxes(tickprefix="$", tickformat="0.2s", row=1, col=i+1)  # Format as '0.2s' #original
                            fig.update_xaxes(tickprefix="₩", tickformat="0.2s", row=1, col=i+1)
                        if i > 0:  # Check if it's the second or third column
                            fig.update_yaxes(showticklabels=False, row=1, col=i+1)  # Hide y-axis labels
                        # Update grid settings for both x-axis and y-axis
                        fig.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
                        fig.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
       
                # Update y-axis font size for the first column
                fig.update_yaxes(tickfont=dict(size=15), row=1, col=1)
                
                # Update layout for better appearance
                fig.update_layout(
                    
                    barmode='group',
                    height=800,
                    
                    plot_bgcolor=background_colors[0],  # Set background color for the plot area
                    paper_bgcolor='rgba(0,0,0,0)',  # Set transparent background for the entire subplot
    
                )
                for idx in range(2, len(fig.data)):
                    fig.data[idx].showlegend = False

                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.22,
                        xanchor="right",
                        x=0.6,
                        font=dict(size=20),
                        traceorder="reversed"
                    ),
                    font=dict(size=12, color="black")
                )
             
                with st.container( border=True): 
                    
                    st.subheader("Actual vs Simulation ")
                    
    
                    
                    # st.plotly_chart(chart,use_container_width=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                
        with tab2.subheader(" "):
        # elif selected=="FEC Delta Chart":
            if "Delta" not in st.session_state:
                    st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            elif st.session_state.update_status=="TRUE":
                st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            else:
    
    
                chart_df=st.session_state['Delta']
                
                # Assuming 'measures' and 'chart_df' are defined elsewhere in the code
                
                # Create a subplot figure with 1 row and 2 columns
                fig = make_subplots(rows=1, cols=2, subplot_titles=("SPEND", "Revenue"))
                
                flags = ['SPEND', 'FEC']
                classifications = ['P', 'N']
                
                # Define background colors for each column
                background_colors = ['#e8ebf1']
                
                # Define bar colors for each classification
                bar_colors = {
                    'N': '#cc0000',
                    'P': '#078fd7'
                }
                
                # Loop over each flag and classification to create the plots
                for i, flag in enumerate(flags):
                    
                        chart_data = chart_df[(chart_df['Flag'] == flag) & (chart_df['Tactic'] != "Grand Total") ].sort_values(by=['row_numbers'])  # Sort by Tactic column
                        
                      
                        # Determine the texttemplate based on the flag
                        #text_template = "%{x:$,.2f}" if flag == 'ROI' else "%{x:$,.2s}"  # Format based on the flag # original
                        text_template = "%{x:₩,.2f}" if flag == 'ROI' else "%{x:₩,.2s}"
                        colors = [bar_colors[classification] for classification in chart_data['Classification']]
                
                        fig.add_trace(
                            go.Bar(y=chart_data['Tactic'], 
                                   x=chart_data['Measure'], 
                                   
                                   orientation='h',
                                   text=chart_data['Measure'],  # Enable value labels
                                   texttemplate=text_template,  # Apply the custom text template
                                   textposition='auto',  # Position text auto
                                   marker_color=colors  # Set bar color based on classification
                                  ), 
                            row=1, col=i+1
                        )
                
                        # Update x-axis tick format
                        #fig.update_xaxes(tickprefix="$", tickformat=".2f" if flag == 'ROI' else "0.2s", row=1, col=i+1)# original
                        fig.update_xaxes(tickprefix="₩", tickformat=".2f" if flag == 'ROI' else "0.2s", row=1, col=i+1)
                        if i > 0:  # Check if it's the second column
                            fig.update_yaxes(showticklabels=False, row=1, col=i+1)  # Hide y-axis labels
    
                        # Update grid settings for both x-axis and y-axis
                        fig.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
                        fig.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
    
                # Update layout for better appearance
                fig.update_layout(
                    barmode='group',
                    height=800,
                    showlegend=False,
                    plot_bgcolor=background_colors[0],  # Set background color for the plot area
                    paper_bgcolor='rgba(0,0,0,0)',  # Set transparent background for the entire subplot
    
                )
                
                fig.update_yaxes(tickfont=dict(size=15), row=1, col=1)
               
                
                   
               
                with st.container( border=True): 
                    
                    st.subheader("Delta Value")
                    
                    st.plotly_chart(fig,use_container_width=True)
                       
        
        # st.session_state['grp_output_df']
        with tab3.subheader(" "):
        # elif selected=="Detail Report":
        
            if "output_df" not in st.session_state:
                st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            elif st.session_state.update_status=="TRUE":
                st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
            else:
                with st.container(border=True):
                    final_Detail_Report=st.session_state['output_df'][['Brand','Tactic','MONTH_YEAR','Yearly','Monthly','Quarterly','Actual Spend (Selected Months)',
                                                                       'Actual Profit','Actual LTE Profit','Simulation Spend_WK','Simulation Profit','Simulation LTE Profit',
                                                                      'Actual FEC','Actual LTE FEC','Simulation FEC','Simulation LTE FEC']]
                    # Brand_sim_filter_data=st.session_state['Brand_sim_filter_data']
                    # Brand_sim_filter_data['Actual Profit']=[calculate_revenue(row['Actual Spend (Selected Months)'], row["C1"], row["C2"], row["C3"], row["C4"], row["CURVE_TYPE"], row["CPM"],
                    #                                                       row["ADJUSTMENT_FACTOR"], row["REF_ADJ_FCTR"], row["TACTIC_ADJ_FCTR"], row["SEASONAL_ADJ_FCTR"], row["ADSTOCK_X"], 
                    #                                                       row["ECOMM_ROI"]) for i, row in Brand_sim_filter_data.iterrows()]
                    # Brand_sim_filter_data['Actual FEC']=Brand_sim_filter_data['Actual Profit']*Brand_sim_filter_data['FEC_FCTR']
                    # columns_to_define_zero = ['Simulation Spend_WK', 'Simulation Profit',  'Simulation FEC']
                    # Brand_sim_filter_data[columns_to_define_zero] = 0
                    # Brand_sim_filter_data=Brand_sim_filter_data[['Brand','Tactic','MONTH_YEAR','Yearly','Monthly','Quarterly','Actual Spend (Selected Months)',
                    #                                                    'Actual Profit','Simulation Spend_WK','Simulation Profit',
                    #                                                   'Actual FEC','Simulation FEC']]
    
                    
                    # final_Detail_Report= pd.concat([final_Detail_Report, Brand_sim_filter_data], ignore_index=True)
                    
                    final_Detail_Report=final_Detail_Report.rename(columns={"MONTH_YEAR":"Week Start","Simulation Spend_WK":'Simulation Spend (Selected Months)',
                                                                            'Actual Profit':'Budget Profit',
                                                                           'Actual FEC':'Actual Revenue',
                                                                           'Simulation FEC' : 'Simulation Revenue'
                                                                           })
                    # final_Detail_Report
                    # Define the columns for grouping and the remaining columns
                    final_Detail_Report['Monthly'] = pd.to_datetime(final_Detail_Report['Week Start']).dt.strftime('%B').astype(str)
                    #st.write(final_Detail_Report.columns)
                    
                    group_cols = ['Brand', 'Tactic', 'Yearly', 'Monthly', 'Quarterly']
                    sum_cols = ['Actual Spend (Selected Months)','Budget Profit','Actual LTE Profit','Simulation Spend (Selected Months)','Simulation Profit','Simulation LTE Profit','Actual Revenue','Actual LTE FEC','Simulation Revenue','Simulation LTE FEC']
                    # Group by the specified columns and sum the remaining columns
                    final_Detail_Report = final_Detail_Report.groupby(group_cols)[sum_cols].sum().reset_index()
                    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                    final_Detail_Report['Monthly'] = pd.Categorical(final_Detail_Report['Monthly'], categories=month_names, ordered=True)
                    final_Detail_Report=final_Detail_Report.sort_values(['Tactic','Monthly'])

                    col1,col2,col3,col4,col5 = st.columns(5)
                    with col1:
                        st.subheader("Detail Report")
                    with col5:
                        @st.cache_data
                        def convert_df(df):
                            # 
                            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                            return df.to_csv().encode("utf-8")
                        
                        csv = convert_df(final_Detail_Report)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="Detail_report.csv",
                            mime="text/csv",
          
                        )
                    # Define function to format currency
                    def format_currency(amount):
                        #return '$ {:,.0f}'.format(amount)# original
                        return '₩ {:,.0f}'.format(amount)
                    # Apply formatting function to the specified columns
                    currency_columns = sum_cols
                    for col in currency_columns:
                        final_Detail_Report[col] = final_Detail_Report[col].map(format_currency)
    
                    st.dataframe(final_Detail_Report,use_container_width=True,height=800 ,hide_index=True)