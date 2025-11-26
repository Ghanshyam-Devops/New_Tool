# Import necessary libraries
import pandas as pd
import numpy as np
import math
import streamlit as st
import plotly.express as px
# from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from st_aggrid import AgGrid, GridOptionsBuilder
import multiprocessing
import streamlit as st
from  datetime import  datetime, timedelta
from dateutil.parser import parse
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
import re
 
from auth import require_login

require_login()





def get_snowflake_Data(date = (None, None), actual_data_yr_filter1 = "2025_04 Rollup v2"):


    # session = get_active_session() 
    
    # user_email=str(st.experimental_user.email)
    
    # if date[0] is None and date[1] is None:
    #         today = datetime.now().date()
    #         date = ((today - timedelta(days=30)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    
    
    # sql1 = f"""
    # select a.*,case when b.shared_jobid is null then a.user_email else 'shared' end as final_user_email from ANALYTICS.UMM_OPTIM.PROD_SONIC_SIMULATION_TABLE a 
    # left join ANALYTICS.UMM_OPTIM.PROD_SONIC_SHARED_OUTPUT_TABLE b 
    # on a.job_id=b.shared_jobid  
    # -- where final_user_email in ('{user_email}','shared' )
    # WHERE a.date BETWEEN TO_DATE('{date[0]}', 'YYYY-MM-DD') AND TO_DATE('{date[1]}', 'YYYY-MM-DD')

    
    
    # """
    # sql2 = f"""
    
    # select a.*,case when b.shared_jobid is null then a.user_email else 'shared' end as final_user_email from ANALYTICS.UMM_OPTIM.PROD_SONIC_OPTIMIZATION_TABLE a 
    # left join ANALYTICS.UMM_OPTIM.PROD_SONIC_SHARED_OUTPUT_TABLE b 
    # on a.job_id=b.shared_jobid  
    # WHERE a.date BETWEEN TO_DATE('{date[0]}', 'YYYY-MM-DD') AND TO_DATE('{date[1]}', 'YYYY-MM-DD')
    

    
    # """
    # sql3 = f"""
    #           SELECT CONCAT(MAP.ROLLUP_NAME,'--',WKM.SONIC_YEAR,' Yr') as ROLLUP_NAME, MAP.ROLLUP_NAME as RAW_ROLLUP_NAME,BATCH_ID, INSERT_TIMESTAMP, INCLUDE_NPD, INITCAP(BS.BRAND) as BRAND , INITCAP(BS.TACTIC) as TACTIC, 
    # BUDGET_WEEK_START_DT, BUDGET_SPEND, 
    # CPM_REF_START_DT, CPM_REF_END_DT, CPM_OVERRIDE,
    # ORIGINAL_CPM, BS.CPM_ADJ_FCTR, CPM_VALUE, EXECUTION_MILLIONS, EXECUTION, CURVE_TYPE, C1, C2, C3, C4, 
    # FUNCTION, ADSTOCK, ADSTOCK_WEEK, EXECUTION_CUTOFF, X0, ADSTOCK_X, X, MOD_FUNC, CURVE_PROFIT, 
    # CURVE_ADJ_START_DT, CURVE_ADJ_END_DT, ADJUSTMENT_FACTOR, ECOMM_EFF_START_DT, ECOMM_EFF_END_DT, ECOMM_ROI,
    # ADJUSTED_PROFIT, BS.AUDIENCE_ADJ_FCTR, BS.CREATIVE_ADJ_FCTR, BS.SUBBRAND_ADJ_FCTR, BS.NPD_ADJ_FCTR, BS.MISC_ADJ_FCTR,
    # TACTIC_ADJ_FCTR, ADJ_TACTIC_PROFIT, SEASONAL_ADJ_FCTR, ADJ_SEASONAL_PROFIT, FINAL_PROFIT, ROI, FEC, 
    # CASE WHEN NT.REF_ADJ_FCTR IS NULL THEN 1 ELSE NT.REF_ADJ_FCTR END AS REF_ADJ_FCTR, 
    # WKM.SONIC_YEAR, SONIC_QUARTER, SONIC_MONTH, MONTH_YEAR 
    # ,INITCAP(th.SONIC_METRIC_GROUP) as grp_name,FF.FEC_PROFIT_FCTR as FEC_FCTR,FLTE.LTE_FCTR
    
    # FROM ANALYTICS.UMM_SONIC.UMM_SONIC_BUDGET_WKLY_SNAPSHOT BS
    
    # INNER JOIN ANALYTICS.UMM_SONIC_STAGE.UMM_SONIC_WKLY_METADATA WKM
    # ON BS.BUDGET_WEEK_START_DT = WKM.WEEK_START_MONDAYS
    # left join ANALYTICS.UMM_SONIC.UMM_SONIC_FEC_FCTR_LKP FF 
    # on INITCAP(BS.BRAND)=INITCAP(FF.BRAND)
    # left join (select distinct BRAND, SONIC_METRIC,SONIC_METRIC_GROUP from  ANALYTICS.UMM_SONIC.SONIC_MEDIA_MAPPING ) TH
    # ON BS.BRAND = TH.BRAND
    # AND BS.TACTIC = TH.SONIC_METRIC
    
    # LEFT OUTER JOIN (
    #     SELECT 
    #     CASE 
    #         WHEN TARGET_BRAND = 'ASO' THEN 'ALKA SELTZER ORIGINAL'
    #         WHEN TARGET_BRAND = 'ASP' THEN 'ALKA SELTZER PLUS' 
    #         WHEN TARGET_BRAND = 'OAD' THEN 'ONE A DAY' 
    #         WHEN TARGET_BRAND = 'ASPIRIN' THEN 'BAYER ASPIRIN' 
    #         ELSE TARGET_BRAND
    #     END AS TARGET_BRAND,  
    #     TARGET_TACTIC, 
    #     REF_ADJ_FCTR 
    #     FROM ANALYTICS.UMM_CURVES.UMM_CURVE_COEFF_NEW_TACTICS
    # ) NT
    # ON BS.BRAND = UPPER(NT.TARGET_BRAND)
    # AND BS.TACTIC = UPPER(NT.TARGET_TACTIC)
    
    # left join  ANALYTICS.UMM_SONIC.UMM_SONIC_ADJ_TACTIC_FCTR_LKP FLTE
    # ON BS.BRAND = 
    # (CASE 
    #         WHEN FLTE.BRAND = 'ASO' THEN 'ALKA SELTZER ORIGINAL'
    #         WHEN FLTE.BRAND = 'ASP' THEN 'ALKA SELTZER PLUS' 
    #         WHEN FLTE.BRAND = 'OAD' THEN 'ONE A DAY' 
    #         WHEN FLTE.BRAND = 'ASPIRIN' THEN 'BAYER ASPIRIN' 
    #         ELSE FLTE.BRAND END)
    # AND BS.TACTIC = FLTE.TACTIC
    # AND BS.BUDGET_WEEK_START_DT=FLTE.WEEK_START_MONDAYS
    # left join ANALYTICS.UMM_SONIC.UMM_SONIC_BATCH_ROLLUP_LKP MAP on BS.BATCH_ID=MAP.SONIC_BATCH_ID
    # -- WHERE MAP.published_flag=True
    # WHERE BS.BATCH_ID in (select distinct sonic_batch_id from ANALYTICS.UMM_SONIC.UMM_SONIC_BATCH_ROLLUP_LKP where published_flag=True order by 1 desc)
    # AND lower(BS.TACTIC) NOT IN ('non-working costs','not in model','unallocated')  
    # and lower(bs.brand) in ('claritin','aleve','one a day','miralax','astepro','alka seltzer plus','alka seltzer original','afrin','lotrimin','bayer aspirin','midol','flintstones','phillips','coricidin','a&d')
    # and map.rollup_name = '{actual_data_yr_filter1}'
    # """
    

    #start = datetime.now()
    #with ProcessPoolExecutor(max_workers=3) as executor:
        
    #    future1 = executor.submit(lambda: pd.DataFrame(Session.builder.configs(snowflake_config).create().sql(sql1).collect()))
    #    future2 = executor.submit(lambda: pd.DataFrame(Session.builder.configs(snowflake_config).create().sql(sql2).collect()))
    #    future3 = executor.submit(lambda: pd.DataFrame(Session.builder.configs(snowflake_config).create().sql(sql3).collect()))


    #    data1 = future1.result()
    #    data2 = future2.result()
    #    data3 = future3.result()

        
    #    output_history_data1 = pd.DataFrame(data1)
    #    output_history_data2 = pd.DataFrame(data2)
    #    st.session_state.actual_data = pd.DataFrame(data3)

    #st.write("Parallel Execution:", datetime.now() - start) 

    # start = datetime.now()
    
    # data1 = session.sql(sql1).collect()
    # output_history_data1=pd.DataFrame(data1)

    # data2 = session.sql(sql2).collect()
    # output_history_data2=pd.DataFrame(data2)

    # if not data1 or not data2:
    #    st.warning("No data found for the selected date range. Please try a different range.")
    #    st.stop()
    # output_history_data2['JOB_ID'] = output_history_data2['JOB_ID'].str.split("_", expand=True)[0]
    
    # data3 = session.sql(sql3).collect()
    # st.session_state.actual_data=pd.DataFrame(data3)
    # st.write("Sequential Execution:", datetime.now() - start)
    


    #st.write("data1", output_history_data1.shape)
    #st.write("data2", output_history_data2.shape)
    #st.write("actual", st.session_state.actual_data.shape)
    
    # output_history_data1.rename(columns={'Simulation Spend_WK': 'OPTIM/SIM_SPEND','Simulation FEC':'OPTIM/SIM_FEC','Simulation Profit':'OPTIM/SIM_PROFIT',"Simulation LTE Profit":"OPTIM/SIM_LTE_PROFIT",'Simulation LTE FEC':'OPTIM/SIM_LTE_FEC'}, inplace=True)
    # output_history_data2.rename(columns={'OPTIMZED_SPEND': 'OPTIM/SIM_SPEND','OPTIMIZED_FEC':'OPTIM/SIM_FEC','OPTIMIZED_PROFIT':'OPTIM/SIM_PROFIT'}, inplace=True)
    # output_history_data1["OPTIM/SIM"]="SIM"
    # output_history_data2["OPTIM/SIM"]="OPTIM"
    

    # output_history_data=pd.concat([output_history_data1,output_history_data2],ignore_index=True)
    # st.dataframe(output_history_data.head())
    
    # output_history_data['TARGET'].fillna('SIM',inplace=True)
    
    # output_history_data[['DATE_only', 'TIME']] = output_history_data['DATE'].str.split(' ',expand=True)
    # output_history_data['DATE_only'] = pd.to_datetime(output_history_data['DATE_only'], format='%Y-%m-%d').dt.date
    # output_history_data=output_history_data[output_history_data['SELECTED_MONTHS']==True]

    #st.dataframe(output_history_data.head(2))

    output_history_data = pd.read_excel("output_file.xlsx")
    # st.write(output_history_data.shape
    output_history_data = output_history_data.reset_index(drop=True)
    # output_history_data
    output_history_data.rename(columns={'OPTIMZED_SPEND': 'OPTIM/SIM_SPEND',
                                        'OPTIMIZED_FEC': 'OPTIM/SIM_FEC',
                                        'OPTIMIZED_PROFIT': 'OPTIM/SIM_PROFIT'}, inplace=True)
    output_history_data
    # output_history_data[['DATE_only', 'TIME']] = output_history_data['DATE'].str.split(' ',expand=True)
    # Convert DATE column to string if it's not already
    output_history_data['DATE'] = output_history_data['DATE'].astype(str)

    # Split the 'DATE' column into 'DATE_only' and 'TIME'
    output_history_data[['DATE_only', 'TIME']] = output_history_data['DATE'].str.split(' ', expand=True)

    #st.dataframe(output_history_data)

    #output_history_data['DATE_only'] = pd.to_datetime(output_history_data['DATE_only'], format='%Y-%m-%d').dt.date
    #output_history_data['DATE_only'] = pd.to_datetime(output_history_data['DATE_only'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
    output_history_data['DATE_only'] = pd.to_datetime(output_history_data['DATE_only'].str.strip(), format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    output_history_data=output_history_data[output_history_data['SELECTED_MONTHS']==True]
    output_history_data.loc[:, 'OPTIM/SIM'] = 'OPTIM'
    
    st.session_state.output_history_data=output_history_data
    st.session_state.brand_list=output_history_data['BRAND_LIST'].unique().tolist()
    
    actual_data = output_history_data.copy()
    actual_data.rename(columns = {'Brand' : 'BRAND'}, inplace=True)
    actual_data.loc[:, 'ROLLUP_NAME'] = "2025_04 Rollup v2"
    actual_data['FEC'] = np.random.normal(900, 3500, actual_data.shape[0])
    actual_data.loc[:, 'RAW_ROLLUP_NAME'] = "2025_04 Rollup v2--2025 Yr"
    actual_data.loc[:, 'SONIC_YEAR'] = 2025
    actual_data['SPEND'] = np.random.normal(1000000, 500000, actual_data.shape[0])
    actual_data['PROFIT'] = output_history_data['Actual Spend (Selected Months)'].copy()

    #st.dataframe(actual_data.columns)
    #st.dataframe(output_history_data.head(10))
    # st.write("SNOWFLAKEEE")
    # st.dataframe(output_history_data)
    
    st.session_state.actual_data = actual_data

# @st.cache_data
def report_generate(output_df):
    # output_df[output_df['Quarterly']=='Q1-24']

    st.session_state['output_df']=output_df
    
    grp_output_df = output_df.groupby(['TARGET','Brand', 'Tactic']).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit': 'sum', 'OPTIM/SIM_SPEND': 'sum', 'OPTIM/SIM_PROFIT': 'sum','Actual FEC':'sum','OPTIM/SIM_FEC':'sum', 'Actual LTE FEC':'sum',"OPTIM/SIM_LTE_FEC":'sum'}).reset_index()
    
    grand_total_row = grp_output_df.groupby('TARGET')[['Actual Spend (Selected Months)', 'Actual Profit', 'OPTIM/SIM_SPEND', 'OPTIM/SIM_PROFIT', 'Actual FEC', 'OPTIM/SIM_FEC', 'Actual LTE FEC', 'OPTIM/SIM_LTE_FEC']].sum().reset_index()
    
    grand_total_row["Tactic"]="Grand Total"
    
    # Appending the grand total row to grp_output_df
    grp_output_df = pd.concat([grp_output_df, grand_total_row], ignore_index=True)
    grp_output_df['Spend_DIFF'] = ((grp_output_df['OPTIM/SIM_SPEND'] / grp_output_df['Actual Spend (Selected Months)'])-1)
    grp_output_df['Profit_DIFF'] = (grp_output_df['OPTIM/SIM_PROFIT'] / grp_output_df['Actual Profit'])-1
    grp_output_df['Original ROI']=(grp_output_df['Actual Profit']/grp_output_df['Actual Spend (Selected Months)']).round(2)
    grp_output_df['Simulation ROI']=(grp_output_df['OPTIM/SIM_PROFIT']/grp_output_df['OPTIM/SIM_SPEND']).round(2)
    
    grp_output_df['FEC_Diff']=(grp_output_df['OPTIM/SIM_FEC']-grp_output_df['Actual FEC'])
    
    grp_output_df['Spend_Diff_value']=(grp_output_df['OPTIM/SIM_SPEND'] - grp_output_df['Actual Spend (Selected Months)'])
    grp_output_df['Classification'] = np.where(grp_output_df['FEC_Diff'] > 0, 'P', 'N')
    grp_output_df['Spend_Classification'] = np.where(grp_output_df['Spend_Diff_value'] > 0, 'P', 'N')
    grp_output_df=grp_output_df.sort_values(['Classification','FEC_Diff'], ascending=False)

    #st.dataframe(grp_output_df)

    FECdelta_grp= grp_output_df[['TARGET','Tactic','FEC_Diff','Classification']].rename(columns={'FEC_Diff':'Measure'})
    FECdelta_grp['Flag']='FEC'
    
    Spenddelta_grp= grp_output_df[['TARGET','Tactic','Spend_Diff_value','Spend_Classification']].rename(columns={'Spend_Diff_value':'Measure','Spend_Classification':'Classification'})
    Spenddelta_grp['Flag']='SPEND'
    
    st.session_state['grp_output_df']=grp_output_df
    
    # st.write("Report Generated Successfully")
    # st.write(st.session_state.grp_output_df.head(2))

    grp_output_diff = grp_output_df[['Spend_DIFF','Profit_DIFF']].copy()
    
    actual_grp_output_df= grp_output_df[['TARGET','Brand', 'Tactic','Actual Spend (Selected Months)','Actual Profit','Original ROI','Actual FEC','Actual LTE FEC']].rename(columns={'Actual Spend (Selected Months)': 'SPEND','Actual Profit':'PROFIT', 'Original ROI': 'ROI','Actual FEC':'FEC','Actual LTE FEC':'LTE FEC'})
    actual_grp_output_df['Flag']='Actual'
    
    sim_grp_output_df=grp_output_df[['TARGET','Brand', 'Tactic','OPTIM/SIM_SPEND','OPTIM/SIM_PROFIT','Simulation ROI',"OPTIM/SIM_FEC",'OPTIM/SIM_LTE_FEC']].rename(columns={'OPTIM/SIM_SPEND': 'SPEND','OPTIM/SIM_PROFIT':'PROFIT', 'Simulation ROI': 'ROI',"OPTIM/SIM_FEC":'FEC','OPTIM/SIM_LTE_FEC':'LTE FEC'})
    sim_grp_output_df['Flag']='OPTIM/SIM'
    
    final_grp_output_df=pd.concat([sim_grp_output_df,actual_grp_output_df])

    # final_grp_output_df=pd.concat([final_grp_output_df,st.session_state.final_actula_data],ignore_index=True)
    
    st.session_state.a_test= final_grp_output_df

    final_grp_output_df=final_grp_output_df.melt(id_vars=['TARGET','Brand', 'Tactic', 'Flag'], 
                                     value_vars=['SPEND', 'ROI',"FEC",'LTE FEC'], 
                                     var_name='Measure', 
                                     value_name='Value')

    final_grp_output_df['Measure']=pd.Categorical(final_grp_output_df['Measure'], categories=['SPEND', 'ROI',"FEC",'LTE FEC'], ordered=True)
    final_grp_output_df=final_grp_output_df.sort_values(by=['Measure','Value'])
    for_order=final_grp_output_df[final_grp_output_df['Measure']=="SPEND"]
    for_order=for_order[for_order["TARGET"]==final_grp_output_df["TARGET"].dropna().unique()[0]]

    for_order=for_order[for_order['Flag']=='OPTIM/SIM']
    for_order=for_order.sort_values(by=['Value'])

    for_order.reset_index(drop=True, inplace=True)
    for_order['row_numbers'] = for_order.index + 1 
    for_order=for_order[['Brand', 'Tactic','row_numbers']]

    final_grp_output_df["TARGET"] = final_grp_output_df.apply(lambda row: None if row["Flag"] == "Actual" else row["TARGET"], axis=1)

    final_grp_output_df=final_grp_output_df.drop_duplicates().reset_index(drop=True)

    final_grp_output_df=final_grp_output_df.merge(for_order, on=['Brand','Tactic'])
    # final_grp_output_df
    # Clean values (shared logic)
    clean = final_grp_output_df['Value'].replace([np.inf, -np.inf], 0).fillna(0)

    # Apply ROI rule (float, 2 decimals)
    final_grp_output_df.loc[final_grp_output_df['Measure'] == 'ROI', 'Value'] = (
        clean.round(2)
    )

    # Apply SPEND/FEC rule (int)
    final_grp_output_df.loc[
        final_grp_output_df['Measure'].isin(['SPEND', 'FEC']), 'Value'
    ] = clean.astype(int)
 
    st.session_state['final_grp_output_df']=final_grp_output_df

    Delta_df=pd.concat([FECdelta_grp,Spenddelta_grp]).merge(for_order, on='Tactic').sort_values(by='row_numbers')

    #st.dataframe(Delta_df)
    
    #Delta_df["Tactic"] = np.where(Delta_df["TARGET"] == "FEC", Delta_df["Tactic"] + "- Revenue", Delta_df["TARGET"])
    
    st.session_state['Delta']=Delta_df

    # st.write("CHECKING")
    # st.dataframe(Delta_df)

    return final_grp_output_df, Delta_df


def update_actaul_compare_data(hist_brand_filter, actual_data_yr_filter1 = '2025_04 Rollup v2', actual_data_yr_filter2 = '2025'):
    # for actual 2025 vs 2024

    actual_data=st.session_state['actual_data']
    # st.dataframe(actual_data.head())
    actual_data=st.session_state.actual_data[st.session_state.actual_data['BRAND'].isin(hist_brand_filter)]
    
    # st.write("Actual")
    # st.dataframe(actual_data.head())
    #actual_data['MONTH_YEAR'] = pd.to_datetime(actual_data['MONTH_YEAR'], format='%Y-%m')
    #actual_data['MONTH_YEAR'] = pd.to_datetime(actual_data['MONTH_YEAR']).dt.to_period('M').dt.to_timestamp()
    #actual_data['MONTH_YEAR'] = pd.to_datetime(actual_data['MONTH_YEAR'], format='mixed', dayfirst=True, errors='coerce').dt.to_period('M').dt.to_timestamp()

    #actual_data['MONTH_YEAR'] = pd.to_datetime(actual_data['MONTH_YEAR'], format='mixed', dayfirst=True)
    
    # Now you can use it for further operations, like accessing quarters
    # actual_data['Quarterly'] = 'Q' + actual_data['MONTH_YEAR'].dt.quarter.astype(str) + "-" + actual_data['MONTH_YEAR'].dt.strftime('%y')
    #actual_data['Quarterly'] = 'Q' + actual_data['MONTH_YEAR'].dt.quarter.astype(str) + "-" + actual_data['MONTH_YEAR'].dt.strftime('%y')
    #actual_data['Month'] = actual_data['MONTH_YEAR'].dt.strftime('%b')

    actual_data['Month'] = actual_data['Monthly']

    if "selected_month_list" in st.session_state:
        actual_data=actual_data[actual_data['Month'].isin(st.session_state.selected_month_list)]
    else:
        actual_data=actual_data[actual_data['Month'].isin([ "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"])]
    
    actual_data=actual_data.rename(columns={'BRAND':'Brand','TACTIC':'Tactic','ROLLUP_NAME':'Flag','BUDGET_SPEND':'SPEND','FINAL_PROFIT':'PROFIT'})
    actual_data['Flag']=actual_data['Flag'].astype(str)
    actual_data["JOB_ID"]=actual_data['Flag']
    actual_data["DESCRIPTION"]=""
    actual_data['LTE_FEC']=actual_data['FEC']*actual_data['LTE_FCTR']
    actual_data=actual_data[['Month','Brand','Tactic','JOB_ID','Flag','RAW_ROLLUP_NAME','SONIC_YEAR','DESCRIPTION','SPEND','PROFIT','FEC','LTE_FEC']]
    actual_data1=actual_data[actual_data['RAW_ROLLUP_NAME']==actual_data_yr_filter1]
    actual_data1=actual_data1[actual_data1['SONIC_YEAR'].isin([actual_data_yr_filter2])]
    
    final_actula_data=pd.concat([actual_data1],ignore_index=True)
    # final_actula_data=final_actula_data[final_actula_data['Brand'].isin(hist_brand_filter)]
    # temp fix - need to figure out why dupplicate records
    # final_actula_data.drop_duplicates(inplace=True)
    final_actula_data=final_actula_data.groupby(['Brand','Tactic','JOB_ID','Flag','DESCRIPTION']).agg({'SPEND':'sum','PROFIT':'sum','FEC':'sum','LTE_FEC':'sum'}).reset_index()
    # st.session_state.a_test=final_actula_data
    final_actula_data['ROI']=final_actula_data['PROFIT']/final_actula_data['SPEND']
    
    st.session_state.final_actula_data = final_actula_data
    


def sidebarcontent():
    with st.container():
        
    # with st.container(border=True):
        h1,h2,h3,h4=st.columns([0.3,0.3,0.3,0.1])
    
        with st.sidebar.container():    
            st.header('',divider=True)
                
        with st.sidebar.container(border=True):
            st.subheader("Filter by:")
            hist_brand_filter=st.multiselect("Brand Name",["Select All"]+st.session_state.brand_list,default ="Select All")
            if "Select All" in hist_brand_filter:
                hist_brand_filter=st.session_state.brand_list 
        
        with st.sidebar.container(border=True):
            today = datetime.now()
            date_filter = st.date_input(
                "Select the Created Scenario run date range",
                (today - timedelta(days=6), today),
                format="MM.DD.YYYY",
            )
            #st.write(date_filter)
    
        with st.sidebar.container(border=True):
            Yr_filter =st.selectbox(" Scenario Year list:", [2025])
    
        with st.sidebar.container(border=False):
            if Yr_filter == 2024:
                month_list=[ "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
            elif Yr_filter == 2025:
                month_list = [ "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"]
        
        Month_filter='Select All'
        
        if "Select All" in Month_filter:
            Month_filter = month_list
            # Custom month order
        custom_month_order = {month: index for index, month in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    
        # Sort the month_list based on the custom order
        month_list.sort(key=lambda x: (x[-2:], custom_month_order[x[:3]]))  # Sort by year and custom month order
    
        # Concatenate the selected months while preserving custom order
        concat_month_list = ', '.join(sorted(set(Month_filter), key=lambda x: (x[-2:], custom_month_order[x[:3]])))
        # concat_month_list2 = ', '.join(sorted(set(Month_filter), key=lambda x: (custom_month_order[x[:3]])))
        # st.write(concat_month_list2)
        
        st.session_state.selected_month_list=[month[:3] for month in month_list]
        # st.session_state['output_history_data']
        # st.write(st.session_state.actual_data.head())
        
        with st.sidebar.container(border=False):
            # f1_=st.session_state.actual_data['RAW_ROLLUP_NAME'].unique().tolist()
            # f1_.sort(reverse=True)
            
            # f1_index
            # .index('2024_09 Rollup')
            # actual_data_yr_filter1=st.selectbox("Select the Batch Rollup and Period 1",f1_,index=0) #,index ='2024_09 Rollup--2024 Yr'
            actual_data_yr_filter1 = "2025_04 Rollup v2"
            
        with st.sidebar.container(border=False):
            # actual_data_yr_filter2_list=list(filter(lambda x: x != actual_data_yr_filter1, st.session_state.actual_data['SONIC_YEAR'].unique().tolist()))
            # # f2_index=actual_data_yr_filter2_list.index('2024_09 Rollup--2025 Yr')
            # actual_data_yr_filter2=st.multiselect("Select the Year",actual_data_yr_filter2_list,default=[2025])
            actual_data_yr_filter2 = 2025
            
        with st.sidebar.container():
            Apply_hist_filter=st.button("▶ Apply",key="histfilter",type="primary")

    return hist_brand_filter, date_filter, Yr_filter, Month_filter, actual_data_yr_filter1, actual_data_yr_filter2, Apply_hist_filter


# @ st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")



def cards_html(header_input, input1, input2, input3,input4, wch_colour_box, wch_colour_font, fontsize1, fontsize2, iconname):
    header_input = 'Revenue' if header_input == 'FEC' else header_input
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
                            border-radius: 0px; 
                            padding-left: 12px; 
                            padding-top: 2px; 
                            padding-bottom: 2px; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>
                    <p class="card-text">
                        <span style='font-size: {fontsize2}px; padding-right: 12px;padding-top: 25px; float: right;'> Difference : {input4} (<span class="bold-text">{input3}</span>)</span>
                        <br><br>
                        <span class="bold-text">Optimized : {input1}</span>
                    </p>
                </div>"""
#<span class="bold-text">{header_input} Optimized : {input1}</span>


def format_currency(value):
    if abs(value) >= 1e9:
        return '₩{:.2f}B'.format(value / 1e9)
    else:
    # elif abs(value) >= 1e6:
        return '₩{:.2f}M'.format(value / 1e6)
        

def create_chart(chart_df, measures, flags, bar_colors, background_colors, no_columns, subplot_list):
    #fig2 = go.Figure
    st.write("FIGURE 2!!!!!!!!!!!!!!!!!!!!")
    fig2 =make_subplots(rows=1, cols=no_columns, subplot_titles=subplot_list,
                                                      horizontal_spacing= 0.02
                                                    )

    for i, measure in enumerate(measures):
        for j, flag in enumerate(flags):
            chart_data = chart_df[
                (chart_df['Flag'] == flag) &
                (chart_df['Measure'] == measure) &
                (chart_df['Tactic'] != "Grand Total")
            ].sort_values(by='row_numbers')

            text_template = "%{x:₩,.2f}" if i == 1 else "%{x:₩,.2s}"
            tickformat = ".2f" if i == 1 else "0.2s"

            fig2.add_trace(
                go.Bar(
                    y=chart_data['Tactic'],
                    x=chart_data['Value'],
                    name=flag,
                    orientation='h',
                    text=chart_data['Value'],
                    texttemplate=text_template,
                    textposition='outside',  # Changed from 'inside' to 'outside' for better readability
                    marker_color=bar_colors[flag]
                ),
                row=1, col=i+1
            )

            # Remove annotation loop (not robust for all cases)
            # ...existing code...

            # Update x-axis tick format
            if i == 1:
                fig2.update_xaxes(tickprefix="₩", tickformat=".2f", row=1, col=i+1)
            else:
                fig2.update_xaxes(tickprefix="₩", tickformat="0.2s", row=1, col=i+1)

            if i > 0:
                fig2.update_yaxes(showticklabels=False, row=1, col=i+1)
            fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
            fig2.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)

    fig2.update_yaxes(tickfont=dict(size=15), row=1, col=1)
    fig2.update_layout(
        barmode='group',
        height=800,
        plot_bgcolor=background_colors[0],
        paper_bgcolor='rgba(0,0,0,0)',
    )
    # Hide extra legends robustly
    for idx in range(2, len(fig2.data)):
        fig2.data[idx].showlegend = False
    fig2.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.6,
            font=dict(size=20),
            traceorder="reversed"
        ),
        font=dict(size=12, color="black")
    )

    for idx in range(2, 11):
        try:
            fig2['data'][idx]['showlegend'] = False
        except IndexError:
            break

    return fig2


def report_tab1(lte_fec, chart_df):
    
    #st.write("Shape", chart_df.shape)

    #if "grp_output_df" not in st.session_state:
    if chart_df.empty:
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT. GROUP ERROR", icon="⚠️")
        
    elif st.session_state.update_status=="TRUE":
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
        
    else:
        #chart_df = st.session_state['final_grp_output_df']
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
                      horizontal_spacing= 0.02)
        
        flags = [ 'OPTIM/SIM','Actual']
        
        # Define background colors for each column
        background_colors = ['#e8ebf1']
        
        # Define bar colors for each flag
        bar_colors = {
            'OPTIM/SIM': '#f5a968',
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
              
                chart_data = chart_df[(chart_df['Flag'] == flag) & (chart_df['Measure'] == measure) & (chart_df['Tactic'] != "Grand Total")].sort_values(by='row_numbers')
                fig.add_trace(
                    go.Bar(
                        y=chart_data['Tactic'],
                        x=chart_data['Value'],
                        name='Optimized' if flag == 'OPTIM/SIM' else flag,
                        orientation='h',
                        text=chart_data['Value'],
                        texttemplate=text_template,
                        textposition='auto',  
                        textfont=dict(color='black',      # readable over darker bars
                                    size=17,
                                    family='Arial',),
                        insidetextanchor='middle',
                        marker_color=bar_colors[flag]
                    ),
                    row=1, col=i+1
                )

                # Update x-axis tick format
                if i == 1:
                    fig.update_xaxes(tickprefix="₩", tickformat=".2f", row=1, col=i+1)
                else:
                    fig.update_xaxes(tickprefix="₩", tickformat="0.2s", row=1, col=i+1)

                if i > 0:
                    fig.update_yaxes(showticklabels=False, row=1, col=i+1)
                fig.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
                fig.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)

        fig.update_yaxes(tickfont=dict(size=15), row=1, col=1)
        fig.update_layout(
            barmode='group',
            height=800,
            plot_bgcolor=background_colors[0],
            paper_bgcolor='rgba(0,0,0,0)',
        )
        # Hide extra legends robustly
        for idx in range(2, len(fig.data)):
            fig.data[idx].showlegend = False
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.6,
                font=dict(size=20),
                traceorder="reversed"
            ),
            font=dict(size=12, color="black")
        )
     


# for chart2
        
       # chart_df=st.session_state['final_grp_output_df']
        # st.dataframe(chart_df[''])
        # gt_df=chart_df[ (chart_df['Tactic'] == "Grand Total")]
        
        # if lte_fec==True:
        #     subplot_list=("Spend", "ROI", "Revenue",'LTE FEC')
        #     no_columns=4
        #     measures = ['SPEND', 'ROI', 'FEC','LTE FEC']
        # else:
        #     subplot_list=("Spend", "ROI", "Revenue")
        #     no_columns=3
        #     measures = ['SPEND', 'ROI', 'FEC']

        # fig2 = make_subplots(rows=1, cols=no_columns, subplot_titles=subplot_list,
        #               horizontal_spacing= 0.02
        #             )
        
        
        # flags = ['OPTIM/SIM'] + st.session_state.final_actula_data['Flag'].unique().tolist()

        # # [actual_data_yr_filter1,actual_data_yr_filter2]
        
        # # Define background colors for each column
        # background_colors = ['#e8ebf1']
        
        # # Define bar colors for each flag
        # # Define a list of colors
        # color_palette = ['#F4D03F', '#F1948A', '#76D7C4', '#A2DFF7', '#D7BDE2']

        
        # # Create a color mapping using itertools.cycle to repeat colors if needed
        # bar_colors = {flag: color for flag, color in zip(flags, itertools.cycle(color_palette))}
        
        
            
        
        # color_discrete_sequence = ['#f5a968', '#6686a6']
        # for i, measure in enumerate(measures):
        #     for j, flag in enumerate(flags):
        #           # Sort by  column
                
        #         # Determine the texttemplate based on the column index
        #         if i == 1:  # Column 2 (ROI)
        #             text_template = "%{x:₩,.2f}"  # Format as dollars with 2 decimal places
        #         else:  # Column 1 (Spend) and Column 3 (FEC)
        #             text_template = "%{x:₩,.2s}"  # Format as '0.2s' with dollar symbol
              
        #         chart_data = chart_df[(chart_df['Flag'] == flag) & (chart_df['Measure'] == measure) & (chart_df['Tactic'] != "Grand Total")].sort_values(by='row_numbers')
        #         fig2.add_trace(
        #         go.Bar(y=chart_data['Tactic'], 
        #                x=chart_data['Value'], 
        #                name=flag, 
        #                orientation='h',
        #                text=chart_data['Value'],  # Enable value labels
        #                texttemplate=text_template,  # Apply the custom text template
        #                textposition='outside',  # Position text auto
        #                marker_color=bar_colors[flag] 
        #               ), 
        #         row=1, col=i+1
        #     )
                
        #         # Update x-axis tick format
        #         if i == 1:  # Column 2 (ROI)
        #             fig2.update_xaxes(tickprefix="₩", tickformat=".2f", row=1, col=i+1)  # Format as dollars with 2 decimal places
        #         else:  # Column 1 (Spend) and Column 3 (FEC)
        #             fig2.update_xaxes(tickprefix="₩", tickformat="0.2s", row=1, col=i+1)  # Format as '0.2s'
        
        #         if i > 0:  # Check if it's the second or third column
        #             fig2.update_yaxes(showticklabels=False, row=1, col=i+1)  # Hide y-axis labels
        #         # Update grid settings for both x-axis and y-axis
        #         fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)
        #         fig2.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1, row=1, col=i+1)

        # # Update y-axis font size for the first column
        # fig2.update_yaxes(tickfont=dict(size=15), row=1, col=1)
        
        # # Update layout for better appearance
        # fig2.update_layout(
            
        #     barmode='group',
        #     height=800,
            
        #     plot_bgcolor=background_colors[0],  # Set background color for the plot area
        #     paper_bgcolor='rgba(0,0,0,0)',  # Set transparent background for the entire subplot

        # )
        
        # #st.write(fig2['data'])
        
        # fig2['data'][2]['showlegend'] = False
        # # fig2['data'][3]['showlegend'] = False
        # # fig2['data'][4]['showlegend'] = False
        # # fig2['data'][5]['showlegend'] = False
        # try:
        #     fig2['data'][6]['showlegend'] = False
        #     fig2['data'][7]['showlegend'] = False
        #     fig2['data'][8]['showlegend'] = False
        #     fig2['data'][9]['showlegend'] = False
        #     fig2['data'][10]['showlegend'] = False
            
        # except:
        #     pass
        # fig2.update_layout(
        #     legend=dict(
        #         orientation="h",  # Horizontal orientation
        #         yanchor="bottom",
        #         y=1.1,
        #         xanchor="center",
        #         x=0.6,
        #         font=dict(
        #             size=20,
        #             # Adjust the font size as needed
        #         )
        #         ,traceorder="reversed" 
        #     )
        # )
        # fig2.update_layout(
        #     # Specify font color for text labels
        #     font=dict(
        #          size=12,color="black"
        #     )
        # )
        # with st.container( border=True): 
            
        #     st.subheader("Yearly vs OPTIM/SIM ")
            

            
        #     # st.plotly_chart(chart,use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
        #     st.divider()
        #     st.plotly_chart(fig2, use_container_width=True)

        
def report_tab2(del_df): 

    #if "Delta" not in st.session_state:
    if del_df.empty:
            st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
        
    elif st.session_state.update_status=="TRUE":
        st.warning("PLEASE START THE SIMULATION TO VIEW THE RESULT.", icon="⚠️")
        
    else:
        
        #chart_df=st.session_state['Delta']
        chart_df = del_df
        
        #st.dataframe(chart_df)
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
            text_template = "%{x:₩,.3f}" if flag == 'ROI' else "%{x:₩,.3s}"  # Format based on the flag
            colors = [bar_colors[classification] for classification in chart_data['Classification']]
    
            fig.add_trace(
                go.Bar(y=chart_data['Tactic'], 
                       x=chart_data['Measure'], 
                       orientation='h',
                       text=chart_data['Measure'],  # Enable value labels
                       texttemplate=text_template,  # Apply the custom text template
                       textposition='auto',  # Position text auto
                       textfont=dict(color='white',
                                    size=14,
                                    family='Arial',),
                       insidetextanchor='middle',
                       marker_color=colors  # Set bar color based on classification
                      ), 
                row=1, col=i+1
            )
    
            # Update x-axis tick format
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
            #st.dataframe(chart_df[['Flag', 'Tactic']])
            st.plotly_chart(fig,use_container_width=True)


def calculate_metrics(df, brand=None, tactic=None, period_val=None, period_name=None):
    row = {}
    row['Spend Change'] = df['OPTIM/SIM_SPEND'].sum() - df['Actual Spend (Selected Months)'].sum()
    row['FEC Change'] = df['OPTIM/SIM_FEC'].sum() - df['Actual FEC'].sum()
    ana_spend = df['Actual Spend (Selected Months)'].sum()
    opt_spend = df['OPTIM/SIM_SPEND'].sum()
    row['Actual ROI'] = df['Actual Profit'].sum() / ana_spend if ana_spend else 0
    row['Scenario ROI'] = df['OPTIM/SIM_PROFIT'].sum() / opt_spend if opt_spend else 0
    row['Brand'] = brand
    row['Tactic'] = tactic
    
    if period_name:
       row[period_name] = period_val
    return row


def format_roi(value):
    return f"₩ {value:.2f}" if pd.notna(value) else "-"


def report_tab3(chart_df):

    #output_df = st.session_state['output_df'].copy()
    output_df = chart_df.copy()
    #st.write("Output DF:", output_df.shape)

    numeric_cols = ['Spend Change', 'FEC Change', 'Actual ROI', 'Scenario ROI']
    # st.subheader(" Scenario Summary")
    tab1, tab2, tab3, tab4 = st.tabs(["Overall", "Quarterly", "Monthly","Detail Report",])

    with tab1:
        st.subheader("Overall Summary")
        full_rows = []
        
        main_brand = output_df['Brand'].dropna().unique()[0]
        full_rows.append(calculate_metrics(output_df, main_brand, "Grand Total"))
        
        for brand in output_df['Brand'].dropna().unique():
           df_brand = output_df[output_df['Brand'] == brand]
           for tactic in df_brand['Tactic'].dropna().unique():
               df_tac = df_brand[df_brand['Tactic'] == tactic]
               full_rows.append(calculate_metrics(df_tac, brand, tactic))
               

        df1 = pd.DataFrame(full_rows)
        
        for col in ['Spend Change', 'FEC Change']:
           df1[col] = df1[col].map(format_currency)
            
        for col in ['Actual ROI', 'Scenario ROI']:
           df1[col] = df1[col].map(format_roi)
        
        df1 = df1.rename(columns={'FEC Change': 'Revenue Change'})

        st.dataframe(df1[['Brand', 'Tactic', 'Spend Change', 'Revenue Change', 'Actual ROI', 'Scenario ROI']],
                    use_container_width=True, hide_index=True)


    with tab2:
        st.subheader("Quarterly Summary")
        
        df_q = output_df[output_df['Quarterly'].notna()].copy()
        df_q['Quarterly'] = df_q['Quarterly'].str.replace(' - ', '-').str.upper()
        full_rows = []
        
        main_brand = df_q['Brand'].dropna().unique()[0]
        full_rows.append(calculate_metrics(df_q, main_brand, "Grand Total", "", "Quarterly"))
        
        for brand in df_q['Brand'].dropna().unique():
           df_brand = df_q[df_q['Brand'] == brand]
           for tactic in df_brand['Tactic'].dropna().unique():
               df_tac = df_brand[df_brand['Tactic'] == tactic]
               for q in df_tac['Quarterly'].dropna().unique():
                   df_sub = df_tac[df_tac['Quarterly'] == q]
                   full_rows.append(calculate_metrics(df_sub, brand, tactic, q, "Quarterly"))
                   
        df2 = pd.DataFrame(full_rows)

        # Sort everything except Grand Total
        quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
        df2['Quarter_Order'] = df2['Quarterly'].str.extract(r'(Q\d)').fillna('')
        df2['Year'] = df2['Quarterly'].str.extract(r'(\d{2,4})').fillna('')
        df2['Quarter_Order'] = pd.Categorical(df2['Quarter_Order'], categories=quarter_order, ordered=True)
        
        gt_row = df2[df2['Tactic'] == 'Grand Total']
        sorted_rows = df2[df2['Tactic'] != 'Grand Total'].sort_values(by=['Brand', 'Tactic', 'Year', 'Quarter_Order'], na_position='last')
        
        df2 = pd.concat([gt_row, sorted_rows], ignore_index=True).drop(columns=['Quarter_Order', 'Year'])
        
        for col in ['Spend Change', 'FEC Change']:
           df2[col] = df2[col].map(format_currency)
        for col in ['Actual ROI', 'Scenario ROI']:
           df2[col] = df2[col].map(format_roi)

        df2 = df2.rename(columns={'FEC Change': 'Revenue Change'})
        
        st.dataframe(df2[['Brand', 'Tactic', 'Quarterly', 'Spend Change', 'Revenue Change', 'Actual ROI', 'Scenario ROI']],
                    use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Monthly Summary")
        
        df_m = output_df[output_df['Monthly'].notna()].copy()
        full_rows = []
        
        main_brand = df_m['Brand'].dropna().unique()[0]
        full_rows.append(calculate_metrics(df_m, main_brand, "Grand Total", "", "Monthly"))
        
        for brand in df_m['Brand'].dropna().unique():
           df_brand = df_m[df_m['Brand'] == brand]
           for tactic in df_brand['Tactic'].dropna().unique():
               df_tac = df_brand[df_brand['Tactic'] == tactic]
               for m in df_tac['Monthly'].dropna().unique():
                   df_sub = df_tac[df_tac['Monthly'] == m]
                   full_rows.append(calculate_metrics(df_sub, brand, tactic, m, "Monthly"))
        
        df3 = pd.DataFrame(full_rows)
        # Sort Monthly using actual datetime
        
        df3['Monthly_dt'] = pd.to_datetime(df3['Monthly'], format='%b-%y', errors='coerce')
        gt_row = df3[df3['Tactic'] == 'Grand Total']
        sorted_rows = df3[df3['Tactic'] != 'Grand Total'].sort_values(by=['Brand', 'Tactic', 'Monthly_dt'], na_position='last')
        df3 = pd.concat([gt_row, sorted_rows], ignore_index=True).drop(columns=['Monthly_dt'])
        for col in ['Spend Change', 'FEC Change']:
           df3[col] = df3[col].map(format_currency)
        for col in ['Actual ROI', 'Scenario ROI']:
           df3[col] = df3[col].map(format_roi)

        df3 = df3.rename(columns={'FEC Change': 'Revenue Change'})
        st.dataframe(df3[['Brand', 'Tactic', 'Monthly', 'Spend Change', 'Revenue Change', 'Actual ROI', 'Scenario ROI']],
                    use_container_width=True, hide_index=True)

    with tab4:
        with st.container(border=True):
            # final_Detail_Report=st.session_state['output_df'][['TARGET', 'Brand','Tactic','MONTH_YEAR','Yearly','Monthly','Quarterly','Actual Spend (Selected Months)',
            #                                                     'Actual Profit','Actual LTE Profit','OPTIM/SIM_SPEND','OPTIM/SIM_PROFIT','OPTIM/SIM_LTE_PROFIT',
            #                                                     'Actual FEC','Actual LTE FEC','OPTIM/SIM_FEC','OPTIM/SIM_LTE_FEC']]
            final_Detail_Report = output_df.copy()
            
            final_Detail_Report=final_Detail_Report.rename(columns={"MONTH_YEAR":"Week Start","OPTIM/SIM_SPEND":'Simulation Spend (Selected Months)',
                                                                    'Actual Profit':'Budget Profit',
                                                                    'Actual FEC':'Actual FEC'
                                                                    })
            
            # Define the columns for grouping and the remaining columns
            final_Detail_Report['Monthly'] = pd.to_datetime(final_Detail_Report['Week Start']).dt.strftime('%B').astype(str)
            group_cols = ['TARGET', 'Brand', 'Tactic', 'Yearly', 'Monthly', 'Quarterly']
            
            sum_cols = ['Actual Spend (Selected Months)','Budget Profit','Actual LTE Profit','Simulation Spend (Selected Months)','OPTIM/SIM_PROFIT','OPTIM/SIM_LTE_PROFIT','Actual FEC','OPTIM/SIM_FEC']
            
            # Group by the specified columns and sum the remaining columns
            final_Detail_Report = final_Detail_Report.groupby(group_cols)[sum_cols].sum().reset_index()
            
            month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            
            final_Detail_Report['Monthly'] = pd.Categorical(final_Detail_Report['Monthly'], categories=month_names, ordered=True)
            final_Detail_Report=final_Detail_Report.sort_values(['Tactic','Monthly'])
            
            col1,col2,col3,col4,col5 = st.columns(5)
            with col1:
                st.subheader("Detail Report")
                
            with col5:
                csv = convert_df(final_Detail_Report)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="Detail_report.csv",
                    mime="text/csv",
    
                )
            # Apply formatting function to the specified columns
            currency_columns = sum_cols
            
            for col in currency_columns:
                final_Detail_Report[col] = final_Detail_Report[col].map(format_currency)

            final_Detail_Report.drop(['Budget Profit', 'Actual LTE Profit', 'OPTIM/SIM_LTE_PROFIT', 'OPTIM/SIM_FEC'], axis=1, inplace=True, errors='ignore')

            final_Detail_Report = final_Detail_Report.rename(columns={'Actual FEC': 'Actual Revenue',
                                                                      'OPTIM/SIM_PROFIT': 'Optimization Profit',
                                                                      'Simulation Spend (Selected Months)': 'Optimization Spend (Selected Months)',})

            st.dataframe(final_Detail_Report,use_container_width=True,height=800 ,hide_index=True)
            
        
def draw_bar(value, max_val):
    
   if pd.isnull(value):
       return ""
       
   color = '#2E8B57' if value >= 0 else '#DC143C'
   width_pct = min((abs(value) / max_val) * 60, 50)
   label = f"₩{value/1e6:.2f}M"
   side = 'left' if value >= 0 else 'right'
   return (
       f"<div style='width:420px; margin:0 auto; height:22px; position:relative;'>"
       # bar div separately
       f"<div style='position:absolute; top:0; bottom:0; {side}:50%; transform:translateX(0); "
       f"background:{color}; height:22px; width:{width_pct}%; min-width:4px; "
       f"{'border-radius:4px 0 0 4px;' if value < 0 else 'border-radius:0 4px 4px 0;'}'></div>"
       # label div separately
       f"<div style='position:absolute; top:0; bottom:0; {side}:50%; transform:translateX(0); "
       f"display:flex; align-items:center; {side}:calc(50% + {width_pct}%);'>"
       f"<span style='font-size:13px; color:black; margin-{side}:6px;'>{label}</span>"
       f"</div>"
       f"</div>"
   )


def report_tab4(chart_df):
    df = chart_df.copy()
    df = df[df['Measure'].str.upper() == 'SPEND']

    #st.dataframe(df.head())
    
    
    has_fec = 'OPTIM/SIM_FEC' in df['Flag'].unique()
    has_ltfec = 'OPTIM/SIM_LT FEC' in df['Flag'].unique()
    
    pivot = df.pivot_table(index=['Brand', 'Tactic'], columns='Flag', values='Value', aggfunc='sum').reset_index().fillna(0)
    pivot = pivot.drop(columns=[col for col in pivot.columns if 'Flag' in str(col)], errors='ignore')
    pivot = pivot.loc[:, ~pivot.columns.str.contains('Flag')]
    pivot['Actual Spend'] = pivot.get('Actual', 0)
    pivot['Current Spend (₩M)'] = pivot['Actual Spend'].apply(lambda x: f"₩{x/1e6:.2f}M")
    
    if has_fec:
       pivot['FEC Spend'] = pivot.get('OPTIM/SIM_FEC', 0)
       pivot['Short Term Impact'] = pivot['FEC Spend'] - pivot['Actual Spend']
        
    if has_ltfec:
       pivot['LT FEC Spend'] = pivot.get('OPTIM/SIM_LT FEC', 0)
       pivot['Long Term Impact'] = pivot['LT FEC Spend'] - pivot['Actual Spend']

    #st.write("pivot df")
    st.table(pivot)
        
    impact_cols = []
    
    if has_fec: impact_cols.append('Short Term Impact')
        
    if has_ltfec: impact_cols.append('Long Term Impact')
        
    max_val = pivot[impact_cols].abs().max().max() if impact_cols else 1
    
    
    if has_fec:
       #pivot['Short Term Impact Bar'] = pivot['Short Term Impact'].apply(draw_bar)
       pivot['Short Term Impact Bar'] = pivot['Short Term Impact'].apply(lambda x: draw_bar(x, max_val))
        
    if has_ltfec:
       pivot['Long Term Impact Bar'] = pivot['Long Term Impact'].apply(lambda x: draw_bar(x, max_val))
        
    show_cols = ['Brand', 'Tactic', 'Current Spend (₩M)']
    
    if has_ltfec: show_cols.append('Long Term Impact Bar')
        
    if has_fec: show_cols.append('Short Term Impact Bar')
        
    pivot['Sort Spend'] = pivot['Actual Spend']
    display_df = pivot[show_cols + ['Sort Spend']].sort_values(by='Sort Spend', ascending=False).drop(columns='Sort Spend').rename(columns={'Short Term Impact Bar': 'Short Term Impact', 
                                                                                                                                            'Long Term Impact Bar': 'Long Term Impact'})
    #st.write(display_df)
    # === Calculate and append FEC DIFF and LT FEC DIFF Rows ===
    #df_base = st.session_state['final_grp_output_df'].copy()
    df_base = chart_df.copy()

    
    # Calculate FEC and LT FEC from Actual
    fec_ana = df_base[(df_base['Tactic'] == 'Grand Total') & 
        (df_base['Flag'] == 'Actual') & 
        (df_base['Measure'].str.upper() == 'FEC')]['Value'].sum()
    
    ltfec_ana = df_base[ (df_base['Tactic'] == 'Grand Total') & 
        (df_base['Flag'] == 'Actual') & 
        (df_base['Measure'].str.upper() == 'LTE FEC')]['Value'].sum()

    # Calculate FEC Diff
    fec_short = df_base[(df_base['Tactic'] == 'Grand Total') & 
        (df_base['Flag'] == 'OPTIM/SIM_FEC') & 
        (df_base['Measure'].str.upper() == 'FEC')]['Value'].sum() - fec_ana
    
    fec_long = df_base[(df_base['Tactic'] == 'Grand Total') &
        (df_base['Flag'] == 'OPTIM/SIM_LT FEC') &
        (df_base['Measure'].str.upper() == 'FEC')]['Value'].sum() - fec_ana

    # Calculate LT FEC Diff
    ltfec_short = df_base[(df_base['Tactic'] == 'Grand Total') &
        (df_base['Flag'] == 'OPTIM/SIM_FEC') &
        (df_base['Measure'].str.upper() == 'LTE FEC')]['Value'].sum() - ltfec_ana
    
    ltfec_long = df_base[(df_base['Tactic'] == 'Grand Total') &
        (df_base['Flag'] == 'OPTIM/SIM_LT FEC') &
        (df_base['Measure'].str.upper() == 'LTE FEC')]['Value'].sum() - ltfec_ana
    
    fec_row = {
       'Brand': 'FEC DIFF',
       'Tactic': 'FEC DIFF',
       'Current Spend (₩M)': 'FEC DIFF'
    }

    if has_fec:
        #fec_row['Short Term Impact'] = ₩{fec_short / 1e6:.2f}M"
        fec_row['Short Term Impact'] = f"₩{fec_short / 1e6:.2f}M"

    
    if has_ltfec:
        #fec_row['Long Term Impact'] = f"<div style='text-align:center; font-size:15px'>₩{fec_long / 1e6:.2f}M"
        fec_row['Long Term Impact'] = f"₩{fec_long / 1e6:.2f}M"
        

        
    
    # Create LT FEC DIFF row
    ltfec_row = {
    'Brand': 'LT FEC DIFF',
    'Tactic': 'LT FEC DIFF',
    'Current Spend (₩M)': 'LT FEC DIFF'
    }
    
    if has_fec:
        ltfec_row['Short Term Impact'] = f"₩{ltfec_short / 1e6:.2f}M"
    
    if has_ltfec:
        ltfec_row['Long Term Impact'] = f"₩{ltfec_long / 1e6:.2f}M"


    #st.write("fec_short: ", fec_short, fec_long, "ltfec_short: ", ltfec_short, ltfec_long)
    #st.write("fec_long: ", fec_long)
    #st.write("ltfec_short: ", ltfec_short, ltfec_long)
    #st.write("ltfec_long: ", ltfec_long)
    
    display_df = pd.concat([display_df, pd.DataFrame([fec_row, ltfec_row])], ignore_index=True)
    
    #st.dataframe(display_df)


def download_tab():
    if len(st.session_state.output_history_data)!=0:
        
        output=st.session_state['output_history_data']
    
        scenario_list=output['COMMENTS'].unique().tolist()
    
        download_scn_filter=st.multiselect("Selecte list of Scenario to download",["Select All"]+scenario_list,"Select All")
    
        if "Select All" in download_scn_filter:
            download_scn_filter=scenario_list
            
        downlaod_c=st.empty()
    
        multi_report_df=output[output['COMMENTS'].isin(download_scn_filter)]
        multi_report_df=multi_report_df[['TARGET','Brand', 'Tactic','MONTH_YEAR','Yearly','Monthly','Quarterly','Actual Spend (Selected Months)',
                                                           'Actual Profit','Actual LTE Profit','OPTIM/SIM_SPEND','OPTIM/SIM_PROFIT','OPTIM/SIM_LTE_PROFIT',
                                                          'Actual FEC','OPTIM/SIM_FEC']]
    
    
        multi_report_df=multi_report_df.rename(columns={"MONTH_YEAR":"Week Start","OPTIM/SIM_SPEND":'Simulation Spend (Selected Months)',
                                                                'Actual Profit':'Budget Profit',
                                                               'Actual FEC':'Actual FEC'
                                                               })
        
        #st.dataframe(multi_report_df)

        #multi_report_df['Monthly'] = pd.to_datetime(multi_report_df['Week Start']).dt.strftime('%B').astype(str)
        group_cols = ['TARGET','Brand', 'Tactic', 'Yearly', 'Monthly', 'Quarterly']
        sum_cols = ['Actual Spend (Selected Months)','Budget Profit','Actual LTE Profit','Simulation Spend (Selected Months)','OPTIM/SIM_PROFIT','OPTIM/SIM_LTE_PROFIT','Actual FEC','OPTIM/SIM_FEC']
        
        multi_report_df = multi_report_df.groupby(group_cols)[sum_cols].sum().reset_index()
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        multi_report_df['Monthly'] = pd.Categorical(multi_report_df['Monthly'], categories=month_names, ordered=True)
        multi_report_df=multi_report_df.sort_values(['Tactic','Monthly'])
        multi_report_df=multi_report_df.reset_index(drop=True)
        download = downlaod_c.download_button("🔽 Download",data=convert_df(multi_report_df), key='concated_download',file_name=str('Concated_Detail_report.csv'),
                                                                                                                                                                             mime="text/csv", type="secondary",help="Click to Download data")
    else:
        st.warning("No Data Available at selected Brand or Date Range")


def cards_html_title(header_input, input1, input2, input3,input4, wch_colour_box, wch_colour_font, fontsize1, fontsize2, iconname):
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
                            border-radius: 0px; 
                            padding-left: 12px; 
                            padding-top: 2px; 
                            padding-bottom: 2px; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);'>
                    <p class="card-text">
                        <span class="underlined-text bold-text" style='text-align: center; display: block;'>{header_input}</span>
                        <span class="bold-text">Actual      : {input2} </span>
                    </p>
                </div>"""

def report_pages():      

    output=st.session_state['output_history_data']
                
    #st.write("Under Spinner ", output.shape)
   
    selected_columns = [
            'BRAND_LIST',
            'Actual Tactic',
            'BUDGET_WEEK_START_DT',
            'Actual Spend (Selected Months)',
            'Actual Profit',
            'Actual FEC',
            'OPTIMZED_SPEND',
            'OPTIMIZED_PROFIT',
            'OPTIMIZED_FEC'
        ]

    conditions = [
        output['STATUS'] == "Iteration limit reached",
        output['STATUS'] == "Optimization terminated successfully",
        output['STATUS'] == "Positive directional derivative for linesearch",
        output['STATUS'] == "In Progress"
    ]

    choices = [
        "✅ Success",
        "✅ Success",
        "✅ Success",
        "❕ In Progress"
    ]

    #output['RESULT'] = np.select(conditions, choices, default="Failed")
    output['RESULT'] = np.where(output['STATUS'] == "Iteration limit reached", "Success", 
                                np.where(output['STATUS'] == "Optimization terminated successfully", "✅ Success",
                                         np.where(output['STATUS'] == "In Progress", "❕ In Progress", "Failed")))
    #st.write("Output Condition Initialization", output.shape)

    #st.dataframe(output.head())    
    grouped_output=output.groupby(["DATE",'COMMENTS','JOB_ID',"BRAND_LIST","PERIOD_TYPE","MONTH_LIST","CHANNEL_TYPE","TACTIC_LIST","STATUS",
                                   "OPTIM/SIM","TARGET"]).agg({'Actual Spend (Selected Months)': 'sum','Actual Profit':'sum',
                                                               'OPTIM/SIM_SPEND':'sum', 'OPTIM/SIM_PROFIT':'sum'}).reset_index()
    
    grouped_output['DATE'] = pd.to_datetime(grouped_output['DATE'], dayfirst=True, errors='coerce')
    grouped_output=grouped_output.sort_values(["DATE"],ascending=False)

    #st.dataframe(grouped_output)
            
    grouped_output_og=grouped_output

    
    col_size=[.9,.9,.9,.9,.9,.6,1.2]
    # with st.container(border=True):   
    #     cols   = st.columns(col_size)
    #     fields = ["Optim/Sim","Brand","Scenarious","Created Date","Status"]
        
    # header
    if len(grouped_output)==0:
        st.warning("No Data Available at selected Brand or Date Range")

    rows_per_page = 5
    total_rows = len(grouped_output)
    total_pages = (total_rows - 1) // rows_per_page + 1


    st.session_state.setdefault('page', 1)


    col1, spacer1, col2, spacer2, col3 = st.columns([2, 3, 1, 3, 1])

    with col1:

        # st.markdown("""
        #     <style>
        #     button[kind="primary"], button[kind="secondary"], .stButton > button {
        #         font-size: 10.5rem !important;
        #         font-weight: normal !important;
        #     }
        #     </style>
        # """, unsafe_allow_html=True)
        
        if st.button("Previous", key = "prev") and st.session_state['page'] > 1:
            st.session_state['page'] -= 1

    with col3:

        # st.markdown("""
        #     <style>
        #     button[kind="primary"], button[kind="secondary"], .stButton > button {
        #         font-size: 10.5rem !important;
        #         font-weight: normal !important;
        #     }
        #     </style>
        # """, unsafe_allow_html=True)
        
        if st.button("Next", key = "next") and st.session_state['page'] < total_pages:
            st.session_state['page'] += 1

    with col2:
        #st.write(f"Page {st.session_state['page']} of {total_pages}")

        st.markdown(rf"$\textrm{{\normalsize Page\ {st.session_state['page']}\ of\ {total_pages}}}$")

    with st.container(border=True):   
        cols   = st.columns(col_size)
        fields = ["Optim/Sim","Brand","Scenarios","Created Date","Status"]
        
    for col, field in zip(cols, fields):
        col.markdown(rf"$\textrm{{\Large {field}}}$")
            
    # Calculate start and end indices
    start_idx = (st.session_state['page']  - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    
    # Slice the DataFrame
    paged_df = grouped_output.iloc[start_idx:end_idx]
    #paged_df

    with st.container(border=False):

        #st.write(col_size)
        for index, row in paged_df.iterrows():
            #st.write(row)
            #st.write("Iter Index: ", index)
            with st.container(border=True):
                
                #col1, col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13= st.columns(col_size)
                col1, col2,col3,col4,col11,col12,col13= st.columns(col_size)
                
                if row[9]=="SIM":
                    col1.markdown(f"<span style='font-size:15px;'>🎮 Sim - {row[2]}</span>", unsafe_allow_html=True)
                elif row[9]=="OPTIM":
                    col1.markdown(f"<span style='font-size:18px;'>⚙️ Optim - {row[2]}</span>", unsafe_allow_html=True)
                col2.markdown(f"<span style='font-size:18px;'>{row[3]}</span>", unsafe_allow_html=True)
                col3.markdown(f"<span style='font-size:18px;'>{row[1]}</span>", unsafe_allow_html=True)
                col4.markdown(f"<span style='font-size:18px;'>{row[0]}</span>", unsafe_allow_html=True)
                #col5.write(row[4])
                #col6.write(row[6])
                ## col6.write(row[8])
                #col7.markdown("₩ {:,.0f}".format(round(row[10])))
                #col8.markdown("₩ {:,.0f}".format(round(row[12])))
                #col9.markdown("₩ {:,.0f}".format(round(row[11])))
                #col10.markdown("₩ {:,.0f}".format(round(row[13])))
                status = row[8]
                
                if status == "Iteration limit reached":
                    #col11.write("✅ Success")
                    col11.markdown(f"<span style='font-size:18px;'>✅ Success</span>", unsafe_allow_html=True)

                elif status == "Optimization terminated successfully":
                    col11.write("✅✅ Success")
                    
                elif status == "Positive directional derivative for linesearch" or status ==  "More than 3*n iterations in LSQ subproblem":
                #     col11.write("✅ Success")
                # elif status == "More than 3*n iterations in LSQ subproblem":
                    # col11.write("⚠️ Warning")
                    col11.markdown("""<span style=" cursor:pointer;" title="Warning: The Max Bound allowed in Optimization cannot achieve the designated FEC Hit Target. 
                    Will need to increase Max Bound Constraints to reach desired FEC Target!">⚠️ Warning</span>""", unsafe_allow_html=True)
                
                elif status == "In Progress":

                    dt = row[0] if isinstance(row[0], (datetime, pd.Timestamp)) else parse(row[0], dayfirst=True)

                    #if datetime.now() - datetime.strptime(row[0], "%Y-%m-%d %H:%M") > timedelta(hours=1):
                    if datetime.now() - dt > timedelta(hours=1):
                        #col11.write("🛑 Failed")
                        col11.markdown(f"<span style='font-size:18px;'>🛑 Failed</span>", unsafe_allow_html=True)
                    else:
                        col11.write("❕ In Progress")
                    
                elif status == "Completed":
                    col11.write("✅ Completed")  
                    
                else:
                    col11.write("🛑 Failed")# col10.write(row[8])
                # col11.write("✅ Completed")
                placeholder = col12.empty()

                show_more   = placeholder.button("📃 View", key=index, type="primary", use_container_width = True)
                with col13:
                    b1,b2=st.columns(2)
                    
                    downlaod_b = b1.empty()
                    dwl=output[output['JOB_ID']==row[2]]
                    dwl=dwl.rename(columns={"MONTH_YEAR":"Week Start","OPTIM/SIM_SPEND":'Simulation Spend (Selected Months)',
                                                                'Actual Profit':'Budget Profit',
                                                               'Actual FEC':'Actual FEC'
                                                               })
                    sum_cols = ['Actual Spend (Selected Months)','Budget Profit','Actual LTE Profit','Simulation Spend (Selected Months)','OPTIM/SIM_PROFIT','OPTIM/SIM_LTE_PROFIT','Actual FEC','OPTIM/SIM_FEC']
                    group_cols = ['TARGET','Brand', 'Tactic', 'Yearly', 'Monthly', 'Quarterly']
                    dwl = dwl.groupby(group_cols)[sum_cols].sum().reset_index()

                    download = downlaod_b.download_button("🔽 Data",data=convert_df(dwl), key=index*.1,file_name=str(row[3])+"_"+str(row[2]) + '_Detail_report.csv'
                ,
                    mime="text/csv", type="secondary",help="Click to Download data", use_container_width = True)
                    
                    with b2:
                        delete_button=st.button("🗑️ Delete", key=index**3+0.01,help="Click to Delete", use_container_width = True)
                        if delete_button:
                            with st.spinner(""):
                    #             if row[9]=="SIM":
                    #                 delete=session.sql(f"Delete from ANALYTICS.UMM_OPTIM.SONIC_SIMULATION_TABLE where USER_EMAIL='{st.experimental_user.email}' and JOB_ID='{row[2]}'" ).collect()
                                
                    #             elif row[9]=="OPTIM":
                    #                 delete=session.sql(f"Delete from ANALYTICS.UMM_OPTIM.SONIC_OPTIMIZATION_TABLE where USER_EMAIL='{st.experimental_user.email}' and JOB_ID='{row[2]}'" ).collect()
                               
                    #             else:
                    #                 st.error("🚧")
                    #             # delete=session.sql(f"Delete from ANALYTICS.UMM_OPTIM.SONIC_SIMULATION_TABLE where USER_EMAIL='{user_email}' and JOB_ID='{row[2]}'" ).collect()
                                
                    #             list_of_brands = get_snowflake_Data()
                    #             st.write("Line 816")
                    #             update_actaul_compare_data()
                    #             st.rerun()
                                
                    # st.write(is_disabled = row[2] in (pd.DataFrame(session.sql("select * from ANALYTICS.UMM_OPTIM.PROD_SONIC_SHARED_OUTPUT_TABLE").collect())['SHARED_JOBID'].tolist()))
                    # is_disabled = row[2] in (pd.DataFrame(session.sql("select * from ANALYTICS.UMM_OPTIM.PROD_SONIC_SHARED_OUTPUT_TABLE").collect())['SHARED_JOBID'].tolist())
                                pass
                    is_disabled = True

                    if 'checkbox_checked' not in st.session_state:
                        st.session_state.checkbox_checked = False
                        
                    # s_b3=b3.empty()

                    # def shared(row_key, is_disabled):
                    #     return s_b3.checkbox("Share", key=row_key, disabled=is_disabled, value=is_disabled)
                        
                    # if is_disabled==True:
                    #     try:
                    #         s_b3.button("✅ Shared", key=int(row[2])**2,help="This Scenario is Shared", disabled=True)
                    #     except:
                    #         pass
                    # else:
                    #     share = shared(row[2], is_disabled)
                    # # else:
                    # #     share=st.checkbox("",key=row[2])
                    #     if share and is_disabled==False:
                    #         s_df=pd.DataFrame([[row[2]]],columns=['SHARED_JOBID'])
                            
                    #         session.create_dataframe(s_df).write.mode("append").save_as_table("ANALYTICS.UMM_OPTIM.PROD_SONIC_SHARED_OUTPUT_TABLE")
                    #         # st.session_state.checkbox_checked = True
                    #         # st.rerun()
                    #         s_b3.button("✅ Shared", key=int(row[2])**2,help="This Scenario is Shared",  disabled=True)
                            
                with st.container():

                    ouut_df, del_df = report_generate(output[output['JOB_ID']==row[2]])
                    #chart_df=st.session_state['final_grp_output_df']
                    chart_df = ouut_df.copy() 

                    #CHANGE HERE
                    # ses_col, test_col = st.columns(2)
                    # with ses_col:
                    #     st.dataframe(st.session_state.final_grp_output_df.head(2))
                    # with test_col:
                    #     st.dataframe(chart_df.head(2))
                    
                    gt_df=chart_df[ (chart_df['Tactic'] == "Grand Total")]
                    #st.dataframe(chart_df)
                    
                    with st.container():
                        iconname = "fas fa-asterisk"
                        lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
                        lte_fec=False
                        
                        st.session_state.update_status="FALSE"
                        
                        if lte_fec==True:
                            col1,col2,col3,col4=st.columns(4)
                        else:
                            col1,col2,col3=st.columns(3)
                                    
                        target_list=sorted(chart_df["TARGET"].dropna().unique(), key=["SIM","OPTIM", "Revenue", "LT FEC"].index)
                        
                        with col1:
                                    
                            O_Spend=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'SPEND')].reset_index()['Value'][0]
                            html_output = cards_html_title("Spend",format_currency(0), format_currency(O_Spend), str(round(abs((0) - 1)*100,2))+"%",format_currency(0-O_Spend),'#d1e5f0',(21, 73, 128), 18, 20, "example-icon", )
                                
                            st.markdown(lnk + html_output, unsafe_allow_html=True)
                            
                            for target_name in target_list:
                                S_Spend=gt_df[(gt_df['Flag'] == 'OPTIM/SIM') & (gt_df['TARGET'] == target_name) & (gt_df['Measure'] == 'SPEND')].reset_index()['Value'][0]
                                
                                
                                html_output = cards_html(target_name,format_currency(S_Spend), format_currency(O_Spend), str(round(abs((S_Spend/O_Spend) - 1)*100,2))+"%",format_currency(S_Spend-O_Spend),'#d1e5f0',(21, 73, 128), 18, 20, "example-icon", )
                                
                                st.markdown(lnk + html_output, unsafe_allow_html=True)
                            
                        with col2:
                            
                            O_ROI=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'ROI')].reset_index()['Value'][0]
                           
                            html_output = cards_html_title("ROI","₩"+str(0), "₩"+str(O_ROI), str(round(abs(0)-1,2))+"%","₩"+str(round((0),2)),'#e7d4e8', (78, 27, 101), 18, 20, "example-icon", )
                                
                            st.markdown(lnk + html_output, unsafe_allow_html=True)
                            
                            for target_name in target_list:
                                S_ROI=gt_df[(gt_df['Flag'] == 'OPTIM/SIM') & (gt_df['TARGET'] == target_name) & (gt_df['Measure'] == 'ROI')].reset_index()['Value'][0]
                               
                                html_output = cards_html(target_name,"₩"+str(S_ROI), "₩"+str(O_ROI), str(round(abs(S_ROI/O_ROI)-1,2))+"%","₩"+str(round((S_ROI-O_ROI),2)),'#e7d4e8', (78, 27, 101), 18, 20, "example-icon", )
                                
                                st.markdown(lnk + html_output, unsafe_allow_html=True)
                            
                        with col3:
                                              
                            O_FEC=gt_df[(gt_df['Flag'] == 'Actual') & ((gt_df['Measure'] == 'FEC') | (gt_df['Measure'] == 'Revenue'))].reset_index()['Value'][0]
                            
                            html_output = cards_html_title("Revenue",format_currency(0),format_currency(O_FEC),str(round(abs((0) - 1)*100,2))+"%" ,format_currency(0),'#d9f0d3', (44, 118, 130), 18, 20, "example-icon", )
                                
                            st.markdown(lnk + html_output, unsafe_allow_html=True)
                            
                            for target_name in target_list:
                                S_FEC=gt_df[(gt_df['Flag'] == 'OPTIM/SIM') & (gt_df['TARGET'] == target_name) & ((gt_df['Measure'] == 'FEC') | (gt_df['Measure'] == 'Revenue'))].reset_index()['Value'][0]
                                
                                html_output = cards_html(target_name,format_currency(S_FEC),format_currency(O_FEC),str(round(abs((S_FEC/O_FEC) - 1)*100,2))+"%" ,format_currency(S_FEC-O_FEC),'#d9f0d3', (44, 118, 130), 18, 20, "example-icon", )
                                
                                st.markdown(lnk + html_output, unsafe_allow_html=True)
                        
                        if lte_fec==True:
                            with col4:
                                              
                                O_FEC=gt_df[(gt_df['Flag'] == 'Actual') & (gt_df['Measure'] == 'LTE FEC')].reset_index()['Value'][0]
                                html_output = cards_html_title('LTE FEC',format_currency(0),format_currency(O_FEC),str(round(abs((0) - 1)*100,2))+"%" ,format_currency(0),'#f0d3d9', (102, 51, 51), 18, 20, "example-icon", )
                                    
                                st.markdown(lnk + html_output, unsafe_allow_html=True)
                                
                                for target_name in target_list:
                                    S_FEC=gt_df[(gt_df['Flag'] == 'OPTIM/SIM') & (gt_df['TARGET'] == target_name) & (gt_df['Measure'] == 'LTE FEC')].reset_index()['Value'][0]
                                    
                                    html_output = cards_html(target_name,format_currency(S_FEC),format_currency(O_FEC),str(round(abs((S_FEC/O_FEC) - 1)*100,2))+"%" ,format_currency(S_FEC-O_FEC),'#f0d3d9', (102, 51, 51), 18, 20, "example-icon", )
                                    
                                    st.markdown(lnk + html_output, unsafe_allow_html=True)
                        st.markdown("")
                        
                        # tab1, tab2,tab3 = st.tabs(["📊Comparison Chart","📈 FEC Delta Chart", "🛢 Detail Report"])
                    
                        # css = '''
                        #     <style>
                        #         .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        #             font-size: 1.3rem;
                                   
                        #         }
                        #     </style>
                        # '''
                    
                        # st.markdown(css, unsafe_allow_html=True)
                    
                if show_more:
                        with st.container():
    
    
                            placeholder.button("❌ Hide", help="Hide the report", key=str(index)+"_")
                        
                            lte_fec=False
                            
                            #tab1, tab2,tab3, tab4 = st.tabs(["📊Comparison Chart","📈 Revenue Delta Chart", "🛢 Scenario Summary", "🎯 Impact"])
                            tab1, tab2,tab3 = st.tabs(["📊Comparison Chart","📈 Revenue Delta Chart", "🛢 Scenario Summary"])
                        
                            css = '''
                                <style>
                                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                                        font-size: 1.3rem;
                                       
                                    }
                                </style>
                            '''
                            st.markdown(css, unsafe_allow_html=True)

                            #st.write("Shape of chart_df", chart_df.shape)
        
                            with tab1.subheader(" "):
                                report_tab1(lte_fec, chart_df)
        
                            with tab2.subheader(" "):
                                #st.dataframe(del_df)
                                report_tab2(del_df)
        
                            with tab3.subheader(" "):
                                report_tab3(output[output['JOB_ID']==row[2]])
                            
                            # with tab4.subheader(" "):
                            #     report_tab4(chart_df)

                
                                            



def main():
    


    st.set_page_config(layout="wide", initial_sidebar_state = 'collapsed')

    csss="""
        <style>
            [data-testid=stSidebarContent] {
                background-color: #AEC0DA;

            }
            [data-testid=stNotification] {
                background-color: #b8881e;
                color:black;
                 display: flex;
                # justify-content: center; /* Horizontally center */
                # align-items: center;display: grid;
                # place-items: center;

            }
            
            div.st-emotion-cache-qbiigm.e1wguzas3
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
            div.st-emotion-cache-14teyp2.e1f1d6gn0
            {
            background-color: #ffffff;
            
            
            color: black;
            }
            button.st-emotion-cache-1cp8me5.ef3psqc12
            {
            background-color: #ffffff;
            padding: 5px;

      
            color: black;
            }
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
    
    col1, col2,col3 = st.columns([0.04,0.6,0.1], gap="small")
    with col1:
        st.image(
                "https://www.c5i.ai/wp-content/themes/course5iTheme/new-assets/images/c5i-primary-logo.svg",
                width=60, # Manually Adjust the width of the image as per requirement
            )
    with col2:
        st.header('Optim/Sim Output Data',divider=True)
    
    with col3:
        refresh=st.button("🔄 Refresh",help="Click to refresh History data")
        if refresh:
            st.rerun()

    if "output_history_data" not in st.session_state:
        #st.write("Output history test")
        get_snowflake_Data()
        brand_list = st.session_state.output_history_data['BRAND_LIST'].unique().tolist()
        st.session_state['output_history_data']=st.session_state.output_history_data[st.session_state.output_history_data['BRAND_LIST'].isin(brand_list)]
        st.session_state['output_history_data']=st.session_state.output_history_data

        update_actaul_compare_data(brand_list)

    hist_brand_filter, date_filter, Yr_filter, Month_filter, actual_data_yr_filter1, actual_data_yr_filter2, Apply_hist_filter = sidebarcontent()
    
    if Apply_hist_filter:
        get_snowflake_Data(date_filter, actual_data_yr_filter1) #1758
        
        #st.write("Output History Data", st.session_state['output_history_data'].shape)
        
        st.session_state['output_history_data']=st.session_state.output_history_data[st.session_state.output_history_data['BRAND_LIST'].isin(hist_brand_filter)]
        # st.dataframe(st.session_state.actual_data.head(10))
        #st.dataframe(st.session_state.actual_data.head())


        #actual_data=st.session_state.actual_data[st.session_state.actual_data['BRAND'].isin(hist_brand_filter)]

        actual = st.session_state.actual_data
        actual.rename(columns={"Brand": "BRAND"}, inplace=True) 

        st.session_state.actual_data_yr_filter1=actual_data_yr_filter1
        st.session_state.actual_data_yr_filter2=actual_data_yr_filter2
        st.session_state['page'] = 1
        update_actaul_compare_data(hist_brand_filter, actual_data_yr_filter1, actual_data_yr_filter2)    

    # if "test" not in st.session_state:
    # #if "test" in st.session_state:
    
    #     #get_snowflake_Data(date_filter)
        
    #     st.session_state['output_history_data']=st.session_state.output_history_data[st.session_state.output_history_data['BRAND_LIST'].isin(hist_brand_filter)]
    #     st.session_state['output_history_data']=st.session_state.output_history_data
        
    #         # [st.session_state.output_history_data['MONTH_LIST']==concat_month_list]
    #     #st.session_state['output_history_data']=st.session_state['output_history_data'][(st.session_state['output_history_data']['DATE_only'] >= date_filter[0]) & (st.session_state['output_history_data']['DATE_only'] <= date_filter[1])]
    #     st.session_state.test="test"
    #     update_actaul_compare_data()

    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.5rem !important;
            font-weight: lighter !important;
        }
        </style>
    """, unsafe_allow_html=True)
    

    report,Download=st.tabs(["📚 Report","🔽 Download"])
    
    with Download:
        with st.container(border=True):
            download_tab()

    with report:
        if len(st.session_state.output_history_data)!=0:
            
            with st.spinner("Please wait - Fetching the data"):
                
                report_pages()
                
        else:
            st.warning("No Data available - please check the filter or run the scenario")

if __name__ == "__main__":
    main()