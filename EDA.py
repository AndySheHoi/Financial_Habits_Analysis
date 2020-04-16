import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


class EDA:
    
# =============================================================================
# Exploratory Data Analysis
# =============================================================================
    def eda(df):
        df.head(5) # Viewing the Data
        df.columns
        df.describe() # Distribution of Numerical Variables
        
       
        # Removing NaN
        df.isna().any()
        df.isnull().sum()
        df = df[pd.notnull(df['age'])]
        df = df.drop(columns = ['credit_score', 'rewards_earned'])
        
        
        ## Histograms
        df2 = df.drop(columns = ['user', 'churn'])
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle('Histograms of Numerical Columns', fontsize=20)
        for i in range(1, df2.shape[1] + 1):
            plt.subplot(6, 5, i)
            f = plt.gca()
            f.axes.get_yaxis().set_visible(False)
            f.set_title(df2.columns.values[i - 1])
        
            vals = np.size(df2.iloc[:, i - 1].unique())
            
            plt.hist(df2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
        ## Pie Plots
        df2 = df[['housing', 'is_referred', 'app_downloaded',
                            'web_user', 'app_web_user', 'ios_user',
                            'android_user', 'registered_phones', 'payment_type',
                            'waiting_4_loan', 'cancelled_loan',
                            'received_loan', 'rejected_loan', 'zodiac_sign',
                            'left_for_two_month_plus', 'left_for_one_month']]
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle('Pie Chart Distributions', fontsize=20)
        for i in range(1, df2.shape[1] + 1):
            plt.subplot(4, 4, i)
            f = plt.gca()
            f.axes.get_yaxis().set_visible(False)
            f.set_title(df2.columns.values[i - 1])
           
            values = df2.iloc[:, i - 1].value_counts(normalize = True).values
            index = df2.iloc[:, i - 1].value_counts(normalize = True).index
            plt.pie(values, labels = index, autopct='%1.1f%%')
            plt.axis('equal')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
        ## Exploring Uneven Features
        df[df2.waiting_4_loan == 1].churn.value_counts()
        df[df2.cancelled_loan == 1].churn.value_counts()
        df[df2.received_loan == 1].churn.value_counts()
        df[df2.rejected_loan == 1].churn.value_counts()
        df[df2.left_for_one_month == 1].churn.value_counts()
        
        
        ## Correlation with Response Variable
        df2.drop(columns = ['housing', 'payment_type',
                                 'registered_phones', 'zodiac_sign']
            ).corrwith(df.churn).plot.bar(figsize=(20,10),
                      title = 'Correlation with Response variable',
                      fontsize = 15, rot = 45,
                      grid = True)
        
        ## Correlation Matrix
        sn.set(style="white")
        
        # Compute the correlation matrix
        corr = df.drop(columns = ['user', 'churn']).corr()
        
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(18, 15))
        f.suptitle("Correlation Matrix Before RFE", fontsize = 40)
        # Generate a custom diverging colormap
        cmap = sn.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        # Removing Correlated Fields
        df = df.drop(columns = ['app_web_user'])
        
        return df