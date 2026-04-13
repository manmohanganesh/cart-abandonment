import pandas as pd
import numpy as np

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()

    #total pages viewed
    df['Total_pages']=(
        df['Administrative']
        + df['Informational']
        + df['ProductRelated']
    )

    #Total time spent
    df['total_duration']= (
        df['Administrative_Duration']
        + df['Informational_Duration']
        + df['ProductRelated_Duration']
    )

    df['engagement_score'] = df['Total_pages']*df['total_duration']

    df['avg_time_per_page'] = df['total_duration']/df['Total_pages']

    df['is_low_engagement']=( (df['Total_pages']<3) & (df['total_duration']<60) ).astype(int)

    #Simulated scrool depth (0 to 1)
    np.random.seed(42)
    df['scroll_depth'] = np.random.uniform(0.2,1.0,size=len(df))
    
    # Price sensitivity (proxy using PageValues)
    df['price_sensitivity'] = 1/(df['PageValues']+1)

    #Session stage (simulate early vs late session)
    df['session_stage']=pd.cut(
        df['Total_pages'],
        bins=[-1,3,10,float("inf")],
        labels=['early','mid','late'],
        )
    return df