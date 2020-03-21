# %%
'''Reduce the disproportionate impact on different scales numbers'''

# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd 

# %%
encoded_df = pd.read_csv('./Resouces/loans_data_encoded.csv')
encoded_df.head()

# %%
# instantiate a standardscaler model
rescaler = StandardScaler()
# fit_transform, alternative sequentially with fit() and transform()
rescaled_ndarray = rescaler.fit_transform(encoded_df)
rescaled_ndarray

# %%
import numpy as np 
# validate if the mean is 0, and std is 1
print(np.mean(rescaled_ndarray[ : , 0 ])) # first column
print(np.mean(rescaled_ndarray[ : , 1 ])) # second column
print(np.std(rescaled_ndarray[ : , 3 ])) # third column

# %%
rescaled_ndarray.ravel()

# %%
# rebuild a standardized dataframe
trial_df = pd.DataFrame(rescaled_ndarray, columns = ['amount', 'term', 'age', 'bad','month_num',
        'education_Bachelor', 'education_High School or Below', 'education_Master or Above', 'education_college', 'gender_female', 'gender_male'])
trial_df
# %%
trial_df['amount'].mean()


# %%
