'''
Created on Dec. 23, 2021

@author: cefect

dump a bunch of csvs into a xls table
'''
data_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\figueiredo2018\dmdb'
out_fp = r'C:\LS\09_REPOS\01_COMMON\misc\outs\csv_dump.xls'

import os
import pandas as pd



assert os.path.exists(data_dir)



#get all csvs in the folder
fns = [e for e in os.listdir(data_dir) if e.endswith('.csv')]
fp_d = {os.path.splitext(e)[0]:os.path.join(data_dir, e) for e in fns}

#load as frames
meta_d = dict()
df_d = dict()
for tabNm, fp in fp_d.items():
    name = tabNm.replace('db_', '')
    df = pd.read_csv(fp)
    
    print('loaded \'%s\' as %s'%(name, str(df.shape)))
        
    df_d[name] = df
    
    meta_d[name] = {'tabNm':tabNm, 'fp':fp, 'shape':str(df.shape), 'columns':str(df.columns.tolist())}

#add back summary
df_d = {**{'_smry':pd.DataFrame.from_dict(meta_d).T.reset_index().rename(columns={'index':'name'})},
        **df_d}

#dump into excel
#write a dictionary of dfs
with pd.ExcelWriter(out_fp) as writer:
    for tabnm, df in df_d.items():
        df.to_excel(writer, sheet_name=tabnm, index=False, header=True)
        
print('wrote %i sheets to \n    %s'%(len(df_d), out_fp))
