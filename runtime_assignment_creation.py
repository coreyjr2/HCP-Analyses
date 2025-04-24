import pandas as pd
method_list = pd.read_csv('DR_FS_method_list.csv')

#Make dataframe with models to Run

# Headers should be ['Method','split_index','model']

methods = list(method_list['0'])

splits = [x for x in range(10)]

out_dict = {"method":[],"split":[]}

for split in splits:
    for method in methods:
        out_dict['method'].append(method)
        out_dict['split'].append(split)
        
out_df = pd.DataFrame.from_dict(out_dict)

len(out_df)

out_df.to_csv('array_assignment.csv', index=False)