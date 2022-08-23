import pandas as pd
import string
import re
import numpy as np
from copy import deepcopy

'''
The strategy was to anticipate the strings and expressions that would cause two unique company names to appear 
representing the same company. The risk of aggressively culling company names is the what are in fact two 
separate companies will ultimately have the same canonical name. Thus the challenge was the strike the right 
balance between reducing the unique number of canonical names, while ensuring that each canonical name in 
fact represents one unique company, and you're not reducing two different companies to the same canonical name.

This entailed attention to whitespace, case, punctuation, and generic expressions that don't usually 
differentiate unique companies. 
'''

def remove_punctuation(df,verbose,num_unique=None):
    df['RAW_NAME']=df['RAW_NAME'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    if verbose:
        print(f"Removing punctuation modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def strip_whitespace(df,verbose,num_unique=None):
    df['RAW_NAME']=df['RAW_NAME'].apply(lambda x: x.strip())
    if verbose:
        print(f"Removing leading and trailing whitespace modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def make_lower_case(df,verbose,num_unique=None):
    df['RAW_NAME']=df['RAW_NAME'].str.lower()
    if verbose:
        print(f"Making text lower case operation modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def remove_generic_expressions(df,verbose,num_unique=None):
    generic_words=['the ',r' and associates',r' llc$', r' llp$',r' lp$',r' ltd$',r' group$',r' partners$',r' ventures$',r'inc$',r' company$',r' incorporated$',
                     r' limited$',r' capital$',r' consulting$',r' gmbh$',r' advisors$',r' consultants$',r' labs$',r' investments$',
                     r' technologies$', r' co$', r' associates$',r' pllc$',r' technology$',r' solutions$', r' services$', r' pc$',
                     r' international$', r' corporation$', r' partnership$',r' corp$', r' holdings$','llc ']
    for i in generic_words:
        df['RAW_NAME']=df['RAW_NAME'].str.replace(i,'',regex=True)
        if verbose:
            print(f"Removing {i} modified {num_unique-df.nunique().values[0]} company names")
            num_unique=df.nunique().values[0]
    return df,num_unique

def apply_remove_aliases(df,verbose,num_unique=None):
    def remove_aliases(s):
        p=r' aka .*'
        if re.search(p,s):
            start_idx=re.search(p,s).start()
            return s[:start_idx]
        return s
    df['RAW_NAME']=df['RAW_NAME'].apply(remove_aliases)
    if verbose:
        print(f"Removing alias expressions modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def apply_remove_powered_by(df,verbose,num_unique=None):
    def remove_powered_by(s):
        p=r' powered by .*'
        if re.search(p,s):
            start_idx=re.search(p,s).start()
            return s[:start_idx]
        return s
    df['RAW_NAME']=df['RAW_NAME'].apply(remove_powered_by)
    if verbose:
        print(f"Removing 'powered by' expressions modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def remove_multiple_whitespace(df,verbose,num_unique=None):
    df['RAW_NAME']=df['RAW_NAME'].apply(lambda mystring:' '.join(mystring.split()))
    if verbose:
        print(f"Removing multiple whitespaces modified {num_unique-df.nunique().values[0]} company names")
        num_unique=df.nunique().values[0]
    return df,num_unique

def make_results_df(df, original_df,verbose):
    original_df=original_df.rename(columns={'Skillbox':'RAW_NAME'})
    res=pd.concat([original_df,df.rename(columns={'RAW_NAME':'CANONICAL_NAME'})],axis=1)
    if verbose:
        print(f"From {res['RAW_NAME'].nunique()} unique company names, there are now {res['CANONICAL_NAME'].nunique()} Canonical names")
    return res

def create_canonical_names(filepath_in,filepath_out,verbose=True):
    '''
    filepath_in: where the CSV is stored and will be read from
    filepath_out: where the resulting dataframe with columns 'RAW_NAME' and 'CANONICAL_NAME' will be written to
    If verbose==True, after each operation it will print the number of cases modified by the operation
    '''
    original_df=pd.read_csv(filepath_in)
    df=deepcopy(original_df)
    df_renamed=df.rename(columns={'Skillbox':'RAW_NAME'})
    
    # To avoid future errors from missing values
    df_no_missing_values=df_renamed.fillna('')
    if verbose:
        num_unique=df_no_missing_values.nunique().values[0]
        print(f"Number of unique company names: {num_unique}")
    else:
        num_unique=None
        
    # Remove any punctuation, as a period or comma is highly unlikely to demarcate a different company
    df_no_punctuation,num_unique=remove_punctuation(df_no_missing_values,verbose,num_unique)

    # Remove leading and trailing whitespace
    df_no_whitespace,num_unique=strip_whitespace(df_no_punctuation,verbose,num_unique)
    
    # Convert all company names to lower case
    df_lower_case,num_unique=make_lower_case(df_no_whitespace,verbose,num_unique)

    # We are going to remove these generic words that are likely to cause duplicates of the same company with different names
    df_removed_generic_expressions,num_unique=remove_generic_expressions(df_lower_case,verbose,num_unique)
    
    # Remove leading and trailing whitespace again as removing some expressions led to ending and beginning whitespace
    df_no_whitespace2,num_unique=strip_whitespace(df_removed_generic_expressions,verbose,num_unique)
    
    # If a company names as 'aka _____', remove aka and any text following it
    df_removed_aliases,num_unique=apply_remove_aliases(df_no_whitespace2,verbose,num_unique)
    
    # If a company name has the expression 'powered by ____', remove the expression
    df_removed_powered_by,num_unique=apply_remove_powered_by(df_removed_aliases,verbose,num_unique)
    
    # Deal with company names that have multiple whitespaces where there should be one
    df_removed_multiple_whitespace,num_unique=remove_multiple_whitespace(df_removed_powered_by,verbose,num_unique)
    
    # Rename column names to differentiate original from canonical company names and join in a single dataframe    
    results_df=make_results_df(df_removed_multiple_whitespace,original_df,verbose)
    
    # Write results as a CSV to parameter filepath_out
    results_df.to_csv(filepath_out,index=False)

    return

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate canonical company names')
    parser.add_argument('filepath_in', default='company-names.csv', nargs='?',help='file location for the \
        company names CSV to be read into a dataframe')
    parser.add_argument('filepath_out', default='company-canonical-names.csv', nargs='?',help='file location for the \
        resulting dataframe with raw and canonical names to be written to as a csv')
    parser.add_argument('verbose', choices=['True', 'False'],default=True, help="Determines whether or not to\
        print verbose content. Either 'True' or 'False' ")
    args = parser.parse_args()
    args=vars(args)
    args['verbose']=bool(args['verbose'])

    create_canonical_names(**args)