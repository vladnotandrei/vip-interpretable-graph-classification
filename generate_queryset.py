from dataset import load_mutagenicity_dataset
from mutagenicity_utils import raw_to_nx, nx_to_rdkit, apply_all_fragments_dynamic, get_all_fragment_func_names
import argparse
from tqdm import tqdm
import pandas as pd

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    return args

def main(args):
    dataset = load_mutagenicity_dataset(args.dataset_root)
    df = pd.DataFrame(columns=get_all_fragment_func_names())
    for data in tqdm(dataset):
        G = raw_to_nx(data)
        mol = nx_to_rdkit(G)
        frag_counts_dict = apply_all_fragments_dynamic(mol)  # dict {key=<frag_func_name>: val=<count>}
        df.loc[len(df)] = pd.Series(frag_counts_dict)

    df = df.loc[:, (df != 0).any()]  # Remove columns that have only entries of 0 (i.e. all fragments that appear nowhere in the dataset)
    # df.columns = [col.replace('fr_', '') for col in df.columns]  # Modify column names by replacing all occurneces of 'fr_' with an empty string
    queryset_df = df.melt(var_name='frag_name', value_name='count').drop_duplicates()
    queryset_df.to_csv(f'{args.save_dir}/rdkit_queryset.csv', index=False)
    

if __name__ == '__main__':
    args = parseargs()
    main(args)