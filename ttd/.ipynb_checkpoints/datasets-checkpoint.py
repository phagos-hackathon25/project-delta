import pandas as pd
import numpy as np
from Bio import SeqIO, Seq
from collections import defaultdict
import random
import torch

import pyrodigal

import os
from pathlib import Path

traits_of_interest = ["endolysin", "holin", "Rz-like spanin", "tail fiber protein",
                      "tail spike protein", "capsular depolymerase", 
                      "baseplate protein",
                      "tail sheath", "tail tube", "tail length tape measure protein",
                      "terminase large subunit", "terminase small subunit", 
                      "head-tail adaptor", "head maturation protease", "head closure"]

def load_dataset_boeckaerts():
    """
    Load phage proteins from dataset PHL-Klebsiella.
    """
    boeckaerts_phages = "/home/ec2-user/repos/project-delta/data/phages_genomes"
    
    sequences = []
    sequences_id = []
    for ff_path in Path(boeckaerts_phages).glob("*.fasta"):
        for record in SeqIO.parse(ff_path, "fasta"):
            gene_finder = pyrodigal.GeneFinder(meta=True)
            sid = record.id
    
            for i, pred in enumerate(gene_finder.find_genes(bytes(record.seq))):
                header = f">{record.id}{i+1} {pred.begin}..{pred.end} strand:{'+' if pred.strand == 1 else '-'}"
                x = str(Seq.Seq(pred.sequence()).translate(to_stop=True))
                sequences.append(x)
                sequences_id.append(sid)
    
    return pd.DataFrame({
        "Sequence" : sequences, "Sequence ID": sequences_id
    })

def load_dataset_eskape(by = "traits_of_interest", nlargest = 10, 
                        class_size = 500, random_state = 42) :
    """
    Load sub-sample of dataset ESKAPE according to trait of interest or more frequent.
    """
    # path to ESKAPE
    eskape = '/home/ec2-user/repos/project-delta/data/ESKAPE'
    rng = np.random.default_rng(seed = random_state)

    if (nlargest <= 0):
        raise Exception("Wrong `nlargest`.")

    records_data = []
    for ffn_path in Path(eskape).glob("*.ffn"):
        host = os.path.basename(ffn_path).split("_")[1].split(".")[0]
        for record in SeqIO.parse(ffn_path, "fasta"):
            # Skip if not a multiple of 3 'in-frame'
            if len(record.seq) % 3 != 0:
                continue
                
            label = (" ").join(record.description.split(" ")[1:])

            if (by == "traits_of_interest"):
                if (host == "Klebsiella"):
                    continue
                elif (label not in traits_of_interest):
                    continue
                    
            elif (by == "top_traits"):
                bad_annotations = ["hypothetical protein",
                                   "unannotated protein",
                                   "unknown function"]
                # Skip hypothetical proteins
                if (label in bad_annotations):
                    continue
            else :
                raise Exception("Wrong `by`.")

            # Translate nucleotide to amino acid
            try:
                aa_seq = record.seq.translate(to_stop=True)
            except Exception as e:
                print(f"Translation error in {record.id}: {e}")
                continue

            records_data.append({
                "Sequence ID": record.id,
                "Label": label,
                "Sequence": str(aa_seq),
                "Host": host
            })

    # Convert to DataFrame
    df = pd.DataFrame(records_data).drop_duplicates()

    # find top labels
    sampled_df = []

    if (by == "top_traits"):
        top_labels = df['Label'].value_counts().nlargest(nlargest).index
        sampled_df = (
            df[df['Label'].isin(top_labels)]
            .groupby('Label', group_keys=False)
            .apply(lambda x: x.sample(n = min(len(x), class_size),
                                      random_state = rng.integers(1e5)))
        )
    else :
        sampled_df = (
            df
            .groupby('Label', group_keys=False)
            .apply(lambda x: x.sample(n = min(len(x), class_size),
                                      random_state = rng.integers(1e5)))
        )

    sampled_df["HR_Label"] = sampled_df.Label
    sampled_df["Label"] = sampled_df.Label.astype("category").cat.codes
    sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    
    return sampled_df


def _extract_base_id(sequence_id):
    return sequence_id.split('_')[0]

def _extract_protein_id(sequence_id):
    return sequence_id.split('_')[1]
    
def _associate_label_from_metadata(
    input_data, 
    choice = "Lowest Taxa",
    inphared = "/home/ec2-user/repos/project-delta/data/INPHARED",
    label_name = "Label",
    digitize = True
) :
    # load metadata with corresponding label choice
    label_frame = pd.read_csv(
        inphared + '/14Apr2025_data_excluding_refseq.tsv',
        sep='\t', usecols=["Accession", choice]
    )

    # drop unclassified
    mask = np.logical_and(
        label_frame[choice] != "Unclassified",
        label_frame[choice] != "Unspecified"
    ).values
    
    # convert into numerical values
    if (digitize is True) :
        label_frame[choice] = label_frame[choice].astype("category").cat.codes
    
    print(f"Classification on {choice}")
    print(f"Sequences classified {sum(mask)}")
    print(f"Sequences NOT classified {sum(~mask)}")
    print(f"Unique Labels {len(np.unique(label_frame[choice][mask]))}")
    
    label_frame = label_frame.rename(
        columns={"Accession" : "Sequence ID", choice : label_name}
    )
    
    return pd.merge(input_data, label_frame[mask], 
                    on="Sequence ID", how = "inner")


def load_dataset_inphared(
    data_source = "proteins",
    choice = "Lowest Taxa",
    inphared = "/home/ec2-user/repos/project-delta/data/INPHARED"
):
    """
    Load dataset for classification from INPHARED.
    """
    
    sequence_ids = []
    sequences = []
    if (data_source == "genomes") :
        for record in SeqIO.parse(inphared+'/14Apr2025_genomes_excluding_refseq.fa', "fasta"):
            sequence_ids.append(record.id)
            sequences.append(str(record.seq))
        output = pd.DataFrame({"Sequence ID": sequence_ids,
                                 "Sequence": sequences})

    elif (data_source == "proteins") :
        for record in SeqIO.parse(inphared+'/14Apr2025_vConTACT2_proteins.faa', "fasta"):
            sequence_ids.append(record.id)
            sequences.append(str(record.seq))
        output = pd.DataFrame({"Sequence": sequences})
        output['Sequence ID'] = list(map(_extract_base_id, sequence_ids))
        output['Protein ID'] = list(map(_extract_protein_id, sequence_ids))
        
    else:
        raise Exception("Sorry, inrecognized data source.")

    duplicate_mask = output.duplicated(subset="Sequence", keep=False)
    duplicates_df = output[duplicate_mask].sort_values(by="Sequence")
    duplicate_ids = duplicates_df["Sequence ID"].unique()
    output = output[~output["Sequence ID"].isin(duplicate_ids)]

    return _associate_label_from_metadata(output, choice = choice)        
    
def stratified_split(proteins, labels, test_ratio=0.2, seed=42):
    label_to_indices = defaultdict(list)

    # Group indices by label
    for i, label in enumerate(labels):
        label_to_indices[int(label.item())].append(i)

    random.seed(seed)
    train_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        train_indices += indices[:split]
        test_indices += indices[split:]

    # Shuffle the global order (optional)
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Build final splits
    proteins_train = [proteins[i] for i in train_indices]
    proteins_test = [proteins[i] for i in test_indices]
    labels_train = torch.tensor([labels[i].item() for i in train_indices])
    labels_test = torch.tensor([labels[i].item() for i in test_indices])

    return proteins_train, proteins_test, labels_train, labels_test
