import RNA
import numpy as np
from Bio import AlignIO


def generate_bppm_from_fastafile(alignment_file):
    """
    Generate a BPPM (Base Pairing Probability Matrix) from a given alignment file    
    """
    alignment = AlignIO.read(alignment_file, format="fasta")
    sequences = [str(record.seq) for record in alignment]
    fc = RNA.fold_compound(sequences)
    structure_fold, mfe_fold = fc.mfe()
    (_, mfe_pf) = fc.pf()           # Calculate partition function
    bppm = fc.bpp()                 # Get base pair probabilities
    bppm = np.array(bppm)
    return bppm

def generate_msa_matrix(alignment_file, vocab):
    """
    Generate a MSA matrix from a given alignment file
    """
    alignment = AlignIO.read(alignment_file, format="fasta")
    sequences = [str(record.seq) for record in alignment]
    msa_matrix = np.zeros((len(sequences), len(sequences[0])))
    for i, seq in enumerate(sequences):
        msa_matrix[i,:] = [vocab[base] for base in seq]
    return msa_matrix

