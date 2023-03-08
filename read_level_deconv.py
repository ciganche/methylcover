#!/usr/bin/python3
# TODO: testiraj hg38 genom
# TODO: paralelizuj prepare_bam
# TODO: ubaci --mapq param u prepare bam
# TODO: vise pokusaja optimizacije - pamti dist
# TODO: convertuj index cpg za hg38
# TODO: result collecter
# TODO: detectability threshold
import numpy as np
import pandas as pd
from scipy import optimize
import copy
import multiprocessing as mp
import argparse
import sys
import os


# global result holder
unsorted_results_array = []

def read_params(passed_args):
    parser = argparse.ArgumentParser(description="""Performs deconvolution of a name-sorted SAM file aligned with Bismark. Use prepare_bam functionality to prepare a such SAM file. Outputs the results to stdout.""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.prog = "methylcover read_level_deconv"
    parser._optionals.title = "\nArguments"
    arg = parser.add_argument
    arg("--input_file", "-i", required=True, type=str,
        help="[REQUIRED] The NAME-SORTED SAM file on which deconvolution is going to be performed."
             "Use the output of the prepare functionality.")

    arg("--genome", "-g", required=True, type=str,
         help="[REQUIRED] Name of the genome reference used. hg19 and hg38 are only supported")

    arg("--output_file", "-o", default="[INPUT BASENAME]_read_deconv.tsv", type=str,
        help="The name of the output file.")

    arg("--number_of_basis_sets", "-n", default=369, type=int,
        help="Number of top basis sets to use in the deconvolution. All 369 basis sets are used by default. Should not be larger than 369.")

    arg("--basis_dir", "-bd", default=os.path.dirname(os.path.abspath(__file__))+"/data/basis/", type=str,
        help="The path to the stored basis info files.")

    arg("--atlas_dir", "-ad", default=os.path.dirname(os.path.abspath(__file__))+"/data/atlas/", type=str,
        help="The path to the stored atlas files.")

    arg("--std_file_suffix", "-stds", default="_tissue_std.tsv", type=str,
        help="Name of the file specifying which cell types are to be joined.")

    arg("--max_weight", "-mxw", default=3, type=float,
        help="Positive decimal number denoting the upper boundary of the penalization vector. NNLS penalties will be standardize to (1, max_weight) range.")

    arg("--grouped_cell_types_file", "-grps", default=os.path.dirname(os.path.abspath(__file__))+"/data/cluster_names.tsv", type=str,
        help="Name of the file specifying which cell types are to be joined.")

    arg("--label_list_file", "-lbs", default=os.path.dirname(os.path.abspath(__file__))+"/data/label_list.tsv", type=str,
        help="Name of the file the name format of the groups in the output file.")

    arg("--basis_file_suffix", "-bs", default="_info_basis.bed", type=str,
        help="Suffix of the basis files.")

    arg("--atlas_file_suffix", "-as", default="_final_ratio.tsv", type=str,
        help="Suffix of the atlas ratio files.")

    arg("--mapq", "-mq", default=20, type=int,
        help="Mapping quality threshold for a read.")

    arg("--min_read_cpg", "-mrc", default=3, type=int,
        help="Minimal number of CpG sites on a read for it to be considered.")

    arg("--min_reads_per_basis", "-mrb", default=1, type=int,
        help="Minimal number of reads covering each basis.")

    arg("--methylation_string_adapter", "-msa", default="XM:Z:", type=str,
        help="The prefix added by Bismark aligner to each methylation string. If 'XM:Z:' is not working, see which string is apppended to all reads in the sam file.")

    arg("--threads", "-@", default=0, type=int,
        help="Number of threads for classifying reads. 0 means use all available. ")

    arg("--verbose", default=False, action="store_true",
        help="Print excluded reads due to lacking CpGs and which basis have been removed.")

    return vars(parser.parse_args(passed_args))


def errprint(string):
    print("[Error] " + string, file=sys.stderr)

def print_channel2(string):
    print(str(string), file=sys.stderr)

def print_to_stdout(df):
    index_names = df.index.to_numpy()
    mat = df.to_numpy()
    for i in range(0, len(index_names)):
        print(index_names[i] + "\t" + str(mat[i, 0]))

def arg_checker(pars):

    edited_pars = copy.deepcopy(pars)

    if not os.path.exists(edited_pars["input_file"]):
        errprint("BAM file " + str(edited_pars["input_file"]) + " not found.")
        exit(1)

    if edited_pars["genome"] not in ["hg19", "hg38"]:
        errprint("Genome not supported.")
        exit(2)

    if edited_pars["number_of_basis_sets"] < 1 or edited_pars["number_of_basis_sets"] > 369:
        errprint("Invalid basis set count. Should be between 1 and 369.")
        exit(3)

    if not os.path.exists(edited_pars["basis_dir"]):
        errprint("Basis directory not found.")
        exit(4)
    if edited_pars["basis_dir"][-1] != "/": edited_pars["basis_dir"] = edited_pars["basis_dir"] + "/"

    if not os.path.exists(edited_pars["atlas_dir"]):
        errprint("Atlas directory not found.")
        exit(5)
    if edited_pars["atlas_dir"][-1] != "/": edited_pars["atlas_dir"] = edited_pars["atlas_dir"] + "/"

    edited_pars["output_file"] = edited_pars["output_file"] if edited_pars["output_file"] != "[INPUT BASENAME]_read_deconv.tsv" \
        else str(edited_pars["input_file"].replace(".sam", "") + "_read_deconv.tsv")

    if edited_pars["threads"] < 0 or edited_pars["threads"] > mp.cpu_count():
        errprint("Invalid thead count number. Should be between 1 and " + str(mp.cpu_count()) + ".")
        exit(6)
    if edited_pars["threads"] == 0: edited_pars["threads"] = mp.cpu_count()

    if not os.path.exists(edited_pars["basis_dir"] + edited_pars["genome"] + "/" + "1" + edited_pars["basis_file_suffix"]):
        errprint("Unable to find basis info bed files.")
        exit(7)

    if not os.path.exists(edited_pars["atlas_dir"] + "/" + "1" + edited_pars["atlas_file_suffix"]):
        errprint("Unable to find atlas files.")
        exit(8)

    if not os.path.exists(edited_pars["grouped_cell_types_file"]):
        errprint("Unable to find groupped cell types file.")
        exit(9)

    if not os.path.exists(edited_pars["label_list_file"]):
        errprint("Unable to find label list file.")
        exit(10)

    if edited_pars["mapq"] < 1 or edited_pars["mapq"] > 100:
        errprint("Invalid mapq threshold.")
        exit(11)

    if edited_pars["min_reads_per_basis"] < 1:
        errprint("Invalid minimal number of reads per basis.")
        exit(12)

    if edited_pars["min_read_cpg"] < 1:
        errprint("Invalid minimal number of CpG sites per read.")
        exit(13)

    if not os.path.exists(edited_pars["atlas_dir"] + "/" + "1" + edited_pars["std_file_suffix"]):
        errprint("Unable to find atlas standard deviation files.")
        exit(14)

    if edited_pars["max_weight"] < 1:
        errprint("Max weight parameter must be greater or equal to 1")
        exit(15)

    return edited_pars


# - - - - - - - - - - - - - - - - - - - - -
# - - -  matrix preparation functions - - -
# - - - - - - - - - - - - - - - - - - - - -

def prepare_grouped_matrix(file_path):
    matrix = pd.read_csv(file_path, sep="\t", header=None).to_numpy()
    for row in matrix:
        row[1] = row[1].split(",")
    return matrix

def prepare_basis_matrix(basis_file_dir, genome, file_number, bed_file_suffix, grouped_matrix):
    basis_file_df = pd.read_csv(basis_file_dir + genome + "/" + str(file_number) + bed_file_suffix, sep="\t", header=0)
    matrix = basis_file_df.to_numpy()
    for b in matrix:
        b[5] = [int(x) for x in b[5].split(",")]
        b[7] = [int(x) for x in b[7].split(",")]

        b[8] = [x for x in b[8].split(",")]
        b[9] = [x for x in b[9].split(",")]

        for group in grouped_matrix:
            # if all members of a group are in a string, remove them and add a single group name instead
            if len(np.intersect1d(group[1], b[8])) == len(group[1]):
                b[8] = np.append(np.setdiff1d(b[8], group[1]), group[0])
            if len(np.intersect1d(group[1], b[9])) == len(group[1]):
                b[9] = np.append(np.setdiff1d(b[9], group[1]), group[0])
    return matrix

def prepare_atlas_matrix(atlas_file_dir, file_number, atlas_file_suffix):
    atlas_file_df = pd.read_csv(atlas_file_dir + str(file_number) + atlas_file_suffix, sep="\t", header=0)
    matrix = atlas_file_df.to_numpy()
    matrix = np.delete(matrix, [0, 1, 2], 1)
    return matrix

def prepare_std_matrix(atlas_file_dir, file_number, std_file_suffix):
    std_file_df = pd.read_csv(atlas_file_dir + str(file_number) + std_file_suffix, sep="\t", header=0)
    matrix = std_file_df.to_numpy()
    matrix = np.delete(matrix, [0, 1, 2], 1)
    return matrix

def get_groups(atlas_file_dir, atlas_file_suffix):
    atlas_file_df = pd.read_csv(atlas_file_dir + "1" + atlas_file_suffix, sep="\t", header=0)
    return atlas_file_df.columns.to_numpy()[3:]

def format_output_df(percentage_list, group_list, label_file):
    label_matrix = pd.read_csv(label_file, sep="\t", header=0).to_numpy()
    percentage_list = np.round(percentage_list*100, 4)
    tmp_dict = {k: v for k, v in zip(group_list, percentage_list)}
    ret_val_percentages = [tmp_dict[g] for g in label_matrix[:, 0]]
    ret_val_df = pd.DataFrame(ret_val_percentages, index=label_matrix[:, 1])
    return ret_val_df

def prepare_sam_file(sam_file_dir, file_number, sam_file_suffix, mapq_threshold, methylation_adapter):
    sam_file_df = pd.read_csv(sam_file_dir + str(file_number) + sam_file_suffix,
                              skip_blank_lines=True, sep="\t", header=None)
    # Filter for MAPQ
    sam_file_df = sam_file_df[sam_file_df[4] > mapq_threshold]
    sam_file_df.reset_index()
    matrix = sam_file_df.to_numpy()
    for row in matrix:
        row[13] = row[13].replace(methylation_adapter, "")
        if len(row[13]) != len(row[9]):
            raise Exception("Length mismatch between methylation str and the sequence in read: " + row[0])
    return matrix

# - - - - - - - - - - - - - - - - - - - -
# - - -  SAM file parsing functions - - -
# - - - - - - - - - - - - - - - - - - - -

def intersection_exists(basis_row, read_row):
    if basis_row[0] != read_row[2]:
        return False    # not on same chr
    b_start = basis_row[1]
    b_end = basis_row[2]
    r_start = read_row[3]
    r_end = r_start + len(read_row[9]) - 1  # the end position
    if r_start <= b_start <= r_end:
        return True
    elif r_start <= b_end <= r_end:
        return True
    elif r_start >= b_start and r_end <= b_end:
        return True
    else:
        return False

def print_erroneous_sites(read_row, methylation, site):
    if site == len(read_row[9]) - 1:
        print_channel2("\t\tNo CpG site on: " + read_row[2] + ":" + str(site + read_row[3]) + "\t read id:" + read_row[0])
        print_channel2("\t\t\t" + methylation[site - 1] + " on " + read_row[9][site - 1] + " <--prev--| " +
              methylation[site] + " on " + read_row[9][site] + " | END OF READ ")
    elif site == 0:
        print_channel2("\t\tNo CpG site on: " + read_row[2] + ":" + str(site + read_row[3]) + "\t read id:" + read_row[0])
        print_channel2("\t\t\t" + "START OF READ | " +
              methylation[site] + " on " + read_row[9][site] + " |--next--> " +
              methylation[site + 1] + " on " + read_row[9][site + 1])
    else:
        print_channel2("\t\tNo CpG site on: " + read_row[2] + ":" + str(site + read_row[3]) + "\t read id:" + read_row[0])
        print_channel2("\t\t\t" + methylation[site - 1] + " on " + read_row[9][site - 1] + " <--prev--| " +
              methylation[site] + " on " + read_row[9][site] + " |--next--> " +
              methylation[site + 1] + " on " + read_row[9][site + 1])

def separate_sites(letter, array):
    if letter == "z":
        array.append(0)
    elif letter == "Z":
        array.append(1)
    else:
        raise Exception("Wrong spot in the methylation str: " + letter)

# Returns the state of the CpGs
def process_read(processed_read, absolute_cpg_positions, verbosity):
    relative_cpg_positions = absolute_cpg_positions - processed_read[3]
    methylation_str = processed_read[13]
    binary_holder = []
    for p in relative_cpg_positions:
        if methylation_str[p] in ["z", "Z"]:
            separate_sites(methylation_str[p], binary_holder)
        elif p != len(processed_read[9]) - 1 and methylation_str[p + 1] in ["z", "Z"]:  # check next position if it exists
            separate_sites(methylation_str[p + 1], binary_holder)
        elif p != 0 and methylation_str[p - 1] in ["z", "Z"]:     # check previous position if it exists
            separate_sites(methylation_str[p - 1], binary_holder)
        else:
            if verbosity:
                print_erroneous_sites(processed_read, methylation_str, p)
    return binary_holder

# Fills up results row with counts per cluster
def process_fragment(results_row, read_list, absolute_cpg_position_list, basis, cpg_cnt_threshold, verbosity):
    binary_holder = []
    # ako jedan read nema preseka sa bazom a drugi ima, absolute_cpg_position_list je prazna pa preskoci taj
    for i in range(0, len(read_list)):
        if len(absolute_cpg_position_list[i]) == 0:
            continue
        binary_holder = binary_holder + process_read(read_list[i], absolute_cpg_position_list[i], verbosity)

    if len(binary_holder) < cpg_cnt_threshold:
        if verbosity:
            print_channel2("\tFragment with first read ID: " + read_list[0][0] + " skipped - identified CpGs less than: " + str(cpg_cnt_threshold))
        return

    methylation = np.mean(binary_holder)
    # half of the distance between the highest value of cluster1 and lowest value of cluster two
    tolerance = (basis[12] - basis[11])/2
    if methylation <= basis[11] + tolerance:
        results_row[0] += 1
    else:
        results_row[1] += 1


# Goes through a -n sorted SAM file, processes single/paired reads and fills up a result matrix (counts per cluster)
def classify_reads_per_basis(index, basis_file_np, sam_file_np, min_cpgs_in_read, verbosity):
    # reads like cluster 1, like cluster 2 and uncategorized-between for each basis
    classified_reads = np.zeros((len(basis_file_np), 2))
    for basis_index in range(0, len(basis_file_np)):
        basis = basis_file_np[basis_index]

        read_index = 0
        while read_index < len(sam_file_np):
            current_read = sam_file_np[read_index]
            if read_index+1 != len(sam_file_np):
                next_read = sam_file_np[read_index+1]
            # if ids are not the same - single end fragment should be processed
            if current_read[0] != next_read[0]:
                if intersection_exists(basis, current_read):
                    current_read_start = current_read[3]
                    current_read_end = current_read_start + len(current_read[9])  # postion after the last position
                    current_read_cpgs_absolute = np.intersect1d(basis[5], np.arange(current_read_start, current_read_end))
                    if len(current_read_cpgs_absolute) >= min_cpgs_in_read:
                        process_fragment(classified_reads[basis_index], [current_read], [current_read_cpgs_absolute],
                                         basis, min_cpgs_in_read, verbosity)
                read_index += 1
            # if ids are the same - pair end fragment should be processed
            else:
                if intersection_exists(basis, current_read) or intersection_exists(basis, next_read):

                    current_read_start = current_read[3]
                    current_read_end = current_read_start + len(current_read[9])  # postion after the last position
                    next_read_start = next_read[3]
                    next_read_end = next_read_start + len(next_read[9])  # postion after the last position

                    current_read_cpgs_absolute = np.intersect1d(basis[5], np.arange(current_read_start, current_read_end))
                    next_read_cpgs_absolute = np.intersect1d(basis[5], np.arange(next_read_start, next_read_end))
                    if len(np.union1d(current_read_cpgs_absolute, next_read_cpgs_absolute)) >= min_cpgs_in_read:
                        process_fragment(classified_reads[basis_index], [current_read, next_read],
                                         [current_read_cpgs_absolute, next_read_cpgs_absolute], basis, min_cpgs_in_read,
                                         verbosity)
                read_index += 2
    return index, classified_reads

def collect_result(result):
    # fill in the global variable
    unsorted_results_array.append(result)

def release_classification_process(index, basis_file_np, sam_file_np, min_cpgs_in_read, verbosity, pool_obj):
    pool_obj.apply_async(classify_reads_per_basis, args=(index, basis_file_np, sam_file_np, min_cpgs_in_read, verbosity),
                         callback=collect_result)

# - - - - - - - - - - - - - - - - - -
# - - -  Optimization functions - - -
# - - - - - - - - - - - - - - - - - -

def uncovered_basis_printout(basis_matrix, index_list, min_read_per_basis, verbosity):
    print_channel2(str(len(index_list)) + " basis have been removed due to having less than " + str(min_read_per_basis) +
          " covering it.\n")
    if verbosity:
        for i in index_list:
            print_channel2("Removed basis: " + basis_matrix[i, 0] + ":" + str(basis_matrix[i, 1]) + "-" + str(basis_matrix[i, 2]))
    print_channel2("\n")

def get_atlas_coefficient_matrix(cluster_names, selection, group_list, atlas):
    # number of basis x groups
    ret_val = np.zeros((len(cluster_names), len(group_list)))
    for i in range(0, len(ret_val)):
        if selection[i] == 1:  # fitting c2
            ret_val[i, :] = 1 - atlas[i, :]
        else:   # fitting c1
            ret_val[i, :] = atlas[i, :]
    return ret_val

def standardize(array, a, b):
    mmax = np.max(array)
    mmin = np.min(array)
    return a + (((array - mmin)*(b-a))/(mmax - mmin))

def fit_NNLS(c1_c2_mat, basis_mat, atlas, new_group_list, max_w_range, std_vector):
    cluster_selection = [0 if len(row[8]) <= len(row[9]) else 1 for row in basis_mat]
    A = get_atlas_coefficient_matrix(basis_mat[:, [8, 9]], cluster_selection, new_group_list, atlas)

    target = []
    # coverage diagional matrix - only the reads falling into the two clusters are considered
    coverage_matrix = np.zeros((len(c1_c2_mat), len(c1_c2_mat)))
    for i in range(0, len(c1_c2_mat)):
        target.append(c1_c2_mat[i, cluster_selection[i]])
        coverage_matrix[i, i] = sum(c1_c2_mat[i, :])

    flipped_std_vector = np.array(1 - std_vector, dtype=np.float32)
    weights = standardize(flipped_std_vector, 1, max_w_range)
    nnls = optimize.nnls(np.sqrt(weights)[:, None] * np.matmul(coverage_matrix, A),
                         np.sqrt(weights) * np.asarray(target), maxiter=100000000)
    cell_proportions = nnls[0] / sum(nnls[0])
    cell_proportions = cell_proportions / sum(cell_proportions)
    return cell_proportions



# - - - - - - - - - - - - - - - - - -
#  - - - - - - - Main - - - - - - - -
# - - - - - - - - - - - - - - - - - -
def main(args):
    par = arg_checker(read_params(args))
    SAM_FILE = par["input_file"]
    GENOME = par["genome"]
    BASIS_GENERATED = par["number_of_basis_sets"]
    BASIS_FILE_DIR = par["basis_dir"]
    ATLAS_FILE_DIR = par["atlas_dir"]
    STD_FILE_SUFFIX = par["std_file_suffix"]
    MAX_WEIGHT = par["max_weight"]
    CLUSTER_FILE = par["grouped_cell_types_file"]
    LABEL_FILE = par["label_list_file"]
    MAPQ_THRESHOLD = par["mapq"]
    MIN_READS_PER_BASIS = par["min_reads_per_basis"]
    INTRA_READ_CPG_THRESHOLD = par["min_read_cpg"]
    BASIS_FILE_SUFFIX = par["basis_file_suffix"]
    ATLAS_FILE_SUFFIX = par["atlas_file_suffix"]
    METHYLATION_ADAPTER_STR = par["methylation_string_adapter"]
    N_CPUS = par["threads"]
    VERBOSE = par["verbose"]

    # prepare the inputs
    grouped_cell_types_matrix = prepare_grouped_matrix(CLUSTER_FILE)
    sam_file_matrix = prepare_sam_file("", SAM_FILE, "", MAPQ_THRESHOLD, METHYLATION_ADAPTER_STR)
    basis_array = []
    atlas_array = []
    std_array = []
    pool = mp.Pool(N_CPUS)

    # classify all the reads
    for file in range(1, BASIS_GENERATED + 1):
        basis_file_matrix = prepare_basis_matrix(BASIS_FILE_DIR, GENOME, file, BASIS_FILE_SUFFIX, grouped_cell_types_matrix)
        basis_array.append(basis_file_matrix)
        atlas_file_matrix = prepare_atlas_matrix(ATLAS_FILE_DIR, file, ATLAS_FILE_SUFFIX)
        atlas_array.append(atlas_file_matrix)
        std_file_matrix = prepare_std_matrix(ATLAS_FILE_DIR, file, STD_FILE_SUFFIX)
        std_array.append(std_file_matrix)
        release_classification_process(file, basis_file_matrix, sam_file_matrix, INTRA_READ_CPG_THRESHOLD, VERBOSE, pool)
    pool.close()
    pool.join()

    # sort the results
    indices = [result[0] for result in unsorted_results_array]
    order_of_indices = np.argsort(indices)
    results_array = [unsorted_results_array[i][1] for i in order_of_indices]
    atlas_chosen = np.vstack(atlas_array)
    std_chosen = np.vstack(std_array)
    basis_chosen = np.vstack(basis_array)
    results_chosen = np.vstack(results_array)

    # remove empty basis
    total_array = np.sum(results_chosen, axis=1)
    empty_basis_index = np.where(total_array < MIN_READS_PER_BASIS)[0]
    uncovered_basis_printout(basis_chosen, empty_basis_index, MIN_READS_PER_BASIS, VERBOSE)
    to_keep = np.setdiff1d(range(0, len(basis_chosen)), empty_basis_index)
    atlas_chosen = atlas_chosen[to_keep, :]
    basis_chosen = basis_chosen[to_keep, :]
    results_chosen = results_chosen[to_keep, :]
    std_chosen = std_chosen[to_keep, :]

    # show average basis coverage
    avg_coverage_per_basis = np.round(np.mean(total_array), decimals=3)
    avg_coverage_per_considered = np.round(np.mean(np.sum(results_chosen, axis=1)), decimals=3)
    print_channel2("Total considered fragments: " + str(sum(total_array)) + " on " + str(len(basis_chosen)) + " regions.")
    print_channel2("Mean fragment coverage across all regions: " + str(avg_coverage_per_basis) + "x")
    print_channel2("Mean fragment coverage across considered regions: " + str(avg_coverage_per_considered) + "x")

    # fit the model
    groups = get_groups(ATLAS_FILE_DIR, ATLAS_FILE_SUFFIX)
    std_vector = np.nanmax(std_chosen, axis=1)
    percentages = fit_NNLS(results_chosen, basis_chosen, atlas_chosen, groups, MAX_WEIGHT, std_vector)

    # write the output file
    output_df = format_output_df(percentages, groups, LABEL_FILE)
    print_channel2("\n")
    print_to_stdout(output_df)


if __name__ == "__main__":
    main(sys.argv[1:])
