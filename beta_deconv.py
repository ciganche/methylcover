#!/usr/bin/python3

import pandas as pd
import numpy as np
from scipy import optimize
import argparse
import sys
import os
import copy

def read_params(passed_args):
    parser = argparse.ArgumentParser(description="""Performs deconvolution using mean region beta values. Use prepare_bam functionality to get the right bed format for this type of deconvolution. Outputs the results to stdout.""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser._optionals.title = "\nArguments"
    arg = parser.add_argument
    arg("--input_file", "-i", required=True, type=str,
        help="[REQUIRED] Target beta values headerless bed file from which the proportions are estimated."
             " Use output of the prepare functionality. Columns order: chr, start, end, region mean beta value, mean sequencing depth.")

    arg("--genome", "-g", required=True, type=str,
         help="[REQUIRED] Name of the genome reference used. hg19 and hg38 are only supported")

    arg("--max_weight", "-mxw", default=10, type=float,
        help="Positive decimal number denoting the upper boundary of the penalization vector. NNLS penalties will be standardize to (1, max_weight) range.")

    arg("--detection_threshold", "-dth", default=0.005, type=float,
        help="Decimal ratio value (0 to 1) below which a detection is considered a false positive and erased. Keep at 0.005 or around.")

    arg("--multistart", "-ms", default=10, type=int,
        help="The number of times to retry fitting.")

    arg("--number_of_basis_sets", "-n", default=369, type=int,
        help="Number of basis sets to use in the deconvolution.")

    arg("--minimal_coverage", "-mc", default=1, type=int,
        help="The minimal coverage a basis needs to have, otherwise it will be removed.")

    arg("--basis_dir", "-bd", default=os.path.dirname(os.path.abspath(__file__))+"/data/basis/", type=str,
        help="The location of the stored basis files.")

    arg("--betas_dir", "-btd", default=os.path.dirname(os.path.abspath(__file__))+"/data/betas/", type=str,
        help="The location of the beta tsv files.")

    arg("--std_dir", "-stdd", default=os.path.dirname(os.path.abspath(__file__))+"/data/beta_std/", type=str,
        help="The location of the files with standard deviation for all bases sets accros all cell types. ")

    arg("--grouped_cell_types_file", "-grps", default=os.path.dirname(os.path.abspath(__file__))+"/data/cluster_names.tsv", type=str,
        help="Name of the file specifying which cell types are to be joined.")

    arg("--label_list_file", "-lbs", default=os.path.dirname(os.path.abspath(__file__))+"/data/label_list.tsv", type=str,
        help="Name of the file the name format of the groups in the output file.")

    arg("--basis_file_suffix", "-bs", default="_info_basis.bed", type=str,
        help="Suffix of the basis files.")

    arg("--beta_file_suffix", "-ts", default="_beta_basis.tsv", type=str,
        help="Suffix of the beta value files.")

    arg("--std_file_suffix", "-stds", default="_beta_std.tsv", type=str,
        help="Suffix of the standard deviation files.")

    return vars(parser.parse_args(passed_args))


def errprint(string):
    print("[Error] " + str(string), file=sys.stderr)

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

    if not os.path.exists(edited_pars["betas_dir"]):
        errprint("Beta directory not found.")
        exit(5)
    if edited_pars["betas_dir"][-1] != "/": edited_pars["betas_dir"] = edited_pars["betas_dir"] + "/"

    if not os.path.exists(edited_pars["basis_dir"] + edited_pars["genome"] + "/" + "1" + edited_pars["basis_file_suffix"]):
        errprint("Unable to find basis info bed files.")
        exit(6)

    if not os.path.exists(edited_pars["betas_dir"] + "/" + "1" + edited_pars["beta_file_suffix"]):
        errprint("Unable to find atlas files.")
        exit(7)

    if not os.path.exists(edited_pars["grouped_cell_types_file"]):
        errprint("Unable to find groupped cell types file.")
        exit(8)

    if not os.path.exists(edited_pars["label_list_file"]):
        errprint("Unable to find label list file.")
        exit(9)

    if edited_pars["minimal_coverage"] < 0.001:
        errprint("Invalid mean basis coverage.")
        exit(10)

    if not os.path.exists(edited_pars["std_dir"]):
        errprint("Standard deviation directory not found.")
        exit(11)

    if not os.path.exists(edited_pars["std_dir"] + "/" + "1" + edited_pars["std_file_suffix"]):
        errprint("Unable to find standard deviation files.")
        exit(12)
    if edited_pars["std_dir"][-1] != "/": edited_pars["std_dir"] = edited_pars["std_dir"] + "/"

    if edited_pars["max_weight"] < 1:
        errprint("Max weight parameter must be greater or equal to 1")
        exit(13)

    if edited_pars["detection_threshold"] < 0 or edited_pars["detection_threshold"] >= 1:
        errprint("Detection threshold must be greater than 0 and less than 1.")
        exit(14)

    if edited_pars["multistart"] < 1:
        errprint("Fitting must be performed at least once. Set multistart to a non-zero, positive value.")
        exit(15)

    return edited_pars

def prepare_target_matrix(target_file, basis_mat):
    target_matrix = pd.read_csv(target_file, sep="\t", header=None).dropna(axis=0, how="all").to_numpy()
    basis_ids = np.char.add(basis_mat[:, 0].astype(str), basis_mat[:, 1].astype(str))
    target_file_ids = np.char.add(target_matrix[:, 0].astype(str), target_matrix[:, 1].astype(str))
    order = []
    missing_basis = []
    for b_index in range(0, len(basis_ids)):
        missed = True
        basis_id = basis_ids[b_index]
        for i in range(0, len(target_file_ids)):
            if basis_id == target_file_ids[i]:
                order.append(i)
                missed = False
                break
        if missed:
            missing_basis.append(b_index)
            region = basis_mat[b_index, 0] + ":" + str(basis_mat[b_index, 1]) + "-" + str(basis_mat[b_index, 2])
            print_channel2("Basis " + region + " is missing from the input file - region not considered.")
    return target_matrix[order, :], missing_basis

def get_removed_ones(target_mat, min_coverage):
    non_numerical_ids = []
    for row_id in range(0, len(target_mat)):
        region = target_mat[row_id, 0] + ":" + str(target_mat[row_id, 1]) + "-" + str(target_mat[row_id, 2])
        try:
            beta_val = float(target_mat[row_id, 3])
            cov_val = float(target_mat[row_id, 4])
            if not np.isfinite(beta_val):
                non_numerical_ids.append(row_id)
                print_channel2("Wrongly formatted beta value for region " + region + " = " + str(beta_val) +
                               " - region not considered. [CHECK INPUT FILE]")
                continue

            if not np.isfinite(cov_val):
                non_numerical_ids.append(row_id)
                print_channel2("Wrongly formatted coverage value for region " + region + " = " + str(cov_val) +
                               " - region not considered. [CHECK INPUT FILE]")
                continue

            if cov_val < min_coverage:
                non_numerical_ids.append(row_id)
                print_channel2("Low coverage ( < " + str(min_coverage) + " ) for region " + region + " = " +
                               str(cov_val) + " region - not considered.")
                continue

            if beta_val > 1:
                non_numerical_ids.append(row_id)
                print_channel2("Wrosngly formatted beta value for region " + region + " = " + str(beta_val) +
                               " - region not considered. [CHECK INPUT FILE]")
                continue

        except ValueError as ve:
            non_numerical_ids.append(row_id)
            print_channel2("Non-numerical values detected for region " + region + " = " + str(target_mat[row_id, 3]) +
                           " (beta)" + "; " + str(target_mat[row_id, 4]) +
                           " (coverage) - region not considered. [CHECK INPUT FILE]")
    return non_numerical_ids

def prepare_grouped_matrix(file_path):
    matrix = pd.read_csv(file_path, sep="\t", header=None).to_numpy()
    for row in matrix:
        row[1] = row[1].split(",")
    return matrix

def prepare_basis_matrix(bed_file_dir, genome, file_number, bed_file_suffix, grouped_matrix):
    basis_file_df = pd.read_csv(bed_file_dir + genome + "/" + str(file_number) + bed_file_suffix, sep="\t", header=0)
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

def prepare_beta_matrix(beta_dir, file_number, beta_file_suffix, new_groups_list, grouped_matrix):
    beta_file_df = pd.read_csv(beta_dir + str(file_number) + beta_file_suffix, sep="\t", header=0, index_col=0)
    old_groups = beta_file_df.index
    old_groups_matrix = beta_file_df.to_numpy().T  # n regions x n groups

    # n regions x n new groups
    ret_val = np.zeros((len(old_groups_matrix), len(new_groups_list)))

    for region_index in range(0, len(ret_val)):
        for new_group_index in range(0, len(new_groups_list)):
            if new_groups_list[new_group_index] in old_groups:
                ret_val[region_index, new_group_index] = old_groups_matrix[region_index,
                                                         np.where(old_groups == new_groups_list[new_group_index])[0][0]]
            else:
                for cluster in grouped_matrix:
                    if new_groups_list[new_group_index] == cluster[0]:
                        groups_to_mean_indices = []
                        for x in cluster[1]:
                            groups_to_mean_indices.append(np.where(old_groups == x)[0][0])
                        ret_val[region_index, new_group_index] = np.mean(old_groups_matrix[region_index,
                                                                                           groups_to_mean_indices])
                        break
    return ret_val

def prepare_std_vector(std_dir, std_suffix, basis_set_cnt, missing_basis):
    array = []
    for i in range(1, basis_set_cnt+1):
        std_mat = pd.read_csv(std_dir + str(i) + std_suffix, sep="\t", header=0).to_numpy()
        array.append(std_mat)
    complete = np.vstack(array)
    complete = np.delete(complete, missing_basis, axis=0)
    return np.nanmax(complete, axis=1)

def print_out(matrix):
    for row in matrix:
        print_channel2("\t".join(str(x) for x in row))

def uncovered_basis_printout(missing, removed):
    removed_total = len(missing) + len(removed)
    print_channel2("\n" + str(removed_total) + " basis were removed before fitting.")
    print_channel2("--------------- end of stderr printout ---------------")


def standardize(array, a, b):
    mmax = np.max(array)
    mmin = np.min(array)
    return a + (((array - mmin)*(b-a))/(mmax - mmin))

def fit_NNLS(beta_mat, target_vec, max_w_range, std_vector):
    flipped_std_vector = np.array(1 - std_vector, dtype=np.float32)
    weights = standardize(flipped_std_vector, 1, max_w_range)
    nnls = optimize.nnls(np.sqrt(weights)[:, None] * beta_mat,
                         np.sqrt(weights) * np.asarray(target_vec), maxiter=100000000)
    cell_proportions = nnls[0] / sum(nnls[0])
    cell_proportions = cell_proportions / sum(cell_proportions)
    return cell_proportions, nnls[1]

def format_output_df(percentage_list, group_list, label_file):
    label_matrix = pd.read_csv(label_file, sep="\t", header=0).to_numpy()
    percentage_list = np.round(percentage_list*100, 4)
    tmp_dict = {k: v for k, v in zip(group_list, percentage_list)}
    ret_val_percentages = [tmp_dict[g] for g in label_matrix[:, 0]]
    ret_val_df = pd.DataFrame(ret_val_percentages, index=label_matrix[:, 1])
    return ret_val_df


# - - - - - - - - - - - - - - - - - -
#  - - - - - - - Main - - - - - - - -
# - - - - - - - - - - - - - - - - - -
def main(args):
    par = arg_checker(read_params(args))
    TARGET_FILE = par["input_file"]
    GENOME = par["genome"]
    BASIS_GENERATED = par["number_of_basis_sets"]
    MIN_COV = par["minimal_coverage"]
    BASIS_DIR = par["basis_dir"]
    BETA_DIR = par["betas_dir"]
    STD_DIR = par["std_dir"]
    CLUSTER_FILE = par["grouped_cell_types_file"]
    LABEL_FILE = par["label_list_file"]
    BASIS_FILE_SUFFIX = par["basis_file_suffix"]
    BETA_FILE_SUFFIX = par["beta_file_suffix"]
    STD_FILE_SUFFIX = par["std_file_suffix"]
    max_weight = par["max_weight"]
    DETECTION_THRESHOLD = par["detection_threshold"]
    MULTISTART = par["multistart"]

    grouped_types_matrix = prepare_grouped_matrix(CLUSTER_FILE)
    basis = prepare_basis_matrix(BASIS_DIR, GENOME, 1, BASIS_FILE_SUFFIX, grouped_types_matrix)
    new_groups = np.union1d(basis[0, 8], basis[0, 9])
    beta = prepare_beta_matrix(BETA_DIR, 1, BETA_FILE_SUFFIX, new_groups, grouped_types_matrix)
    basis_array = []
    beta_array = []
    basis_array.append(basis)
    beta_array.append(beta)
    for i in range(2, BASIS_GENERATED + 1):
        basis_array.append(prepare_basis_matrix(BASIS_DIR, GENOME, i, BASIS_FILE_SUFFIX, grouped_types_matrix))
        beta_array.append(prepare_beta_matrix(BETA_DIR, i, BETA_FILE_SUFFIX, new_groups, grouped_types_matrix))
    basis_matrix = np.vstack(basis_array)
    beta_matrix = np.vstack(beta_array)
    # Remove absent basis
    target_file_matrix, missing_basis_vec = prepare_target_matrix(TARGET_FILE, basis_matrix)
    std_vec = prepare_std_vector(STD_DIR, STD_FILE_SUFFIX, BASIS_GENERATED, missing_basis_vec)
    beta_matrix = np.delete(beta_matrix, missing_basis_vec, axis=0)
    basis_matrix = np.delete(basis_matrix, missing_basis_vec, axis=0)
    # remove non-number covered regions
    to_delete = get_removed_ones(target_file_matrix, MIN_COV)
    target_file_matrix = np.delete(target_file_matrix, to_delete, axis=0)
    target_vector = np.asarray(target_file_matrix[:, 3], dtype=np.float32)
    beta_matrix = np.delete(beta_matrix, to_delete, axis=0)
    basis_matrix = np.delete(basis_matrix, to_delete, axis=0)
    std_vec = np.delete(std_vec, to_delete)
    uncovered_basis_printout(missing_basis_vec, to_delete)


    # multistart fitting
    dist_array = []
    res_array = []
    for run in range(0, MULTISTART):
        percentages, dist = fit_NNLS(beta_matrix, target_vector, max_weight, std_vec)
        dist_array.append(dist)
        res_array.append(percentages)
    best_run = np.argmin(dist_array)
    percentages = res_array[best_run]

    # remove cell types below detection threshold
    percentages[np.where(percentages < DETECTION_THRESHOLD)[0]] = 0
    percentages = percentages / sum(percentages)

    # write the output file
    output_df = format_output_df(percentages, new_groups, LABEL_FILE)
    print_channel2("\n")
    print_to_stdout(output_df)


if __name__ == "__main__":
    main(sys.argv[1:])
