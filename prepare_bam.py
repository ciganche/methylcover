#!/usr/bin/python3

import argparse
import copy
import sys
import os
import numpy as np
import pandas as pd
import subprocess
import multiprocessing as mp


def read_params(passed_args):
    parser = argparse.ArgumentParser(description="""Before deconvolving a sample, prepare_bam functionality should be used. A BAM file, aligned with Bismark, is given (any compression level, sorted or not sorted) and the program outputs 1) a name sorted SAM file covering the regions of interest (used for read level deconvolution) and 2) a bed file with beta methylation values for each region of interest (used for beta based deconvolution).""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.prog = "methylcover prepare_bam"
    parser._optionals.title = "\nArguments"
    arg = parser.add_argument
    arg("--input_bam_file", "-i", required=True, type=str,
        help="[REQUIRED] A BAM file to be prepared for deconvolution. All type of sorting and compression are allowed.")

    arg("--genome", "-g", required=True, type=str,
        help="[REQUIRED] Name of the genome reference used. hg19 and hg38 are only supported.")

    arg("--output_dir", "-o", default="", type=str,
        help="Directory of the output files.")

    arg("--generate_read_level", default=False, action="store_true",
        help="Generate also a BED file for the beta_deconvolution.")

    arg("--keep_intermid", default=False, action="store_true",
        help="Keep intermediate coordinate-sorted BAM file.")

    arg("--only_properly_paired", default=False, action="store_true",
         help="In case of pair-ended experiment, use only properly paired.")

    arg("--number_of_basis_sets", "-n", default=369, type=int,
        help="Number of top basis sets to use in the deconvolution. All 369 basis sets are used by default. Should not be larger than 369.")

    arg("--basis_dir", "-bd", default=os.path.dirname(os.path.abspath(__file__))+"/data/basis/", type=str,
        help="The path to the stored basis info files.")

    arg("--basis_file_suffix", "-bs", default="_info_basis.bed", type=str,
        help="Suffix of the basis files.")

    arg("--output_file_suffix", "-os", default="_prepared", type=str,
        help="Suffix of the output files.")

    arg("--threads", "-@", default=0, type=int,
        help="Number of threads for classifying reads. 0 means use all possible.")
    return vars(parser.parse_args(passed_args))

def errprint(string):
    print("[Error] " + str(string), file=sys.stderr)

def arg_checker(pars):
    edited_pars = copy.deepcopy(pars)

    if not os.path.exists(edited_pars["input_bam_file"]):
        errprint("BAM file " + str(edited_pars["input_bam_file"]) + " not found.")
        exit(1)

    if edited_pars["genome"] not in ["hg19", "hg38"]:
        errprint("Genome not supported.")
        exit(2)

    if not os.path.exists(edited_pars["basis_dir"]):
        errprint("Basis directory not found.")
        exit(3)
    if edited_pars["basis_dir"][-1] != "/": edited_pars["basis_dir"] = edited_pars["basis_dir"] + "/"

    if edited_pars["number_of_basis_sets"] < 1 or edited_pars["number_of_basis_sets"] > 369:
        errprint("Invalid basis set count. Should be between 1 and 369.")
        exit(4)

    if edited_pars["output_dir"].strip() != "" and not os.path.exists(edited_pars["output_dir"]):
        errprint("Output directory not found.")
        exit(5)
    if len(edited_pars["output_dir"]) == 0: edited_pars["output_dir"] = edited_pars["output_dir"] + "./"
    if edited_pars["output_dir"][-1] != "/": edited_pars["output_dir"] = edited_pars["output_dir"] + "/"

    if edited_pars["threads"] < 0 or edited_pars["threads"] > mp.cpu_count():
        errprint("Invalid thead count number. Should be between 1 and " + str(mp.cpu_count()) + ".")
        exit(6)
    if edited_pars["threads"] == 0: edited_pars["threads"] = mp.cpu_count()

    if not os.path.exists(edited_pars["basis_dir"] + edited_pars["genome"] + "/" + "1" + edited_pars["basis_file_suffix"]):
        errprint("Unable to find basis info bed files.")
        exit(7)

    return edited_pars


def get_basis_bed_files(basis_dir, genome, basis_set_cnt, basis_suffix, bn, output_dir, output_suffix):
    basis_array = []
    for i in range(1, basis_set_cnt + 1):
        basis_array.append(pd.read_csv(basis_dir + genome + "/" + str(i) + basis_suffix, sep="\t", header=0).to_numpy()[:, 0:5])
    basis_matrix = np.vstack(basis_array)
    # beta fitting unsorted bed file with their chromosome
    beta_fitting_filename = output_dir + bn + output_suffix + "_beta_fitting.bed"
    pd.DataFrame(basis_matrix).to_csv(beta_fitting_filename, sep="\t", header=False, index=False)
    # sorted bed
    bedtools_bed_filename = output_dir + bn + output_suffix + "_bedtools.bed"
    bedtools_matrix = basis_matrix[:, 0:3]
    # sort by chromosome
    chr_factorized = np.char.replace(bedtools_matrix[:, 0].astype(str), "chr", "").astype(int)
    bedtools_matrix = bedtools_matrix[np.lexsort((bedtools_matrix[:, 1], chr_factorized)), :]
    pd.DataFrame(bedtools_matrix).to_csv(bedtools_bed_filename, sep="\t", header=False, index=False)
    return bedtools_bed_filename, beta_fitting_filename


def extract_n_filter_reads(bam_file, bn, whitelist_file, output_dir, output_suffix, properly_paired):
    tmp_output = output_dir + bn + output_suffix + "_extracted.bam"
    file = open(tmp_output, "w")
    extract_command = ["bedtools", "intersect", "-abam", bam_file, "-b", whitelist_file]
    process_command = subprocess.run(extract_command, stdout=file)
    file.close()
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to bedtools intersect error (bedtools exit code" + str(process_command.stderr)
                 + ").")
        exit(8)

    output_file = output_dir + bn + output_suffix + "_filtered.bam"
    if properly_paired:
        view_command = ["samtools", "view", "-h", "-f" "2", "-F", "256", "-O" "BAM", "-o", output_file, tmp_output]
    else:
        view_command = ["samtools", "view", "-h", "-F", "256", "-O" "BAM", "-o", output_file, tmp_output]
    process_command = subprocess.run(view_command)
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to samtools view error (samtools exit code" + str(process_command.stderr)
                 + ").")
        exit(9)
    os.remove(tmp_output)
    return output_file


def sort_by_name(bam_file, bn, output_dir, output_suffix):
    output_file = output_dir + bn + output_suffix + "_namesorted.bam"
    sort_command = ["samtools", "sort", "-n", "-O", "BAM", "-o", output_file, bam_file]
    process_command = subprocess.run(sort_command)
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to samtools -n sort error (samtools exit code" + str(process_command.stderr)
                 + ").")
        exit(10)
    return output_file

def get_sam(bam_file, bn, output_dir, output_suffix):
    output_file = output_dir + bn + output_suffix + ".sam"
    file = open(output_file, "w")
    # exclude those not propperly mapped and not being primary alignments
    view_command = ["samtools", "view", "--no-header", bam_file]
    process_command = subprocess.run(view_command, stdout=file)
    file.close()
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to samtools view error (samtools exit code" + str(process_command.stderr)
                 + ").")
        exit(11)
    return output_file

def sort_by_coord(bam_file, bn, output_dir, output_suffix):
    output_file = output_dir + bn + output_suffix + "_coordsorted.bam"
    file = open(output_file, "w")
    sort_command = ["samtools", "sort", bam_file]
    process_command = subprocess.run(sort_command, stdout=file)
    file.close()
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to samtools coordinate sort error (samtools exit code" + str(process_command.stderr)
                 + ").")
        exit(12)
    return output_file


# get coverage per region samtools -r "region" -a bam_file | awk '{ total += $3 } END { print total/NR }'
def get_coverage_for_region(region_str, filepath):
    generate_pat_command_string_array = ["samtools", "depth", "-r", region_str, "-a", filepath]
    process_command = subprocess.run(generate_pat_command_string_array, capture_output=True)
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to samtools depth error (samtools depth exit code" + str(process_command.stderr)
                 + ").")
        exit(13)
    stdout_matrix = pd.DataFrame([x.split("\t") for x in process_command.stdout.decode().split("\n")]).to_numpy()
    # remove empty rows if there are any
    stdout_matrix = stdout_matrix[np.where(stdout_matrix[:, 2] != None)[0], :]
    return np.round(np.mean(stdout_matrix[:, 2].astype(int)), 4)


def get_target_beta_file(bam_file, genome, blocks_file, basename_original, output_dir, output_suffix, cpus):
    # bam 2 pat - also generates a beta file: goes to output_dir
    generate_pat_command_string_array = ["wgbstools", "bam2pat", "--genome", genome, "-@", str(cpus),
                                         "-o", output_dir, bam_file]
    process_command = subprocess.run(generate_pat_command_string_array, capture_output=True)
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to wgbstools bam2pat error (wgbstools exit code" + str(process_command.stderr)
                 + ").")
        exit(14)
    # a bit of cleanup
    tmp_basename = (bam_file.split("/")[-1]).replace(".bam", "")
    os.remove(output_dir + tmp_basename + ".pat.gz")
    os.remove(output_dir + tmp_basename + ".pat.gz.csi")

    # beta 2 table: ide na stdout
    output_file = output_dir + basename_original + output_suffix + ".bed"
    betas_file = output_dir + tmp_basename + ".beta"
    beta_to_table_command_string_array = ["wgbstools", "beta_to_table", "--betas", betas_file,
                                          "-@", str(cpus),  blocks_file]
    process_command = subprocess.run(beta_to_table_command_string_array, capture_output=True)
    if process_command.returncode != 0:
        errprint(process_command.stderr)
        errprint("Exiting due to wgbstools beta_to_table error (wgbstools exit code" + str(process_command.stderr)
                 + ").")
        exit(15)
    # format the stdout, add mean region depth
    regions_matrix = pd.DataFrame([x.split("\t") for x in process_command.stdout.decode().split("\n")]).to_numpy()
    regions_matrix = regions_matrix[1:, [0, 1, 2, 5]]
    regions_matrix = regions_matrix[np.where(regions_matrix[:, 2] != None)[0], :]
    depth_vector = np.array([get_coverage_for_region(x[0] + ":" + str(x[1]) + "-" + str(x[2]), bam_file) for x in regions_matrix])
    regions_matrix = np.hstack((regions_matrix, depth_vector.reshape(len(regions_matrix), 1)))
    pd.DataFrame(regions_matrix).to_csv(output_file, sep="\t", header=False, index=False)

    # subfinal cleanup
    os.remove(betas_file)
    return output_file


# - - - - - - - - - - - - - - - - - -
#  - - - - - - - Main - - - - - - - -
# - - - - - - - - - - - - - - - - - -
def main(args):
    par = arg_checker(read_params(args))
    BAM_FILE = par["input_bam_file"]
    GENOME = par["genome"]
    BED_FILE_DIR = par["basis_dir"]
    BASIS_SET_CNT = par["number_of_basis_sets"]
    OUTPUT_DIR = par["output_dir"]
    BASIS_FILE_SUFFIX = par["basis_file_suffix"]
    OUTPUT_SUFFIX = par["output_file_suffix"]
    N_CPUS = par["threads"]
    PROPERLY_PAIRED = par["only_properly_paired"]
    GENERATE_READ_LEVEL = par["generate_read_level"]
    KEEP_INTERMID = par["keep_intermid"]

    basename = BAM_FILE.replace(".bam", "").split("/")[-1]
    bedtools_bed, beta_fitting_basis_bed = get_basis_bed_files(BED_FILE_DIR, GENOME, BASIS_SET_CNT, BASIS_FILE_SUFFIX,
                                                               basename, OUTPUT_DIR, OUTPUT_SUFFIX)
    extracted_bam = extract_n_filter_reads(BAM_FILE, basename, bedtools_bed, OUTPUT_DIR, OUTPUT_SUFFIX, PROPERLY_PAIRED)
    # prepare for read based deconvolution
    if GENERATE_READ_LEVEL:
        name_sorted_bam = sort_by_name(extracted_bam, basename, OUTPUT_DIR, OUTPUT_SUFFIX)
        sam_file = get_sam(name_sorted_bam, basename, OUTPUT_DIR, OUTPUT_SUFFIX)
        os.remove(name_sorted_bam)
    # prepare for beta fitting deconvolution
    coord_sorted = sort_by_coord(extracted_bam, basename, OUTPUT_DIR, OUTPUT_SUFFIX)
    final_for_beta_based = get_target_beta_file(coord_sorted, GENOME, beta_fitting_basis_bed, basename, OUTPUT_DIR,
                                                OUTPUT_SUFFIX, N_CPUS)

    if not KEEP_INTERMID:
        os.remove(coord_sorted)

    # cleanup
    os.remove(coord_sorted + ".bai")
    os.remove(extracted_bam)
    os.remove(bedtools_bed)
    os.remove(beta_fitting_basis_bed)


if __name__ == "__main__":
    main(sys.argv[1:])
