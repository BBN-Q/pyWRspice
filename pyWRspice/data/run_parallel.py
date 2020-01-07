# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Run multiple python scripts in parallel using multiprocessing
"""
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import os, sys, getopt, subprocess

def run_command(cmd):
    """ Run a command """
    with subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=os.environ.copy()) as process:
        proc_stds = process.communicate() # Get output messages
        proc_stdout = proc_stds[0].strip()
        msg = proc_stdout.decode('ascii')
        proc_stderr = proc_stds[1].strip()
        msg_err = proc_stderr.decode('ascii')
        if len(msg_err)>0 :
            print("ERROR when running: %s" %cmd)
            print(msg_err)
            return msg + msg_err
        else:
            return msg

def run_batch(fconfig,processes=64):
    """ Run a file with multiprocessing

    The file fconfig must contain the command in the second line
    then the list of file names in the column "circuit_file"

    When done: write a dummy file starting with 'finish_'
    """
    with open(fconfig,'r') as f:
        f.readline()
        command = f.readline().strip()[1:]
    df = pd.read_csv(fconfig,skiprows=2)
    fnames = np.array(df["circuit_file"])
    cmds = [command + " {}".format(fname) for fname in fnames]
    with Pool(processes=processes) as pool:
        results = []
        for cmd in cmds:
            results.append(pool.apply_async(run_command, (cmd,)))
        results = [result.get() for result in results]
    # Write a dummy file to indicate the completion of the simulation
    now = datetime.now()
    fend = os.path.join(os.path.dirname(fconfig),"finish_" + os.path.basename(fconfig)[:-4] + ".txt")
    with open(fend,'w') as f:
        f.write("Finished simulation of: %s \n" %fconfig)
        f.write("Finished at: %s \n" %now.strftime("%Y-%m-%d %H:%M:%S"))
    return results

def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "d", ["processes="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        sys.exit(2)
    if len(args)==0:
        raise ValueError("No input file given")
    fname = args[0]
    processes = 64
    display = False
    for o, a in opts:
        if o=="--processes":
            processes = int(float(a))
        if o=="-d":
            display = True
    result = run_batch(fname,processes=processes)
    if display:
        for r in result:
            print(r)
    return result


if __name__ == '__main__':
    main()
