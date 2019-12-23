# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Run multiple python scripts in parallel using multiprocessing
"""
from multiprocessing import Pool
import os, sys, getopt, subprocess

def run_one(cmd):
    """ Run one file """
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

def run_batch(fin,processes=16):
    """ Run a file with multiprocessing

    The file fin must contain the command in the first line
    then the list of file names thereafter
    """
    with open(fin,'r') as f:
        lines = f.readlines()
    command = lines[0].strip()
    fnames = [line.strip() for line in lines[1:]]
    cmds = [command + " {}".format(fname) for fname in fnames]
    with Pool(processes=processes) as pool:
        results = []
        for cmd in cmds:
            results.append(pool.apply_async(run_one, (cmd,)))
        results = [result.get() for result in results]
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
    processes = 16
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
