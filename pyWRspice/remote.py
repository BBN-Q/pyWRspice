# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Python wrapper for WRspice run on an SSH server
    class WRWrapper_SSH: Run WRspice script via WRspice simulator on an SSH server.

    Required package: paramiko to handle SSH connections
"""

import numpy as np
import pandas as pd
import os, tempfile, time
from datetime import datetime
import uuid, itertools, logging
from multiprocessing import Pool
from paramiko.client import SSHClient

from simulation import RawFile, backslash

logging.basicConfig(level=logging.WARNING)

# Get the run_parallel.py file
dir_path = os.path.dirname(os.path.realpath(__file__))
fexec = "run_parallel.py"
fexec_orig = os.path.join(dir_path,"data",fexec)

#------------------------------------------
# Wrapper for convenient parallel loops
#------------------------------------------

class WRWrapper_SSH:
    """ Wrapper for WRspice simulator via SSH connection.

    script: Declare the script with python format strings.
    '{output_file}' should be written by the script in the .control block.
    Any other keywords (which become mandatory) are added as named slots in the format string.

    server, login_user, login_pass: server address and login credentials
    local_dir: local working directory. If None: use temporary directory.
    remote_dir: remote working directory. If None: use tmp folder in current location.

    source: WRspice .cir source file
    command: location of the wrspice exec file, depending on specific system:
    For Unix systems, it is likely "/usr/local/xictools/bin/wrspice"
    For Windows, it is likely "C:/usr/local/xictools/bin/wrspice.bat"
    """
    def __init__(self, server, login_user, login_pass,
                local_dir=None, remote_dir=None,
                script=None, source=None,
                command="/usr/local/xictools/bin/wrspice"):
        self.server = server
        self.login_user = login_user
        self.login_pass = login_pass
        self.command   = backslash(command)

        self.script    = script
        if source is not None:
            self.get_script(source)

        if local_dir is None:
            self.local_dir = backslash(tempfile.TemporaryDirectory())
        else:
            self.local_dir = backslash(local_dir)
        if remote_dir is None:
            # Create a tmp folder in the current location
            ssh = self.new_connection()
            list_dir = SSH_run(ssh,"ls")
            if "tmp" not in list_dir:
                SSH_run(ssh,"mkdir tmp")
            self.remote_dir = os.path.join(SSH_run(ssh,"pwd")[0],"tmp")
            ssh.close()
            logging.info("Created new remote temporary folder: %s" %self.remote_dir)
        else:
            self.remote_dir = remote_dir

    def new_connection(self):
        """ Make new SSH connection to the server """
        logging.debug("Open a new SSH connection")
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.server,username=self.login_user,password=self.login_pass)
        return ssh

    def remote_fname(self,fname):
        """ Convert a base filename to remote filename by adding remote directory """
        if (not isinstance(fname,str)) and hasattr(fname,'__iter__'):
            return [backslash(os.path.join(self.remote_dir,name)) for name in fname]
        return backslash(os.path.join(self.remote_dir,fname))

    def local_fname(self,fname):
        """ Convert a base filename to local filename by adding local directory """
        if (not isinstance(fname,str)) and hasattr(fname,'__iter__'):
            return [os.path.join(self.local_dir,name) for name in fname]
        return os.path.join(self.local_dir,fname)

    def get(self,fnames,relative=True):
        """ Transfer file(s) from remote to local
        relative: if True, fnames are relative to the remote working directory
        """
        ssh = self.new_connection()
        sftp = ssh.open_sftp()
        if isinstance(fnames,str):
            fnames = [fnames]
        # get local fnames
        local_fnames = []
        for fname in fnames:
            if relative:
                local_fname = os.path.join(self.local_dir,fname)
                remote_fname = backslash(os.path.join(self.remote_dir,fname))
            else:
                local_fname = os.path.join(self.local_dir,os.path.basename(fname))
                remote_fname = backslash(fname)
            sftp.get(remote_fname,local_fname)
            local_fnames.append(local_fname)
        sftp.close()
        ssh.close()
        logging.info("Retrieved files from remote: %s" %fnames)
        if len(local_fnames)==1:
            return local_fnames[0]
        return local_fnames

    def put(self,fnames,relative=True):
        """ Transfer file(s) from local to remote
        relative: if True, fnames are relative to the local working directory
        """
        ssh = self.new_connection()
        sftp = ssh.open_sftp()
        if isinstance(fnames,str):
            fnames = [fnames]
        # get remote fnames
        remote_fnames = []
        for fname in fnames:
            if relative:
                remote_fname = backslash(os.path.join(self.remote_dir,fname))
                local_fname = os.path.join(self.local_dir,fname)
            else:
                remote_fname = backslash(os.path.join(self.remote_dir,os.path.basename(fname)))
                local_fname = fname
            sftp.put(local_fname,remote_fname)
            remote_fnames.append(remote_fname)
        sftp.close()
        ssh.close()
        logging.info("Sent files to remote: %s" %fnames)
        if len(remote_fnames)==1:
            return remote_fnames[0]
        return remote_fnames

    def _new_fname(self, prefix="",suffix=""):
        """ Create a temporary file in the temporary folder """
        return prefix + str(uuid.uuid4()) + suffix

    def get_script(self,fname):
        """ Get WRspice script from .cir file """
        with open(fname,'r') as f:
            lines = f.readlines()
        self.script = "".join(lines)

    def run_file(self,fname_remote):
        """ Run a file on a server """
        client = self.new_connection()
        cmd = self.command + " -b {}".format(fname_remote)
        logging.info("Run on remote: %s" %cmd)
        msg = SSH_run(client,cmd)
        client.close()
        return msg

    def _render(self,script,kwargs):
        """ Render a script by formatting it with kwargs
        then write into a local file
        """
        if "circuit_file" not in kwargs.keys():
            circuit_fname = self._new_fname("tmp_script_",".cir")
        else:
            circuit_fname = kwargs["circuit_file"]
        if "output_file" not in kwargs.keys() or kwargs["output_file"] in [None, ""]:
            output_fname = self._new_fname("tmp_output_",".raw")
        else:
            output_fname = kwargs["output_file"]
        # Make sure the paths use backslashes
        kwargs["output_file"] = backslash(os.path.join(self.remote_dir,output_fname))
        rendered_script = script.format(**kwargs)
        with open(os.path.join(self.local_dir,circuit_fname),'w') as f:
            f.write(rendered_script)
        return circuit_fname, output_fname

    def run(self,*script,display=False,save_file=False,**kwargs):
        """ Write a text into a script file fname on SSH server, then run it
        If save_file==True: save a local copy of the script file, do not remove fname after execution
        If 'circuit_file' and 'output_file' are not specified in kwargs: use randomly generated filenames
        local_dir, remote_dir: Local and Remote working directories. If None: use current location.

        If display: print output of simulation
        """
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
        circuit_fname, output_fname = self._render(self.script,kwargs)
        kwargs["output_file"] = backslash(os.path.join(self.remote_dir,output_fname))
        # Copy the script file on the server
        circuit_fname_remote = self.put(circuit_fname)
        # Execute file
        msg = self.run_file(circuit_fname_remote)
        if display:
            for m in msg:
                print(m)
        # Get the result file back
        output_fname_local = self.get(output_fname)
        rawfile = RawFile(output_fname_local, binary=True)
        if not save_file:
            logging.debug("Remove temporary files")
            os.remove(os.path.join(self.local_dir,circuit_fname))
            os.remove(output_fname_local)
            # Remove files on the server
            client = self.new_connection()
            sftp = client.open_sftp()
            sftp.remove(circuit_fname_remote)
            sftp.remove(kwargs["output_file"])
            # Close SFTP and SSH connection
            sftp.close()
            client.close()
        return rawfile

    def prepare_parallel(self, *script, **params):
        """ Write script files on local and remote locations
        to prepare for the actual simulation execution """
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
        # Disintegrate the parameters (dict)
        iter_params = {}
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                # if param value is a list
                iter_params[k] = v
            else:
                kws[k] = v
        param_vals = list(itertools.product(*[iter_params[k] for k in iter_params.keys()]))
        # Write circuit files
        circuit_fnames = []
        all_params = []
        for i,vals in enumerate(param_vals):
            kws_cp = kws.copy()
            for pname,val in zip(iter_params.keys(), vals):
                kws_cp[pname] = val
            # Make sure they run separate script files
            if "circuit_file" not in kws_cp.keys():
                kws_cp["circuit_file"] = "tmp_circuit_%d.cir" %i
            else:
                kws_cp["circuit_file"] = kws_cp["circuit_file"][:-4] + "_%d.cir" %i
            if "output_file" not in kws_cp.keys() or kws_cp["output_file"] in [None,'']:
                kws_cp["output_file"] = "tmp_output_%d.raw" %i
            else:
                kws_cp["output_file"] = kws_cp["output_file"][:-4] + "_%d.raw" %i

            circuit_fname, output_fname = self._render(self.script,kws_cp)
            circuit_fnames.append(circuit_fname)
            kws_cp["circuit_file"] = backslash(os.path.join(self.remote_dir,kws_cp["circuit_file"]))
            all_params.append(kws_cp)
        # Copy all circuit files to server
        circuit_fnames_remote = self.put(circuit_fnames)
        # Write config file
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fconfig = "simconfig_" + now + ".csv"
        fconfig_local = os.path.join(self.local_dir,fconfig)
        comments = []
        comments.append("# To run manually: python %s %s --processes=<num>" %(fexec,fconfig))
        comments.append('#' + self.command + " -b ")
        with open(fconfig_local,'w') as f:
            f.write("".join([cmt + "\n" for cmt in comments]))
        df = pd.DataFrame(all_params)
        df.to_csv(fconfig_local,mode='a',index=False)
        self.put(fconfig)
        # Copy exec file
        self.put(fexec_orig,relative=False)
        self.get(fexec)
        return fconfig

    def get_results(self,fconfig,timeout=10,read_raw=False):
        """ Get simulation results from server

        timeout (seconds): Maximum time to wait until the simulation finishes
        read_raw: If True, import raw files into memory; otherwise, return filenames only
        """
        # First check if the simulation has finished
        t0 = time.time()
        t1 = time.time()
        fend = "finish_" + fconfig[:-4] + ".txt"
        fend_remote = backslash(os.path.join(self.remote_dir,fend))
        client = self.new_connection()
        while t1-t0 < timeout:
            flist = SSH_run(client,"ls %s" %self.remote_dir)
            if fend not in flist:
                time.sleep(10)
                t1 = time.time()
            else:
                break
        client.close()
        if fend not in flist:
            logging.error("Timeout: Simulation is not done yet. Try again later.")
            return None
        df = pd.read_csv(os.path.join(self.local_dir,fconfig),skiprows=2)
        fnames = np.array(df["output_file"])
        # Get output files from server
        self.get(fend)
        fnames_local = self.get(fnames,relative=False)
        if read_raw:
            results = [RawFile(fname,binary=True) for fname in fnames_local]
        else:
            results = fnames_local
        df["result"] = results
        return df

    def remove_fconfig(self,fconfig,dest="both"):
        """ Clean up the simulation files on local and remote locations

        dest: "local" or "remote" or "both"
        """
        fconfig_local = os.path.join(self.local_dir,fconfig)
        fconfig_remote = backslash(os.path.join(self.remote_dir,fconfig))
        # Get simulation file names
        df = pd.read_csv(fconfig_local,skiprows=2)
        circuit_files_remote = np.array(df["circuit_file"])
        circuit_files_local = [os.path.join(self.local_dir,os.path.basename(fname)) for fname in circuit_files_remote]
        output_files_remote = np.array(df["output_file"])
        output_files_local = [os.path.join(self.local_dir,os.path.basename(fname)) for fname in output_files_remote]
        fexec_local = os.path.join(self.local_dir,fexec)
        fexec_remote = backslash(os.path.join(self.remote_dir,fexec))
        fend = "finish_" + fconfig[:-4] + ".txt"
        fend_local = os.path.join(self.local_dir,fend)
        fend_remote = backslash(os.path.join(self.remote_dir,fend))
        # Start to clean up server files
        if dest.lower() in ["remote","both"]:
            client = self.new_connection()
            sftp = client.open_sftp()
            sftp.remove(fconfig_remote)
            sftp.remove(fexec_remote)
            sftp.remove(fend_remote)
            for fname in circuit_files_remote:
                sftp.remove(fname)
            for fname in output_files_remote:
                sftp.remove(fname)
            sftp.close()
            client.close()
        # Clean up local files
        if dest.lower() in ["local","both"]:
            os.remove(fconfig_local)
            os.remove(fexec_local)
            os.remove(fend_local)
            for fname in circuit_files_local:
                os.remove(fname)
            for fname in output_files_local:
                os.remove(fname)

    def remove_files(self,fnames,dest="both"):
        """ Clean up the files on local and remote locations

        dest: "local" or "remote" or "both"
        """
        if isinstance(fnames,str):
            fnames = [fnames]
        # Start to clean up server files
        if dest.lower() in ["remote","both"]:
            client = self.new_connection()
            sftp = client.open_sftp()
            fnames_remote = self.remote_fname(fnames)
            for fname in fnames_remote:
                sftp.remove(fname)
            sftp.close()
            client.close()
        # Clean up local files
        if dest.lower() in ["local","both"]:
            fnames_local = self.local_fname(fnames)
            for fname in fnames_local:
                os.remove(fname)

    def reshape_results(self,df,params):
        """ Reshape the results

        df: results DataFrame
        params: simulated script parameters
        """
        # Disintegrate the parameters (dict)
        iter_params = {}
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                # if param value is a list
                iter_params[k] = v
            else:
                kws[k] = v
        param_vals = list(itertools.product(*[iter_params[k] for k in iter_params.keys()]))

        dims = [len(v) for v in iter_params.values() if len(v)>1]
        data = np.array(df["result"]).reshape(dims)
        param_vals = np.array(param_vals).reshape(dims+[len(iter_params)]).T
        param_out = {}
        for i,pname in enumerate(iter_params.keys()):
            param_out[pname] = param_vals[i].T
        return param_out, data

    def run_parallel(self, *script, processes=16, save_file=False,reshape=True,read_raw=True, **params):
        """ Use multiprocessing to run in parallel

        script: WRspice script to be simulated.
        processes: number of parallel processes
        if reshape==False: return output data as a 1-dim array
        if read_raw==True: import raw file into memory
        """
        fconfig = self.prepare_parallel(*script,**params)
        fconfig_remote = backslash(os.path.join(self.remote_dir,fconfig))
        fexec_remote = backslash(os.path.join(self.remote_dir,fexec))
        # Simulate in parallel
        client = self.new_connection()
        cmd = "python %s %s --processes=%d" %(fexec_remote,fconfig_remote,processes)
        logging.info("Run on remote: %s" %cmd)
        SSH_run(client,cmd)
        client.close()
        # Get output files back to local
        df = self.get_results(fconfig,read_raw=read_raw)
        if df is None:
            return df
        # Delete files if necessary
        if not save_file:
            logging.debug("Remove temporary files")
            self.remove_fconfig(fconfig)
        if reshape:
            return self.reshape_results(df,params)
        else:
            return df

#=============================
def SSH_run(client,command):
    """ Run a command on SSH server """
    ssh_stdin, ssh_stdout, ssh_stderr = client.exec_command(command)
    err = [e.strip() for e in ssh_stderr.readlines()]
    out = [o.strip() for o in ssh_stdout.readlines()]
    if len(err)>0:
        print("Error:")
        print("\n".join(err))
    return out
