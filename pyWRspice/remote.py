# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Python wrapper for WRspice run on an SSH server
    class WRWrapperSSH: Run WRspice script via WRspice simulator on an SSH server.

    Required package: paramiko to handle SSH connections
"""

import numpy as np
import pandas as pd
import os, tempfile, time
from datetime import datetime
import uuid, itertools, logging
import multiprocessing as mp
from paramiko.client import SSHClient

from .simulation import RawFile, backslash

try:
    from adapt.refine import refine_scalar_field, refine_1D, well_scaled_delaunay_mesh
except:
    raise Exception("Could not import the 'adapt' package. Please install from github.com/bbn-q/adapt" )

logging.basicConfig(level=logging.WARNING)

# Get the run_parallel.py file
dir_path = os.path.dirname(os.path.realpath(__file__))
fexec = "run_parallel.py"
fexec_orig = os.path.join(dir_path,"data",fexec)

#------------------------------------------
# Wrapper for convenient parallel loops
#------------------------------------------

class WRWrapperSSH:
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
    def __init__(self, server, login_user, login_pass=None,
                local_dir=None, remote_dir=None,
                script=None, source=None,
                command="/usr/local/xictools/bin/wrspice"):
        self.server = server
        self.login_user = login_user
        if login_pass is not None:
            logging.warning("Please consider using a password-protected SSH key and ssh-agent rather than a plaintext password")

        self.login_pass = login_pass
        self.command   = backslash(command)

        self.script    = script
        if source is not None:
            self.get_script(source)
        # If local_dir is not specified, create one
        if local_dir is None:
            self.local_dir = backslash(tempfile.TemporaryDirectory().name)
        else:
            self.local_dir = backslash(local_dir)

        # Create local dir if necessary
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        # If remote_dir is not specified, create one
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
        """ Make new SSH connection to the server

        Note: It is a good practice to close the connection after use
        """
        logging.debug("Open a new SSH connection")
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.server,username=self.login_user,password=self.login_pass)
        return ssh

    def remote_fname(self,fname):
        """ Convert a base filename to remote filename by adding remote directory """
        if (not isinstance(fname,str)) and hasattr(fname,'__iter__'):
            # If fname is a list of file names
            return [backslash(os.path.join(self.remote_dir,name)) for name in fname]
        return backslash(os.path.join(self.remote_dir,fname))

    def local_fname(self,fname):
        """ Convert a base filename to local filename by adding local directory """
        if (not isinstance(fname,str)) and hasattr(fname,'__iter__'):
            # If fname is a list of file names
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
                local_fname = self.local_fname(fname)
                remote_fname = self.remote_fname(fname)
            else:
                local_fname = self.local_fname(os.path.basename(fname))
                remote_fname = backslash(fname)
            sftp.get(remote_fname,local_fname)
            local_fnames.append(local_fname)
        sftp.close()
        ssh.close()
        logging.debug("Retrieved files from remote: %s" %fnames)
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
                remote_fname = self.remote_fname(fname)
                local_fname = self.local_fname(fname)
            else:
                remote_fname = self.remote_fname(os.path.basename(fname))
                local_fname = fname
            sftp.put(local_fname,remote_fname)
            remote_fnames.append(remote_fname)
        sftp.close()
        ssh.close()
        logging.debug("Sent files to remote: %s" %fnames)
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

    def run_command(self,cmd):
        """ Run a command cmd on remote """
        client = self.new_connection()
        msg = SSH_run(client,cmd)
        client.close()
        return msg

    def run_file(self,fname_remote):
        """ Simulate a file on a server """
        cmd = self.command + " -b {}".format(fname_remote)
        logging.info("Run on remote: %s" %cmd)
        return self.run_command(cmd)

    def render(self,script,kwargs):
        """ Render a script by formatting it with kwargs
        then write into a local file

        Return (base) circuit and output file names
        """
        if "circuit_file" not in kwargs.keys():
            circuit_fname = self._new_fname("tmp_script_",".cir")
        else:
            circuit_fname = kwargs["circuit_file"]
        if "output_file" not in kwargs.keys() or kwargs["output_file"] in [None, ""]:
            output_fname = self._new_fname("tmp_output_",".raw")
        else:
            output_fname = kwargs["output_file"]
        kwargs["output_file"] = self.remote_fname(output_fname)
        rendered_script = script.format(**kwargs)
        with open(self.local_fname(circuit_fname),'w') as f:
            f.write(rendered_script)
        return circuit_fname, output_fname

    def run(self,script,display=False,save_file=False,**kwargs):
        """ Write a text into a script file fname on SSH server, then run it

        If save_file==True: save a local copy of the script file, do not remove fname after execution
        If 'circuit_file' and 'output_file' are not specified in kwargs: use randomly generated filenames

        If display: print output of simulation
        """
        self.script = script

        circuit_fname, output_fname = self.render(self.script,kwargs)
        kwargs["output_file"] = self.remote_fname(output_fname)
        # Copy the script file on the server
        circuit_fname_remote = self.put(circuit_fname)
        # Execute file
        msg = self.run_file(circuit_fname_remote)
        if display:
            for m in msg:
                print(m)
        # Get the result file back
        output_fname_local = self.get(output_fname)
        result = RawFile(output_fname_local, binary=True)
        if not save_file:
            logging.debug("Remove temporary files")
            os.remove(self.local_fname(circuit_fname))
            os.remove(output_fname_local)
            # Remove files on the server
            client = self.new_connection()
            sftp = client.open_sftp()
            sftp.remove(circuit_fname_remote)
            sftp.remove(kwargs["output_file"])
            # Close SFTP and SSH connection
            sftp.close()
            client.close()
        return result

    def prepare_parallel(self, script, use_outer_product=True, **params):
        """ Write script files on local and remote locations
        to prepare for the actual parallel simulation execution

        Return: a config file containing information of the simulation
        """
        self.script = script

        # Disintegrate the parameters (dict) into iterative and non-iterative parts
        iter_params = {} # iterative params
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                # if param value is a list
                iter_params[k] = v
            else:
                kws[k] = v

        if use_outer_product:
            param_vals = list(itertools.product(*[iter_params[k] for k in iter_params.keys()]))
        else:
            param_vals = np.vstack([v for v in iter_params.values()]).T

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

            circuit_fname, output_fname = self.render(self.script,kws_cp)
            circuit_fnames.append(circuit_fname)
            kws_cp["circuit_file"] = self.remote_fname(kws_cp["circuit_file"])
            all_params.append(kws_cp)
        # Copy all circuit files to server
        circuit_fnames_remote = self.put(circuit_fnames)
        # Write config file
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fconfig = "simconfig_" + now + ".csv"
        fconfig_local = self.local_fname(fconfig)
        comments = ''.join(["# To run manually: python %s %s --processes=<num>\n" %(fexec,fconfig),
        "#%s -b \n" %self.command])
        with open(fconfig_local,'w') as f:
            f.write(comments)
        df = pd.DataFrame(all_params)
        df.to_csv(fconfig_local,mode='a',index=False)
        # Copy files
        self.put(fconfig)
        self.put(fexec_orig,relative=False)
        self.get(fexec)
        return fconfig

    def get_results(self,fconfig,timeout=10,read_raw=False):
        """ Get simulation results from server

        fconfig: the config file generated by self.prepare_parallel
        timeout (seconds): Maximum time to wait until the simulation finishes
        read_raw: If True, import raw files into memory; otherwise, return filenames only
        """
        # First check if the simulation has finished
        t0 = time.time()
        t1 = time.time()
        fend = os.path.join(os.path.dirname(fconfig),"finish_" + os.path.basename(fconfig)[:-4] + ".txt")
        client = self.new_connection()
        while t1-t0 < timeout:
            flist = SSH_run(client,"ls %s" %self.remote_dir)
            if fend not in flist:
                time.sleep(2)
                t1 = time.time()
            else:
                break
        client.close()
        if fend not in flist:
            logging.error("Timeout: Simulation is not done yet. Try again later.")
            return None
        df = pd.read_csv(self.local_fname(fconfig),skiprows=2)
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

    def remove_files(self,fnames,dest="both"):
        """ Clean up the files on local and/or remote locations

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

    def remove_fconfig(self,fconfig,dest="both"):
        """ Clean up the simulation files on local and remote locations
        based on the information in the fconfig file

        dest: "local" or "remote" or "both"
        """
        # Get simulation file names
        df = pd.read_csv(self.local_fname(fconfig),skiprows=2)
        circuit_files = [os.path.basename(fname) for fname in df["circuit_file"]]
        output_files = [os.path.basename(fname) for fname in df["output_file"]]
        fend = os.path.join(os.path.dirname(fconfig),"finish_" + os.path.basename(fconfig)[:-4] + ".txt")
        # Remove all of them
        all_files = [fconfig,fend,fexec] + circuit_files + output_files
        self.remove_files(all_files,dest=dest)

    def reshape_results(self,df,params):
        """ Reshape the results

        df: results DataFrame as returned by self.get_results
        params: simulated script parameters
        """
        # Get iterative parameters
        iter_params = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                # if param value is a list
                iter_params[k] = v
        param_vals = list(itertools.product(*[iter_params[k] for k in iter_params.keys()]))

        dims = [len(v) for v in iter_params.values() if len(v)>1]
        data = np.array(df["result"]).reshape(dims)
        param_vals = np.array(param_vals).reshape(dims+[len(iter_params)]).T
        param_out = {}
        for i,pname in enumerate(iter_params.keys()):
            param_out[pname] = param_vals[i].T
        return param_out, data

    def run_parallel(self, script, processes=mp.cpu_count(), save_file=True,reshape=True,read_raw=True,use_outer_product=True, **params):
        """ Use multiprocessing to run in parallel on remote

        script: WRspice script to be simulated.
        processes: number of parallel processes
        if save_file==False: remove all relevant simulation files after execution (only if read_raw==True)
        if reshape==False: return output data as a pandas DataFrame
        if read_raw==True: import raw file into memory, otherise provide the list of output raw filenames
        if use_outer_product==True: run over all combinations of the iterable elements of kwargs 
        """
        fconfig        = self.prepare_parallel(script,use_outer_product=use_outer_product,**params)
        fconfig_remote = self.remote_fname(fconfig)
        fexec_remote   = self.remote_fname(fexec)
        # Simulate in parallel
        cmd = "python %s %s --processes=%d" %(fexec_remote,fconfig_remote,processes)
        logging.info("Run on remote: %s" %cmd)
        self.run_command(cmd)
        # Get output files back to local
        df = self.get_results(fconfig,read_raw=read_raw)
        if df is None:
            return df
        # Delete files if necessary
        if (not save_file) and read_raw:
            logging.debug("Remove temporary files")
            self.remove_fconfig(fconfig)
        if reshape:
            return self.reshape_results(df,params)
        else:
            return df

    def run_adaptive_2D(self,script,scalar_func,processes=mp.cpu_count(),
                    max_num_points=100,refine_kwargs={"criterion":"difference"},
                    **params):
        """ Run multiprocessing simulation with adaptive repeats

        script: WRspice script to be simulated.
        scalar_func: function to calculate the desired output to be evaluated for repetition, takes the current dictionary of parameters and the raw file object as positional arguments 
        max_num_points: Maximum number of points
        refine_kwargs: dictionary to be supplied as keyword arguments to the refine function (see bbnadapt documentation for details)
        """

        iter_params = {}
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                iter_params[k] = v
            else:
                kws[k] = v

        if len(iter_params) != 2:
            raise ValueError(f"Must have exactly two (not {len(iter_params)}) sweep axes for 2D adaptive sweep.")
        sweep_axes = list(iter_params.keys())

        # Run the initial batch of points
        # df = self.run_parallel(script, processes=processes, reshape=False, **params)

        # points_new  = df[sweep_axes].values

        points_new  = np.array(list(itertools.product(*[v for v in iter_params.values()]))).flatten()
        num         = int(len(points_new)/2)
        points_new  = points_new.reshape(num,len(iter_params))
        results_all = np.empty(0, dtype=object)
        points_all  = np.empty((0,2))
        scalars_all = np.empty(0)

        def apply_scalar_func(df):
            scalars = np.empty(len(df))
            # This is yucky, but we want the user to be able to use the same analysis functions
            # for both local and remote runs
            df['__scalars'] = df.apply(lambda r: scalar_func(r,r['result']), axis=1)
            return df['__scalars'].values

        while len(points_all) <= max_num_points:
            iter_params[sweep_axes[0]] = points_new[:,0]
            iter_params[sweep_axes[1]] = points_new[:,1]
            df = self.run_parallel(script, processes=processes, use_outer_product=False, reshape=False, **kws, **iter_params)
            
            results_new = df['result'].values

            results_all = np.concatenate([results_all, results_new]) # Accumulate all of the file references
            scalars_new = apply_scalar_func(df) # Caculate the new scalar values
            scalars_all = np.concatenate([scalars_all, scalars_new],axis=0) # Add the new scalar values to the existing scalar values
            points_all  = np.concatenate([points_all, points_new],axis=0) # Add the new points to the existing points
            points_new  = refine_scalar_field(points_all, np.array(scalars_all), **refine_kwargs) # Perform the refinement, find the new points
            
            print(f"Found {len(points_new)} new points.")



        # Return results and points
        # param_out = {}
        # points = points_all.T
        # for i,pname in enumerate(iter_params.keys()):
        #     param_out[pname] = points[i].T

        # Get the mesh for plotting and such
        mesh, scales, offsets = well_scaled_delaunay_mesh(points_all)
        for i in range(points_all.shape[1]):
            mesh.points[:,i] = mesh.points[:,i]/scales[i] + offsets[i]

        return results_all, points_all, scalars_all, mesh
        # return param_out, results_all, points_all, scalars_all, mesh

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
