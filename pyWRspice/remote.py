# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Python wrapper for WRspice run on an SSH server
    class WRWrapper_SSH: Run WRspice script via WRspice simulator on an SSH server.

    Required package: paramiko to handle SSH connections
"""

import numpy as np
import os, tempfile, time
import uuid, itertools, logging
from multiprocessing import Pool
from paramiko.client import SSHClient

from simulation import RawFile, backslash

logging.basicConfig(level=logging.WARNING)

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
        else:
            self.remote_dir = remote_dir

    def new_connection(self):
        """ Make new SSH connection to the server """
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.server,username=self.login_user,password=self.login_pass)
        return ssh

    def get(self,fnames):
        """ Transfer file(s) from remote to local """
        ssh = self.new_connection()
        sftp = ssh.open_sftp()
        if isinstance(fnames,str):
            fnames = [fnames]
        # get local fnames
        local_fnames = []
        for fname in fnames:
            local_fname = os.path.join(self.local_dir,fname)
            sftp.get(backslash(os.path.join(self.remote_dir,fname)),local_fname)
            local_fnames.append(local_fname)
        sftp.close()
        ssh.close()
        if len(local_fnames)==1:
            return local_fnames[0]
        return local_fnames

    def put(self,fnames):
        """ Transfer file(s) from local to remote """
        ssh = self.new_connection()
        sftp = ssh.open_sftp()
        if isinstance(fnames,str):
            fnames = [fnames]
        # get remote fnames
        remote_fnames = []
        for fname in fnames:
            remote_fname = backslash(os.path.join(self.remote_dir,fname))
            sftp.put(os.path.join(self.local_dir,fname),remote_fname)
            remote_fnames.append(remote_fname)
        sftp.close()
        ssh.close()
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

    def run_parallel(self, *script, processes=16, save_file=False,reshape=True, **params):
        """ Use multiprocessing to run in parallel

        script: WRspice script to be simulated.
        processes: number of parallel processes
        if reshape==False: return output data as a 1-dim array
        """
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
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
        output_fnames = []
        circuit_fnames_local = []
        output_fnames_remote = []
        for i,vals in enumerate(param_vals):
            kws_cp = kws.copy()
            for pname,val in zip(iter_params.keys(), vals):
                kws_cp[pname] = val
            # Make sure they run separate script files
            if "circuit_file" not in kws_cp.keys():
                kws_cp["circuit_file"] = "tmp_circuit_" + '_'.join([str(val) for val in vals]) + ".cir"
            else:
                kws_cp["circuit_file"] = kws_cp["circuit_file"][:-4] + ''.join(['_'+str(val) for val in vals]) + ".cir"
            if "output_file" not in kws_cp.keys():
                kws_cp["output_file"] = "tmp_output_" + '_'.join([str(val) for val in vals]) + ".raw"
            else:
                kws_cp["output_file"] = kws_cp["output_file"][:-4] + ''.join(['_'+str(val) for val in vals]) + ".raw"
            circuit_fname, output_fname = self._render(self.script,kws_cp)
            circuit_fnames.append(circuit_fname)
            output_fnames.append(output_fname)
            circuit_fnames_local.append(os.path.join(self.local_dir,circuit_fname))
            output_fnames_remote.append(os.path.join(self.remote_dir,output_fname))

        # Copy all circuit files to server
        circuit_fnames_remote = self.put(circuit_fnames)
        # Simulate in parallel
        with Pool(processes=processes) as pool:
            results = []
            for fname in circuit_fnames_remote:
                results.append(pool.apply_async(self.run_file,(fname,)))
            results = [result.get() for result in results]
        # Get output files back to local
        output_fnames_local = self.get(output_fnames)
        # Extract data
        data = []
        for fname in output_fnames_local:
            data.append(RawFile(fname,binary=True))
        # Delete files if necessary
        if not save_file:
            client = self.new_connection()
            sftp = client.open_sftp()
            for ckt_fname_local, ckt_fname_remote, out_fname_local, out_fname_remote \
             in zip(circuit_fnames_local,circuit_fnames_remote,output_fnames_local,output_fnames_remote):
                os.remove(ckt_fname_local)
                os.remove(out_fname_local)
                sftp.remove(ckt_fname_remote)
                sftp.remove(out_fname_remote)
            sftp.close()
            client.close()
        if reshape:
            dims = [len(v) for v in iter_params.values() if len(v)>1]
            data = np.array(data).reshape(dims)
            param_vals = np.array(param_vals).reshape(dims+[len(iter_params)]).T
        else:
            data = np.array(data)
            param_vals = np.array(param_vals).T
        param_out = {}
        for i,pname in enumerate(iter_params.keys()):
            param_out[pname] = param_vals[i].T
        return param_out, data

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
