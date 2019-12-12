# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Python wrapper for WRspice

    class RawFile: Read and store WRspice output data into Variable structures.

    class WRWrapper: Run WRspice script via WRspice simulator.
"""

import numpy as np
import pandas as pd
import os, tempfile, time, datetime
import uuid, itertools, logging, subprocess
from multiprocessing import Pool

logging.basicConfig(level=logging.WARNING)

#------------------------------------------
# Wrapper for convenient parallel loops
#------------------------------------------

class WRWrapper:
    """ Wrapper for WRspice simulator.

    script: Declare the script with python format strings.
    '{output_file}' should be written by the script in the .control block.
    Any other keywords (which become mandatory) are added as named slots in the format string.

    source: WRspice .cir source file
    command: location of the wrspice exec file, depending on specific system:
    For Unix systems, it is likely "/usr/local/xictools/bin/wrspice"
    For Windows, it is likely "C:/usr/local/xictools/bin/wrspice.bat"
    """
    def __init__(self, script=None, source=None, command="/usr/local/xictools/bin/wrspice"):
        self.script    = script
        if source is not None:
            self.get_script(source)
        self.tmp_dir       = tempfile.TemporaryDirectory()
        self.command   = backslash(command)

    def _new_fname(self, suffix):
        """ Create a temporary file in the temporary folder """
        return os.path.join(self.tmp_dir.name, str(uuid.uuid4())+suffix)

    def get_script(self,fname):
        """ Get WRspice script from .cir file """
        with open(fname,'r') as f:
            lines = f.readlines()
        self.script = "".join(lines)

    def run(self,*script,**kwargs):
        """ Execute the script, return output data from WRspice

        script: (Optional) WRspice script to be simulated.
        kwargs: keyword arguments to be passed to self.script
        """
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
        if "circuit_file" not in kwargs.keys():
            kwargs["circuit_file"] = self._new_fname(".cir")
        if "output_file" not in kwargs.keys() or kwargs["output_file"] in [None, ""]:
            kwargs["output_file"] = self._new_fname(".raw")
        # Make sure the paths use backslashes
        kwargs["circuit_file"] = backslash(kwargs["circuit_file"])
        kwargs["output_file"] = backslash(kwargs["output_file"])
        rendered_script = self.script.format(**kwargs)
        run_script(rendered_script, save_file=True, fname=kwargs["circuit_file"], command=self.command)

        return RawFile(kwargs["output_file"], binary=True)

    def run_parallel(self, *script, processes=16, reshape=True, **params):
        """ Use multiprocessing to run in parallel

        script: (Optional) WRspice script to be simulated.
        processes: number of parallel processes
        if reshape==False: return output data as a 1-dim array
        """
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
        # params has param name and value list
        # if dim(param)==0 (string or scalar), not include in the iteration
        iter_params = {}
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                # if param value is a list
                iter_params[k] = v
            else:
                kws[k] = v
        param_vals = list(itertools.product(*[iter_params[k] for k in iter_params.keys()]))

        with Pool(processes=processes) as pool:
            results = []
            for i,vals in enumerate(param_vals):
                kws_cp = kws.copy()
                for pname,val in zip(iter_params.keys(), vals):
                    kws_cp[pname] = val
                logging.debug("Start to execute %d-th processes with parameters: %s" %(i+1,kws_cp))
                results.append(pool.apply_async(self.run, (self.script,), kws_cp))
            results = [result.get() for result in results]

        if reshape:
            dims = [len(v) for v in iter_params.values() if len(v)>1]
            results = np.array(results).reshape(dims)
            param_vals = np.array(param_vals).reshape(dims+[len(iter_params)]).T
        else:
            results = np.array(results)
            param_vals = np.array(param_vals).T
        param_out = {}
        for i,pname in enumerate(iter_params.keys()):
            param_out[pname] = param_vals[i].T

        return param_out, results

    def run_adaptive(self,*script,func=None,refine_func=None,max_num_points=100,processes=16,criterion="difference",**params):
        """ Run multiprocessing simulation witht adaptive repeats

        script: (Optional) WRspice script to be simulated.
        func: Function to calculate the desired output to be evaluated for repetition
        refine_func: Criterion function to determine the next points to run
        max_num_points: Maximum number of points
        """
        if func is None or refine_func is None:
            raise ValueError("Refine functions not determined.")
        if len(script)>0:
            # Assume the first argument is the script
            self.script = script[0]
        # params has param name and value list
        # if dim(param)==0 (string or scalar), not include in the iteration
        iter_params = {}
        kws = {}
        for k,v in params.items():
            if (not isinstance(v,str)) and hasattr(v,'__iter__'):
                iter_params[k] = v
            else:
                kws[k] = v
        new_points = np.array(list(itertools.product(*[v for v in iter_params.values()]))).flatten()
        num = int(len(new_points)/len(iter_params))
        new_points = new_points.reshape(num,len(iter_params))
        results_all = []
        points_all = None
        while num <= max_num_points:
            """ Execute the simulations in parallel """
            with Pool(processes=processes) as pool:
                results = []
                for i,vals in enumerate(new_points):
                    kws_cp = kws.copy()
                    for pname,val in zip(iter_params.keys(), vals):
                        kws_cp[pname] = val
                    logging.debug("Start to execute %d-th processes with parameters: %s" %(i+1,kws_cp))
                    results.append(pool.apply_async(self.run, (self.script,), kws_cp))
                results = [result.get() for result in results]
            results_all += results
            if points_all is None:
                points_all = new_points
            else:
                points_all = np.concatenate([points_all,new_points],axis=0)
            # import ipdb; ipdb.set_trace()
            new_points = refine_func(points_all,func(points_all,results_all),criterion=criterion)
            num += len(new_points)
        # Return results and points
        results_all = np.array(results_all)
        param_out = {}
        points = points_all.T
        for i,pname in enumerate(iter_params.keys()):
            param_out[pname] = points[i].T
        return param_out, results_all


#------------------------------
# Execute wrspice script
#------------------------------
def backslash(path):
    """ Convert the path to backslash-seperated one """
    p = str(path)
    ps = p.split('\\')
    return '/'.join(ps)


def run_file(fin,fout=None,display=False,command="wrspice",shell=False):
    """ Open a subprocess to run the script file, save data to fresult,
    execution output to fout (print out if display==True)
    if fout==None: not save output

    By default, the command is "wrspice". If it doesn't work,
    use the full location of the wrspice on your system by running "which wrspice" from the terminal
    For Unix systems, it is likely "/usr/local/xictools/bin/wrspice"
    For Windows, it is likely "C:/usr/local/xictools/bin/wrspice.bat"
    """
    cmd = backslash(command) + " -b {}".format(backslash(fin))
    if fout is not None:
        cmd = cmd + " -o {}".format(backslash(fout))
    logging.info("Run command: %s" %cmd)
    t1 = time.time()

    with subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, shell=shell,
                            stderr=subprocess.PIPE, env=os.environ.copy()) as process:
        t2 = time.time()
        proc_stds = process.communicate() # Get output messages
        proc_stdout = proc_stds[0].strip()
        msg = proc_stdout.decode('ascii')
        proc_stderr = proc_stds[1].strip()
        msg_err = proc_stderr.decode('ascii')
        if display:
            print(msg)
            print("Finished execution. Time elapsed: %.1f seconds" %(t2-t1))
        if len(msg_err)>0 :
            print("WRspice ERROR when running: %s" %fin)
            print(msg_err)
        logging.debug(msg)
        logging.info("Finished execution. Time elapsed: %.1f seconds" %(t2-t1))

def run_script(script,save_file=False,fname="tmp.cir",**kwargs):
    """ Write a text into a script file fname, then run it
    If save_file==True: do not remove fname after execution
    """
    fname = backslash(fname)
    with open(fname,'w') as f:
        f.write(script)
    # Execute file
    run_file(fname,**kwargs)
    if not save_file:
        os.remove(fname)
    else:
        logging.info("Save script into file: %s" %fname)


#------------------------------
# Import rawfile
#------------------------------

def read_lines(fname,num=None,start=0):
    """ Read the num of lines in the file.
    If num is None: read the entire file """
    with open(fname,'r') as f:
        if num is None:
            return f.readlines()
        else:
            lines = []
            for i in range(start):
                f.readline() # Discard
            for i in range(num):
                lines.append(f.readline())
            return lines

def search_key(lines,key,strict=False,get_value=False):
    """ Search through the list of text lines for the line containing the key
    if strict: match line == key
        else: line contains key
    if get_value==True, also return the value"""
    i = 0
    while True:
        if (strict and lines[i].strip()==key) or (not strict and lines[i].find(key)>-1):
            break
        else:
            i += 1
    line = lines[i]
    if get_value:
        try:
            val = float(line[line.rfind(':')+1:].strip())
            return i, val
        except:
            return i, None
    return i

def import_rawtext(fname):
    """ Import result rawfile (text format) into pandas DataFrame
    Depricated: use RawFile class to import rawfile instead
    """
    logging.info("Import file: %s" %fname)
    lines = read_lines(fname,num=100) # Assume we do not have more than 80 variables
    # Get the list of variables
    _, num_vari = search_key(lines,"No. Variables",get_value=True)
    num_vari = int(num_vari)
    vari_ind = search_key(lines,"Variables:",strict=True)+1
    logging.info("Number of variables: %d" %num_vari)
    logging.debug("Variables start from line #%d" %vari_ind)
    # Get variable names and units
    vari_names = []
    vari_units = []
    for i in range(vari_ind,vari_ind+num_vari):
        parts = lines[i].strip().split()
        vari_names.append(parts[1])
        vari_units.append(parts[2])
    msg = "Variables: "
    for name,unit in zip(vari_names,vari_units):
        msg += "%s [%s]; " %(name,unit)
    logging.info(msg)
    # Import values into DataFrame
    i = search_key(lines,'Values:') # Line from which values start
    data = pd.read_csv(fname,sep='\t',header=i)
    data_flat = data.values.flatten()
    num = len(data_flat)
    new_data = data_flat.reshape(int(num/num_vari),num_vari)
    logging.info("Imported data shape: %s" %str(new_data.shape))
    df = pd.DataFrame(new_data,columns=vari_names)
    return df

class Variable:
    """ To store a variable: name, index, unit and values """
    def __init__(self,name,unit,values=None,index=None):
        self.name = name
        self.unit = unit
        self.values = values
        self.index = index

    def __repr__(self):
        return "Variable <%s [%s]>" %(self.name,self.unit)

# class RawFile to handle output from WRspice
# some codes borrowed from PySpice NgSpice RawFile class

class RawFile:
    def __init__(self,fname,binary=True):
        """ Import a rawfile in binary (default) or text format """
        self.filename = fname
        logging.info("Import file: %s" %fname)
        if binary:
            self.import_binary()
        else:
            self.import_text()
        # Log the file header
        msg = "File header:\n"
        for k,v in self.header.items():
            msg += "%s:\t%s\n" %(k,v)
        logging.info(msg)

    def import_binary(self):
        """ Import binary rawfile """
        with open(self.filename,'rb') as f:
            stdout = f.read()
        binary_line = b'Binary:'# + os.linesep.encode('ascii')
        binary_location = stdout.find(binary_line)
        if binary_location < 0:
            raise ValueError('Cannot locate binary data')
        raw_data_start = binary_location + len(binary_line) + 1
        header = stdout[:binary_location].splitlines()
        # Start to read the header lines
        self.header_lines = []
        for line in header:
            self.header_lines.append(line.decode("ascii"))
        raw_data = stdout[raw_data_start:]
        self.header = self._read_header() # Keep the original header for debug purpose
        self.variables = self._read_header_variables()
        # Read the binary data into variable values
        self._read_binary_data(raw_data)

    def import_text(self):
        """ Import text rawfile
        Caution: Only for real numbers (not complex)
        """
        lines = read_lines(self.filename,num=300) # Assume we don't have more than 280 variables
        data_location = search_key(lines,'Values:') # Line from which data start
        if data_location < 0:
            raise ValueError('Cannot locate binary data')
        self.header_lines = lines[:data_location] # Keep the original header for debug purpose
        self.header = self._read_header()
        self.variables = self._read_header_variables()
        # Use pandas to read text file
        if self.flags=="real":
            data = pd.read_csv(self.filename,sep='\t',header=data_location)
            data_flat = data.values.flatten()
            raw_data = data_flat.reshape(int(len(data_flat)/self.number_of_variables),self.number_of_variables)
            for variable in self.variables:
                variable.values = raw_data[:,variable.index]
        else:
            raise NotImplementedError

    def _read_header(self):
        """ Get information from header lines """
        header = {}
        for line in self.header_lines:
            parts = line.split(':')
            if len(parts)>1:
                val = parts[1].strip()
                try:
                    val = float(val)
                except:
                    pass # Keep string format
                header[parts[0].strip()] = val
        self.flags = header["Flags"].lower()
        self.number_of_variables = int(header["No. Variables"])
        self.number_of_points = int(header["No. Points"])
        logging.info("Number of variables: %d" %self.number_of_variables)
        return header

    def _read_header_variables(self):
        """ Get information for the variables from header """
        variables = []
        vari_ind = search_key(self.header_lines,"Variables:",strict=True)+1
        for i in range(vari_ind, vari_ind+self.number_of_variables):
            line = self.header_lines[i]
            items = [x.strip() for x in line.split() if x]
            index, name, unit = items[:3]
            variables.append(Variable(name, unit, index=int(float(index))))
        self.header["Variables"] = variables
        msg = "Variables: "
        for vari in variables:
            msg += "%s [%s]; " %(vari.name,vari.unit)
        logging.info(msg)
        return variables

    def _read_binary_data(self, raw_data):
        """ Read the raw data and set the variable values.
        Codes borrowed from PySpice.
        """
        if self.flags == 'real':
            number_of_columns = self.number_of_variables
        elif self.flags == 'complex':
            number_of_columns = 2*self.number_of_variables
        else:
            raise NotImplementedError

        input_data = np.fromstring(raw_data, count=number_of_columns*self.number_of_points, dtype='f8')
        input_data = input_data.reshape((self.number_of_points, number_of_columns))
        input_data = input_data.transpose()
        if self.flags == 'complex':
            raw_data = input_data
            input_data = np.array(raw_data[0::2], dtype='complex64')
            input_data.imag = raw_data[1::2]
        for variable in self.variables:
            variable.values = input_data[variable.index]

    def to_df(self):
        """ Convert variables to pandas DataFrame object """
        data = {}
        for vari in self.variables:
            data[vari.name] = vari.values
        return pd.DataFrame(data)

    def to_array(self):
        """ Convert variables to numpy array """
        data = [vari.values for vari in self.variables]
        return np.array(data).T


#------------------------------
# Write vectors into rawfile
#------------------------------

def variables_to_text(data):
    """ Translate list of variables to header lines """
    num_vari = len(data)
    num_points = len(data[0].values)
    lines = []
    lines.append("Variables:")
    i = 0
    for datum in data:
        if datum.index is None:
            datum.index = i
            i += 1
        lines.append("%s\t%s\t%s" %(datum.index, datum.name, datum.unit))
    return lines

def values_to_text(data):
    """ Translate the values of a list of variables to text representation
    Currently only support real data
    """
    num_vari = len(data)
    num_points = len(data[0].values)
    lines = []
    for i in range(num_points):
        lines.append("%s\t%e" %(i,data[0].values[i]))
        for j in range(1,num_vari):
            lines.append(" \t%e" %(data[j].values[i]))
    return lines

def values_to_binary(data,flags="real"):
    """ Translate the values of a list of variables to a list of binary values """
    num_vari = len(data)
    num_points = len(data[0].values)
    if flags.lower()=="complex":
        num_cols = num_vari*2
        output = np.zeros((num_points,num_cols),dtype="f8")
        for i in range(num_vari):
            output[:,2*i] = [datum.real for datum in data[i].values]
            output[:,2*i+1] = [datum.imag for datum in data[i].values]
    else:
        num_cols = num_vari
        output = np.zeros((num_points,num_cols),dtype="f8")
        for i in range(num_vari):
            output[:,i] = data[i].values
    return output.flatten()

def write_rawfile(fname,data,binary=True,flags="real"):
    """ Write a list of variables to a rawfile
    Not yet support flags='complex' case
    """
    if flags!="real":
        raise NotImplementedError
    # Create a dummy header for the rawfile
    header_template = """Title: {title}
    Date: {date}
    Plotname: {plotname}
    Flags: {flags}
    No. Variables: {num_vari}
    No. Points: {num_points}
    Command: version 4.3.9"""
    title = "Piecewise"
    date = str(datetime.datetime.now())
    plotname = "Transient analysis"
    num_vari = len(data)
    num_points = len(data[0].values)
    header_text = header_template.format(title=title,date=date,plotname=plotname,
                    flags=flags,num_vari=num_vari,num_points=num_points)
    header = [x.strip() for x in header_text.split('\n')]
    header_sum = header + variables_to_text(data)
    logging.info("Write to file: %s" %fname)
    if binary:
        header_sum += ["Binary:"]
        lines = [line.encode("ascii") for line in header_sum]
        with open(fname,'wb') as f:
            sep = os.linesep.encode("ascii")
            f.write(sep.join(lines))
            f.write(sep)
            f.write(values_to_binary(data,flags))
    else:
        header_sum += ["Values:"]
        with open(fname,'w') as f:
            f.write("\n".join(header_sum))
            f.write("\n")
            f.write("\n".join(values_to_text(data)))
