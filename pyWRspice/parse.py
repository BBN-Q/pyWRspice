"""
    Parse a SPICE circuit script (WIP) into a script.Script object
"""

import numpy as np
import re
from . import script

def split_line(line):
    """ Split a line

    Treat part enclosed by parentheses as a single unit
    """
    line = re.sub('\s*=\s*','=',line)
    parts = []
    for p in re.split(',|\s',line):
        part = p.strip()
        if len(part)>0:
            parts.append(part)
    output = []
    i = 0
    # Deal with parentheses
    while i<len(parts):
        if parts[i].find('(') > -1:
            # Open parenthesis, need to find the closing one
            num_open = parts[i].count('(')
            num_close = parts[i].count(')')
            j = i
            while num_open>num_close:
                # Search until all parentheses are closed
                j += 1
                num_open += parts[j].count('(')
                num_close += parts[j].count(')')
            part = ' '.join(parts[i:j+1])
            i = j
        else:
            part = parts[i]
        output.append(part)
        i += 1
    return output

def parse_component(line):
    """ Parse a line of text into circuit component """
    # Split the line
    parts = split_line(line)
    name = parts[0]
    ports = []
    params= {}
    i = len(parts) - 1
    # Extract parameters
    while i>0 and parts[i].find('=')>-1:
        ps = parts[i].split('=')
        params[ps[0]] = ps[1]
        i -= 1
    value = parts[i]
    ports = parts[1:i]
    return script.Component(name,ports,value,params)

def parse_subckt_line(line):
    """ Parse a line of text into subcircuit declaration """
    # Split the line
    parts = split_line(line)
    name = parts[1]
    ports = []
    params= {}
    i = len(parts) - 1
    # Extract parameters
    while i>2 and parts[i].find('=')>-1:
        ps = parts[i].split('=')
        params[ps[0]] = ps[1]
        i -= 1
    ports = parts[2:i+1]
    return script.SubCircuit(name,script.Circuit(),ports,params)

def parse_model(line):
    """ Parse a line of text describing a model """
    # Split the line
    line = line.replace('(',' (')
    parts = split_line(line)
    name = parts[1]
    modtype = parts[2]
    # Extract parameters
    vals = ''.join(parts[3:])
    vals = vals[vals.find('(')+1:vals.rfind(')')]
    vals = re.split(',|\s',vals)
    params = {}
    for val in vals:
        vs = val.split('=')
        if len(vs)==2:
            params[vs[0]] = vs[1]
    return name, modtype, params

def get_subckt_block(lines,start=0):
    """ Get the outer most subcircuit block """
    num_open = 1
    num_close = 0
    for i in range(start+1,len(lines)):
        if lines[i][:7].lower()=="subckt":
            num_open += 1
        if lines[i][:5].lower()==".ends":
            num_close += 1
        if num_open==num_close:
            break
    return lines[start:i+1]

def parse_subckt(block,ckt):
    """ Parse a block of text describing a subcircuit """
    subckt = parse_subckt_line(block[0])
    i = 1
    while i<len(block)-1:
        line = block[i].strip()
        if len(line)==0 or line[0]=='*':
            pass # Ignore comments
        elif line[:6]=='.model':
            name, modtype, params = parse_model(line)
            subckt.circuit.add_model(name,modtype,**params)
        elif line[:7]==".subckt":
            blc = get_subckt_block(block,i)
            parse_subckt(blc,subckt.circuit)
            i = i + len(blc) - 1
        elif line[0]=='.':
            # Don't know what to do, store the original text
            subckt.circuit.add_control(line)
        else:
            subckt.circuit.add_component(parse_component(line))
        i += 1
    ckt.subcircuits[subckt.name] = subckt

class Parse(script.Script):
    """ Parse a SPICE script into a script.Script object

    scr: (optional) SPICE script
    """
    def __init__(self,scr=None):
        """ Parse a SPICE script into a script.Script object

        scr: (optional) SPICE script
        """
        super(Parse,self).__init__("")
        self.orig_script = scr
        if self.orig_script is not None:
            self.parse()

    def parse(self,scr=None):
        """ Parse a SPICE script
        Ignore comments in the script

        scr: (optional) SPICE script
        """
        if scr is not None:
            self.orig_script = scr
        if self.orig_script is None:
            raise ValueError("No SPICE script is specified")
        lines = self.orig_script.split('\n')
        self.title = lines[0]
        circuit = script.Circuit()
        i = 1
        while i<len(lines):
            line = lines[i].strip()
            if len(line)==0 or line[0]=='*':
                pass # Ignore comments
            elif line[0]=='.':
                if line[:5].lower() in [".exec",".cont",".post"]: # Control block
                    j = i
                    while lines[j][:5].lower()!=".endc" and j<len(lines):
                        j += 1
                    self.add_control('\n'.join(lines[i:j+1]))
                    i = j
                elif line[:6]=='.model':
                    name, modtype, params = parse_model(line)
                    circuit.add_model(name,modtype,**params)
                elif line[:7]==".subckt": # subcircuit
                    blc = get_subckt_block(lines,i)
                    parse_subckt(blc,circuit)
                    i = i + len(blc) - 1
                else:
                    # Don't know what to do, store the original text
                    circuit.add_script(line)
            else:
                circuit.add_component(parse_component(line))
            i += 1
        # Wrap up
        self.circuits.append(circuit)
