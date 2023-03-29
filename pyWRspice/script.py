# Copyright (c) 2019 Raytheon BBN Technologies - Quantum Group

"""
    Assist constructing a WRspice script
"""

import numpy as np
import networkx as nx
import pandas as pd
import logging, os
from pyWRspice.simulation import Variable, write_rawfile

# Get the style from NODE_STYLE.csv spreadsheet
dir_path = os.path.dirname(os.path.realpath(__file__))
NODE_STYLE = pd.read_csv(os.path.join(dir_path,"data/NODE_STYLE.csv"), index_col="initial")

class Component:
    """ General form of an electronic component """
    def __init__(self,name,ports=[],value=None,params={},comment=""):
        self.name = name
        self.ports = [str(p) for p in ports]
        self.value = value
        self.params = params
        self.comment = comment

    def script(self):
        """ Generate a WRspice script """
        if len(self.comment)>0:
            disp = ("* %s" %self.comment) + "\n"
        else:
            disp = ""
        disp += str(self.name) + ''.join([' '+str(p) for p in self.ports])
        if self.value is not None:
            disp += ' ' + str(self.value)
        if len(self.params.keys())>0:
            for k,v in self.params.items():
                disp += ' ' + k+'='+str(v)
        return disp

    def __repr__(self):
        return self.script()

class Node:
    """ Electrical element or connection node"""
    existing_nodes = []
    def __init__(self,name=None,attrs={}):
        """ attrs is a dictionary """
        self.name = name
        self.attrs = attrs
        if name is not None and name not in self.existing_nodes:
            self.existing_nodes.append(name)

    @classmethod
    def from_component(self,comp):
        """ Convert a component into a node """
        name = comp.name
        attrs = comp.params.copy()
        attrs["ports"] = comp.ports
        attrs["value"] = comp.value
        return Node(name,attrs)

    @classmethod
    def from_port(self,num):
        if num=='0':
            # Create a separate ground node
            i = 0
            while "GND"+str(i) in self.existing_nodes:
                i += 1
            name = "GND"+str(i)
            attrs = {}
        else:
            name = str(num)
            attrs = {}
        return Node(name,attrs)

class Circuit:
    """ Class to handle circuit graph and list of components """
    def __init__(self,script=""):
        self.components = {}
        self.subcircuits = {}
        self.models = {}
        self._extrascript = script
        self.params = {}
        self.waveforms = []

    def get_params(self):
        """ Get parameters from WRspice template """
        template = self.script()
        idxb = [i for i,c in enumerate(template) if c=='{']
        idxe = [i for i,c in enumerate(template) if c=='}']
        for i in range(min(len(idxb),len(idxe))):
            key = template[idxb[i]+1:idxe[i]]
            if key not in self.params.keys():
                self.params[key] = ""
        return self.params

    def add(self, comp):
        """Alias for add_component and add_components"""
        if isinstance(comp, list):
            self.add_components(comp)
        else:
            self.add_component(comp)

    def add_waveforms(self, wfms, filename="waveforms.raw"):
        wfm_vars = [Variable("tm", "s", values=wfms[0].time)]
        idx = 0
        for wfm in wfms:
            if wfm.name is None:
                wfm.name = f"wfm{idx:d}"
                idx +=1
            wfm_vars.append(Variable(wfm.name, "", values=wfm.wfm))
        write_rawfile(filename, wfm_vars)
        self.waveforms = wfms
        self.wfm_file = filename

    def add_component(self,comp):
        """ Add an instance of Component() to the circuit """
        if comp.name in self.components.keys():
            logging.warning("Ignore adding component. Already added: %s" %str(comp.name))
        else:
            self.components[comp.name] = comp

    def add_components(self,comps):
        """ Add a list of Component instances """
        for comp in comps:
            self.add_component(comp)

    def create_subcircuit(self,ckt,name,ports,params={}):
        """Create a SubCircuit from a Circuit instance"""
        subckt = SubCircuit(name,ckt,ports,params)
        self.add_subcircuit(subckt)

    def add_subcircuit(self, subckt):
        """Create a SubCircuit from a Circuit instance"""
        if subckt.name in self.subcircuits:
            raise Exception(f"Subcircuit {name} is already in this circuit")
        self.subcircuits[subckt.name] = subckt

    def add_model(self,name,modtype,**params):
        model = {"name": name, "type": modtype, "params": params}
        self.models[name] = model

    def add_script(self,script):
        self._extrascript += "\n" + str(script)

    def script(self):
        """ Generate a WRspice script for the circuit """
        text = []

        # Load waveforms
        if self.waveforms:
            par_string = " ".join([f"constants.{wfm.name}={wfm.name}" for wfm in self.waveforms])
            text.append(
            f".exec\n"
            f"load {self.wfm_file}\n"
            f"let constants.tm=tm {par_string}\n"
            f".endc"
            )

        # Declare all subcircuits
        for name,subckt in self.subcircuits.items():
            text.append(subckt.script())
        # Declare models
        for name, model in self.models.items():
            line = [".model ", name, ' ',model["type"],'(']
            pams = []
            for k,v in model["params"].items():
                pams.append("%s=%s" %(str(k),str(v)))
            line.append(','.join(pams))
            line.append(')')
            text.append(''.join(line))
        # Declare components
        for k,comp in self.components.items():
            text.append(comp.script())
        # Extra scripts
        if len(self._extrascript)>0:
            text.append(self._extrascript)
        return "\n".join(text)

    def _get_key(self,node):
        label = str(node)
        if label[:3].upper()=="GND":
            return "GND"
        elif label[:4].upper()=="PORT":
            return "PORT"
        else:
            return label[0].upper()

    def plot(self,show_value=False,node_shape='o',font_weight="bold", pos_func=nx.kamada_kawai_layout, **kwargs):
        """ Plot the circuit schematic using networkx and matplotlib """
        # Set up a graph
        graph = nx.Graph()
        for k,comp in self.components.items():
            nodes = []
            nodes.append(Node.from_component(comp))
            for p in comp.ports:
                if p=='0':
                    nodes.append(Node.from_port(p))
                else:
                    nodes.append(Node.from_port("1_"+str(p)))
            for node in nodes:
                graph.add_node(node.name,**node.attrs)
            for i in range(1,len(nodes)):
                graph.add_edge(nodes[0].name,nodes[i].name)
        # Set up plotting style
        nodes = graph.nodes
        # Get node style
        node_size = [NODE_STYLE["size"][self._get_key(node)] for node in nodes]
        node_color = [NODE_STYLE["color"][self._get_key(node)] for node in nodes]
        # Get node label
        node_label = {}
        for node in nodes:
            label = str(node)
            if label[:3].upper()=="GND":
                label = '0'
            elif label[:2]=="1_":
                label = label[2:]
            if show_value and ("value" in nodes[node].keys()):
                val = nodes[node]["value"]
                if val not in [None,""]:
                    try:
                        # If val is numerical, convert to scientific representation
                        val = float(val)
                        val = "%.3e" %val
                    except:
                        val = str(val)
                    label += ' (' + val + ')'
            node_label[node] = label
        # Position functions
        pos = pos_func(graph) if pos_func else None
        # Plot
        nx.draw_networkx(graph,arrows=False,with_labels=True,
                        nodelist = nodes,
                        labels = node_label,
                        node_size = node_size,
                        node_color = node_color,
                        node_shape = node_shape,
                        font_weight = font_weight,
                        pos = pos,
                        **kwargs)

    def reset(self):
        self.components = {}
        self.subcircuits = {}


class SubCircuit:
    """ Convert a Circuit object to subcircuit """
    def __init__(self,name,circuit,ports,params={}):
        self.name = name
        self.circuit = circuit
        self.ports = [str(p) for p in ports]
        self.params = params

    def plot(self,**kwargs):
        """ Plot the circuit and indicate the ports """
        ckt = Circuit()
        ckt.components = self.circuit.components.copy()
        for i,p in enumerate(self.ports):
            ckt.add_component(Component("PORT %d"%(i+1),[p]))
        ckt.plot(**kwargs)

    def script(self):
        line = ".subckt " + self.name + "".join([' '+str(p) for p in self.ports])
        for k,v in self.params.items():
            line += ' ' + k + '=' + str(v)
        return "\n".join([line,self.circuit.script(),".ends "+self.name])

    def __repr__(self):
        line = ".subckt " + self.name + "".join([' '+str(p) for p in self.ports])
        for k,v in self.params.items():
            line += ' ' + k + '=' + str(v)
        return line


class Script:
    """ To assist writing WRspice script and run simulation """
    def __init__(self,title):
        self.title = title
        self.circuits = []
        self.analysis = ""
        self.controls = []
        self.params = {}
        self.save_ports = ""
        self.save_file = None
        self.save_type = "binary"
        self.options = {}

    def add_circuit(self,ckt):
        self.circuits.append(ckt)

    def add_options(self,**options):
        """ Add simulation options """
        for k,v in options.items():
            self.options[k] = v

    def add_control(self,ctrl):
        self.controls.append(str(ctrl))

    def config_save(self,ports,filename=None,filetype="binary"):
        """ Specify what and how to save data """
        self.save_ports = ports
        self.save_file = filename
        self.save_type = filetype

    def _save_portname(self,port):
        """ Convert a port name to port index """
        port = str(port)
        if port[0].lower() in ['i','v']:
            return port
        else:
            return 'v(' + str(port) + ')'

    def _save_block(self):
        """ Compose a control block specifying saving config """
        if len(self.save_ports)==0:
            return ""
        if self.save_file is not None:
            self.params["output_file"] = self.save_file
        lines = [".control", "run"]
        lines.append("set filetype=%s" %self.save_type)
        line = "write {output_file} "
        if (not isinstance(self.save_ports,str)) and hasattr(self.save_ports,'__iter__'):
            line += " ".join([self._save_portname(p) for p in self.save_ports])
        else:
            line += self._save_portname(self.save_ports)
        lines.append(line)
        lines.append(".endc")
        return "\n".join(lines)

    def script(self):
        """ Return a WRspice script """
        # Get all the ports
        text = ["*"+self.title, self.analysis]
        text += [ckt.script() for ckt in self.circuits]
        text += self.controls
        if len(self.options.keys())>0:
            options = ".options"
            for k,v in self.options.items():
                options += " {}={}".format(k,v)
            text.append(options)
        text.append(self._save_block())
        return "\n".join(text)

    def get_params(self):
        """ Get parameters from WRspice template """
        template = self.script()
        idxb = [i for i,c in enumerate(template) if c=='{']
        idxe = [i for i,c in enumerate(template) if c=='}']
        for i in range(min(len(idxb),len(idxe))):
            key = template[idxb[i]+1:idxe[i]]
            if key not in self.params.keys():
                self.params[key] = ""
        return self.params

    def set_params(self,**kwargs):
        """ A more convenient way to set a bunch of params """
        for k,v in kwargs.items():
            self.params[k] = v

    def _combine_circuit(self):
        """ Combine all the circuits into one """
        cir_all = Circuit()
        for ckt in self.circuits:
            for k,comp in ckt.components.items():
                cir_all.add_component(comp)
        return cir_all

    def plot(self,**kwargs):
        """ Plot the combined circuit """
        cir_all = self._combine_circuit()
        cir_all.plot(**kwargs)
