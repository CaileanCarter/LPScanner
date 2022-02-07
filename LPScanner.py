import weakref
import re
from threading import Thread
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
# import pandas as pd
# import plotly.express as px

from ScrumPy.LP.glpks import exec_ll


#---------------------------------------------
# Handling tokenised input

CMDs = {
    "SCAN" :    "reac",
    "FROM" :    "start",
    "TO" :      "stop",
    "STEP" :    "step",
    "ID" :      "ID"
    }

NAME =      r"(?P<NAME>[\w|\-|_]{5,})"
ACTION =    r"(?P<ACTION>[A-Z]{2,4})"
NUM =       r"(?P<NUM>[-+]?\d*\.?\d+)"
WS =        r"(?P<WS>\s+)"

master_pattern = re.compile('|'.join([NAME, ACTION, NUM, WS]))
Token = namedtuple('Token', ['type', 'value'])


def parse_tokens(msg: str):
    scanner = master_pattern.scanner(msg)
    results = [Token(m.lastgroup, m.group()) for m in iter(scanner.match, None)]
    fil_result = list(filter(lambda x: x.type != 'WS', results)) 

    for i in range(0, len(fil_result)-1, 2):
        action, val = fil_result[i:i+2]
        assert action.type == "ACTION"
        assert val.type in ('NAME', 'NUM')
        yield action.value, val.value

#-------------------------------------------------------

def _new_instance(obj: object) -> object:
    class _obj(obj.__class__):
        def __init__(self) : 
            for attr, value in vars(obj).copy().items():
                setattr(self, attr, value)

            try:
                setattr(self, "glp_free", exec_ll("glp_free", self.lpx))
            except AttributeError:
                pass

    newcopy = _obj()
    newcopy.__class__ = obj.__class__
    return newcopy


class NullOptimalLPError(Exception):
    pass

class NullLPResult(Exception):
    pass

class ThreadAlreadyRunningError(Exception):
    pass

class JobHandlerError(Exception):
    pass


class LPScanner:

    "Perform a rapid LP scan"

    def __init__(self, reac, flux={}, start=0.1, stop=1.1, step=0.1, rev=False):
        self._flux_range = np.arange(start, stop, step)
        self.reverse = rev
        self.flux = flux
        self.reac = reac
        self.ID = None
        self.notes = None

        self._actors = {
            "lp_solve" :        self.lp_solve,
            "get_prim_sol" :    self.get_prim_sol,
            "set_fixed_flux" :  self.set_fixed_flux,
            "transform" :       self.transform
        }

        self._jobs = deque()
        self.lp_results = {}


    def send(self, msg: Tuple["Actor", "arg", "flux"]) -> None:
        actor = self._actors.get(msg[0])
        self._jobs.append(actor(*msg[1:]))

    
    def normalise_flux(self, i: float) -> float:
        i = np.round(i, 2)
        return i if not self.reverse else -i


    def set_fixed_flux(self, lp: object, f : dict, r : str, i :float):
        f[r] = i
        print(f"Setting fixed flux with {f}")
        lp.SetFixedFlux(f)
        yield self.send(("lp_solve", lp, i))
            

    def lp_solve(self, lp: object, i: float):
        print(f"solving LP for {i}")
        lp.Solve(PrintStatus=False)
        yield self.send(("get_prim_sol", lp, i))
        

    def get_prim_sol(self, lp: object, i: float):
        print(f"Getting primary solution for {i}")
        if lp.IsStatusOptimal():
            sol = lp.GetPrimSol()
        else:
            raise NullOptimalLPError
        yield self.send(("transform", sol, i))
        lp.glp_free()

    
    def transform(self, sol: dict, i: float):
        print(f"Transforming data for {i}")
        result = (
            np.array(tuple(sol.keys()), dtype=np.dtype(str)),
            np.array(tuple(sol.values()), dtype=np.dtype(float))            
            )
        self.lp_results[str(i)] = result
        yield
            

    def run(self, model: object):
        print("Getting LP...")
        reac = self.reac
        flux = self.flux

        _lp = model.GetLP()

        flux_range = deque(self._flux_range)
        lp_gen = [_new_instance(_lp) for _ in range(len(flux_range))]
        lps = deque((weakref.proxy(l) for l in lp_gen))

        while True:
            if flux_range:
                i = self.normalise_flux(flux_range.popleft())
                self.send((
                    "set_fixed_flux",
                    lps.popleft(),
                    flux,
                    reac, 
                    i
                    ))
            try:
                task = self._jobs.popleft()
                next(task)
            except IndexError:
                print("Job finished")
                self._lp_gen = lp_gen
                return
        


class JobHandler:

    def __init__(self, model: object):
        self._model = _new_instance(model)
        self.IDs = 1

        self._t = None
        self._jobs = deque()
        self.output = {}

    
    def __getitem__(self, attr):
        return self.__dict__["output"][attr]

    
    def __repr__(self):
        return f"{len(self._jobs)} jobs waiting to be run.\n{len(self.output)} jobs completed."
    

    def _executor(self):
        while self._jobs:
            ID, job = self._jobs.popleft()
            try:
                job.run(self._model)
            except Exception as e:
                raise JobHandlerError(f"JobHandler encountered an error with job ID {ID}.\nJobHandler has now stopped with all remaining jobs still in the queue.\nRun start again to continue.\n\n{e}")

            self.output[str(ID)] = job
        print("No more jobs left. Stopping JobHandler.")


    def start(self):
        if self._t and self._t.is_alive():
            raise ThreadAlreadyRunningError("Cannot start when JobHandler is still running.")
        self._t = Thread(target=self._executor)
        self._t.start()


    def submit(self, reac, ID="", notes="", **kwargs):
        lpscan = LPScanner(reac, **kwargs)

        if ID:
            lpscan.ID = ID 
        else:
            lpscan.ID = self.IDs
            self.IDs += 1

        lpscan.notes = notes
        self._jobs.append((ID, lpscan))


    def token_submit(self, msg: str, flux={}) -> None:

        """Send batch jobs to LPScanner as a token:

        INPUT:
            msg (str): SCAN [reaction] FROM [min] TO [max] STEP [step]
                       Declaring min, max and step flux are optional, 
                       defaults are stated in LPScanner.
            flux (dict): Flux constraints can be declared with flux argument.
        
        """

        form = {
            "notes" : "String input",
            "flux" : flux
            }
        reac = None

        for action, val in parse_tokens(msg):

            if action == "SCAN":
                reac = val
            elif action == "ID":
                form["ID"] = val
            else:
                val = float(val)
                if val < 0:
                    val = -val
                    form["rev"] = True
                act = CMDs[action]
                form[act] = val

        self.submit(reac, **form)
