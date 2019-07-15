# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:55:25 2019

@author: dbensaso
"""

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.tools.optimization import YawOptimizationWindRose
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd
import time
from scipy import interpolate
import math



##Calling on specific wind farm based on Wind_US_Database pickle
#sf= pd.read_pickle('Wind_US_Database')
#dk = (sf.loc[sf['p_name'] == 'Spruce Mountain'])

#Diameter and Rated power based on the wind farm
#D = dk["t_rd"]
#P_r = dk["t_cap"]
#T_Area = (math.pi* (D.iloc[0]**2)) /4

#C_p_rated = 0.43003137
#C_t_rated = 0.70701647

#Rated wind speed for any turbine
#U_turb_rated= (2* P_r.iloc[0]*(10**3)/ (C_p_rated * 1.225* T_Area))**(1/3)

#Making list of U/U_turb_rated//  Pickling can be done before hand 
#zf= pd.read_excel('Cp_look_up_table.xlsx', usecols= [0,1,2,3])  # doctest: +SKIP
#zf.to_pickle('Wind_Cp_look_up_table')
#tf= pd.read_pickle('Wind_Cp_look_up_table')

#Normalized wind speed for any turbine
#U_turb_norm =  tf.iloc[:,0] / U_turb_rated

def cp_for_any_turb(U_turb_norm)  :
    tf= pd.read_pickle('Wind_Cp_look_up_table')
    f = interpolate.interp1d(tf['U/Urated'], tf['C_p'], fill_value='extrapolate')
    _cp= []
    for i in U_turb_norm:
        if i < min(tf['U/Urated']):
            return 0.0 
        else:
            _p= f(i)
            if _p.size > 1:
                _p = _p[0]
            _cp.append(_p)
    return _cp

def ct_for_any_turb(U_turb_norm)  :
    tf= pd.read_pickle('Wind_Cp_look_up_table')
    f = interpolate.interp1d(tf['U/Urated'], tf['C_t'], fill_value='extrapolate')
    _ct= []
    for i in U_turb_norm:
        if i < min(tf['U/Urated']):
            return 0.0
        else:
            _t= f(i)
            if _t.size > 1:
                _t = _t[0]
            _ct.append(_t)
    return _ct
      
    
    